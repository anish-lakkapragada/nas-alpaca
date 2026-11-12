"""Bounded, auditable four-GPU search in the original 27M-parameter frame.

All trials use the unchanged 5000-step recipe and only train/dev data. Start
with --plan-only to review the breadth queue. The scheduler never terminates
cloud instances: the enclosing rental controller owns that independent timer.
"""
from __future__ import annotations

from nas.paths import PROJECT_ROOT, PROVENANCE_SUPPORT, python_command, source_path

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import statistics
import subprocess
import sys
import time

ROOT = PROJECT_ROOT
FRAME = {"vocab_size": 32768, "max_len": 1024, "n_layers": 8,
         "d_model": 384, "n_heads": 6, "d_ff": 1536, "dropout": 0.1}
BASELINE_PARAMETERS = 27_144_960
RECIPE = {"steps": 5000, "batch_rows": 8, "lr": .001, "warmup": 100,
          "min_lr_frac": .1, "weight_decay": .1, "clip": 1., "eval_every": 250}
PROTOCOL = {
    "objective": "Minimize dev answer-token cross entropy; the final winner must have strictly fewer total parameters than the original baseline.",
    "frame": FRAME, "baseline_parameters": BASELINE_PARAMETERS, "recipe": RECIPE,
    "optimizer": "AdamW without bias correction, betas .9/.95, epsilon 1e-8; exact MLX-compatible update.",
    "allowed": ["layer loops with optional input injection", "nonadjacent attention sharing including LayerNorm",
                "nonadjacent FFN sharing including LayerNorm", "approved elementwise activations",
                "attention deletion", "FFN deletion", "FFN transpose tying", "FFN narrowing",
                "fixed or learned scalar on an existing residual branch", "FFN-before-attention layer order"],
    "screen_steps": 5000, "screen_seed": 0, "confirmation_seeds": [1, 2],
    "final_seeds": [10, 11, 12], "final_candidates": 2,
    "selection_metric": "Best dev loss from the unchanged every-250-step evaluations; all trials have K=5000.",
    "confirmation": "Use seeds 0/1/2 to select and freeze two reduced architectures, then report paired fresh-seed 10/11/12 baseline/candidate results without selecting again on those fresh seeds.",
    "adaptation": "Deterministic breadth, then balanced mutations of current low-loss parents including reserved reduced-parameter parents; stop on the time/architecture limits or sustained lack of meaningful improvement.",
    "validation": "Never read by this scheduler. Final held-out evaluation happens separately after the architecture is frozen.",
    "parallelism": "One subprocess per physical GPU, with OS-level GPU locks.",
    "provenance": "Append-only architecture and run event registries include source/spec hashes, seeds, UTC times, physical GPU, host, checkpoint hashes, metrics, and failure/interruption reasons.",
    "retention": "Never delete logs/specs/metrics. Preserve baseline/final checkpoints and leading seed-0/multiseed checkpoints; prune other checkpoint files only after their hashes enter the ledger if disk limits require it.",
    "budget": "Cloud shutdown is enforced by the external rental controller. This scheduler enforces training cutoff, per-job timeouts, and conservative job admission.",
}


def utc_now():
    return datetime.now(timezone.utc)


def iso(value):
    return value.astimezone(timezone.utc).isoformat()


def parse_utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("a timezone is required, e.g. 2026-11-11T16:15:00Z")
    return parsed.astimezone(timezone.utc)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(spec):
    """Canonical executable graph: rename IDs, drop unused options and defaults."""
    maps = {"attn": {}, "ffn": {}}
    layers, ffns = [], {}
    for given in spec["layers"]:
        layer = {}
        for kind in ("attn", "ffn"):
            key = given.get(kind)
            if key is None:
                layer[kind] = None
            else:
                if key not in maps[kind]:
                    maps[kind][key] = ("A" if kind == "attn" else "F") + str(len(maps[kind]))
                layer[kind] = maps[kind][key]
                if kind == "ffn":
                    opts = spec.get("ffns", {}).get(key, {})
                    ffns[layer[kind]] = {"hidden": opts.get("hidden", FRAME["d_ff"]),
                                         "act": opts.get("act", "gelu"),
                                         "transpose_tie": opts.get("transpose_tie", False)}
            scale = given.get(kind + "_scale", 1.)
            if isinstance(scale, dict):
                scale = {"init": float(scale["init"]), "learned": scale["learned"]} if scale["learned"] else float(scale["init"])
            else:
                scale = float(scale)
            if key is not None:
                layer[kind + "_scale"] = scale
        layer.update(loop=given.get("loop", 1), inject=given.get("inject", False),
                     order=given.get("order", "attn_ffn") if all(given.get(kind) is not None for kind in ("attn", "ffn")) else "attn_ffn")
        layers.append(layer)
    return {"layers": layers, "ffns": ffns}


def architecture_hash(spec):
    return hashlib.sha256(json.dumps(canonical(spec), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def learned_scale_count(spec):
    return sum(isinstance(layer.get(kind + "_scale"), dict) and layer[kind + "_scale"].get("learned") is True
               for layer in spec["layers"] for kind in ("attn", "ffn"))


def parameter_count(spec):
    d = FRAME["d_model"]
    total = (FRAME["vocab_size"] + FRAME["max_len"]) * d + 2 * d
    attentions = {layer.get("attn") for layer in spec["layers"]} - {None}
    ffns = {layer.get("ffn") for layer in spec["layers"]} - {None}
    total += len(attentions) * (4 * d * d + 2 * d)
    for key in ffns:
        opts = spec.get("ffns", {}).get(key, {})
        total += d * opts.get("hidden", FRAME["d_ff"]) * (1 if opts.get("transpose_tie") else 2) + 2 * d
    return total + learned_scale_count(spec)


def validate_spec(spec):
    layers = spec.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("nonempty layers required")
    applications = 0
    for i, layer in enumerate(layers):
        if set(layer) - {"attn", "ffn", "loop", "inject", "order", "attn_scale", "ffn_scale"}:
            raise ValueError("unapproved layer field")
        if layer.get("attn") is None and layer.get("ffn") is None:
            raise ValueError("both branches deleted")
        repeats = layer.get("loop", 1)
        if type(repeats) is not int or repeats < 1:
            raise ValueError("invalid loop count")
        applications += repeats
        if type(layer.get("inject", False)) is not bool or (layer.get("inject") and repeats == 1):
            raise ValueError("invalid injection")
        if layer.get("order", "attn_ffn") not in ("attn_ffn", "ffn_attn"):
            raise ValueError("invalid branch order")
        for kind in ("attn", "ffn"):
            key = layer.get(kind)
            if key is not None and (not isinstance(key, str) or not key or "." in key):
                raise ValueError("invalid sublayer ID")
            if i and key is not None and layers[i - 1].get(kind) == key:
                raise ValueError("adjacent sharing must be expressed as a loop")
            if kind + "_scale" in layer:
                if key is None:
                    raise ValueError("scale on deleted branch")
                scale = layer[kind + "_scale"]
                if isinstance(scale, dict):
                    if set(scale) != {"init", "learned"} or type(scale["learned"]) is not bool:
                        raise ValueError("invalid scale schema")
                    scale = scale["init"]
                if isinstance(scale, bool) or not isinstance(scale, (float, int)) or not math.isfinite(scale):
                    raise ValueError("scale must be finite")
    if applications > FRAME["n_layers"]:
        raise ValueError("too many layer applications")
    for opts in spec.get("ffns", {}).values():
        if set(opts) - {"hidden", "act", "transpose_tie"}:
            raise ValueError("unapproved FFN field")
        if opts.get("act", "gelu") not in ("gelu", "relu", "relu2", "silu", "gelu_tanh"):
            raise ValueError("unapproved activation")
        if type(opts.get("hidden", 1536)) is not int or not 1 <= opts.get("hidden", 1536) <= 1536:
            raise ValueError("FFN widening prohibited")
        if type(opts.get("transpose_tie", False)) is not bool:
            raise ValueError("invalid transpose flag")
    if parameter_count(spec) > BASELINE_PARAMETERS + learned_scale_count(spec):
        raise ValueError("structural parameter ceiling exceeded")


def baseline_spec():
    return {"name": "baseline", "description": "Original 8-layer 27,144,960-parameter GELU baseline",
            "layers": [{"attn": f"A{i}", "ffn": f"F{i}"} for i in range(8)], "ffns": {}}


def used_ffns(spec):
    return list(dict.fromkeys(layer["ffn"] for layer in spec["layers"] if layer.get("ffn") is not None))


def activation_spec(act="gelu"):
    spec = baseline_spec()
    spec["ffns"] = {key: {"act": act} for key in used_ffns(spec)}
    return spec


def set_width(spec, width, keys=None):
    for key in keys if keys is not None else used_ffns(spec):
        spec.setdefault("ffns", {}).setdefault(key, {})["hidden"] = width


def remove_branch(layer, kind):
    layer[kind] = None
    layer.pop(kind + "_scale", None)


def candidates():
    """Isolated primitives, positional probes, and conservative combinations."""
    specs, seen = {}, set()

    def add(name, spec, description):
        validate_spec(spec)
        digest = architecture_hash(spec)
        if digest in seen:
            return
        seen.add(digest)
        spec["name"], spec["description"] = name, description
        specs[name] = spec

    add("baseline", baseline_spec(), "Original approved baseline")
    add("relu2", activation_spec("relu2"), "ReLU squared everywhere; unchanged-count loss anchor")
    spec = activation_spec("relu2"); set_width(spec, 1472)
    add("relu2-narrow1472", spec, "Strong prior activation with a modest all-FFN parameter cut")
    spec = activation_spec("relu2")
    for layer in spec["layers"]:
        layer["ffn_scale"] = {"init": 1., "learned": True}
    set_width(spec, 1472)
    add("relu2-narrow1472-learned-ffn", spec, "Modest cut and learned FFN residual scalars")
    for act in ("relu", "silu", "gelu_tanh"):
        add(act, activation_spec(act), f"Isolated activation replacement: {act}")
    regions = {"early": (0, 1, 2), "middle": (3, 4), "late": (5, 6, 7)}
    pairs = [(0, 2), (0, 4), (0, 7), (1, 5), (2, 6), (3, 7)]
    for act in ("gelu", "relu2"):
        for width in (1472, 1408, 1280, 1152, 1024, 768):
            spec = activation_spec(act); set_width(spec, width)
            add(f"{act}-narrow{width}", spec, f"All FFNs narrowed to {width}")
        for region, indices in regions.items():
            for width in (1280, 1024):
                spec = activation_spec(act); set_width(spec, width, [f"F{i}" for i in indices])
                add(f"{act}-{region}-width{width}", spec, f"Narrow {region} FFNs only")
        for kind in ("attn", "ffn"):
            for index in range(8):
                spec = activation_spec(act); remove_branch(spec["layers"][index], kind)
                add(f"{act}-delete-{kind}{index}", spec, f"Delete just {kind} at layer {index}")
        for kind in ("attn", "ffn", "both"):
            for source, target in pairs:
                spec = activation_spec(act)
                for branch in (("attn", "ffn") if kind == "both" else (kind,)):
                    spec["layers"][target][branch] = spec["layers"][source][branch]
                add(f"{act}-share-{kind}{source}-{target}", spec, "Nonadjacent sharing includes the associated LayerNorm")
        for position in (0, 3, 6):
            for inject in (False, True):
                spec = activation_spec(act)
                del spec["layers"][position + 1]
                spec["layers"][position].update(loop=2, inject=inject)
                add(f"{act}-loop{position}" + ("-inject" if inject else ""), spec, "Replace two adjacent layers by one layer applied twice")
        for unique in (4, 2):
            for inject in (False, True):
                spec = activation_spec(act)
                spec["layers"] = [{"attn": f"A{i}", "ffn": f"F{i}", "loop": 8 // unique, "inject": inject} for i in range(unique)]
                add(f"{act}-loop{unique}x{8 // unique}" + ("-inject" if inject else ""), spec, "Reserved large parameter-cut regularization candidate")
        spec = activation_spec(act)
        spec["layers"] = [{"attn": f"A{i % 4}", "ffn": f"F{i % 4}"} for i in range(8)]
        add(f"{act}-cycle4", spec, "Cycle four shared blocks twice")
        for index in range(8):
            spec = activation_spec(act); spec["ffns"][f"F{index}"]["transpose_tie"] = True
            add(f"{act}-transpose{index}", spec, "Transpose-tie a single FFN")
        spec = activation_spec(act)
        for options in spec["ffns"].values():
            options["transpose_tie"] = True
        add(f"{act}-transpose-all", spec, "All FFNs transpose-tied; isolated broad cut")
        for kind in ("attn", "ffn", "both"):
            for learned in (False, True):
                for value in ((.5, math.sqrt(.5), 1.25) if not learned else (.1, .5, 1.)):
                    spec = activation_spec(act)
                    for layer in spec["layers"]:
                        for branch in (("attn", "ffn") if kind == "both" else (kind,)):
                            layer[branch + "_scale"] = {"init": value, "learned": True} if learned else value
                    suffix = f"{value:.3g}".replace(".", "p")
                    add(f"{act}-{'learned' if learned else 'fixed'}-{kind}-{suffix}", spec, "Scale existing residual branches; no new branch or normalization")
        for label, indices in {"all": range(8), "even": (0, 2, 4, 6), "odd": (1, 3, 5, 7), "early": range(4), "late": range(4, 8)}.items():
            spec = activation_spec(act)
            for index in indices:
                spec["layers"][index]["order"] = "ffn_attn"
            add(f"{act}-reverse-{label}", spec, "Apply existing FFN before existing attention")
        for index in range(8):
            spec = activation_spec(act); spec["layers"][index]["order"] = "ffn_attn"
            add(f"{act}-reverse{index}", spec, "Reverse one layer's branch order")
    for act in ("relu2", "silu", "gelu_tanh"):
        for region, indices in regions.items():
            spec = activation_spec("gelu")
            for index in indices:
                spec["ffns"][f"F{index}"]["act"] = act
            add(f"mixed-{act}-{region}", spec, "Regional activation replacement")
    for width in (1472, 1280):
        for kind in ("attn", "ffn"):
            for learned in (False, True):
                spec = activation_spec("relu2"); set_width(spec, width)
                for layer in spec["layers"]:
                    layer[kind + "_scale"] = {"init": 1., "learned": True} if learned else .5
                add(f"relu2-width{width}-{'learned' if learned else 'half'}-{kind}", spec, "Combine modest narrowing and residual scaling")
        spec = activation_spec("relu2"); set_width(spec, width)
        for layer in spec["layers"]:
            layer["order"] = "ffn_attn"
        add(f"relu2-width{width}-reverse", spec, "Combine modest narrowing and reversed branch order")
    return specs


def mutations(parent, round_index):
    """A balanced set of single edits around an already measured architecture."""
    options = []

    def offer(label, spec):
        try:
            validate_spec(spec)
        except (ValueError, TypeError, KeyError):
            return
        options.append((label, spec))

    keys = used_ffns(parent)
    for key in keys:
        for act in ("gelu", "relu2", "silu", "gelu_tanh"):
            spec = copy.deepcopy(parent); spec.setdefault("ffns", {}).setdefault(key, {})["act"] = act
            offer(f"act-{key}-{act}", spec)
        width = parent.get("ffns", {}).get(key, {}).get("hidden", 1536)
        for narrowed in sorted({max(64, width - 64), max(64, width - 256), min(width, 1024)}):
            spec = copy.deepcopy(parent); set_width(spec, narrowed, [key])
            offer(f"width-{key}-{narrowed}", spec)
        spec = copy.deepcopy(parent)
        options_for_key = spec.setdefault("ffns", {}).setdefault(key, {})
        options_for_key["transpose_tie"] = not options_for_key.get("transpose_tie", False)
        offer(f"transpose-{key}", spec)
    for index, layer in enumerate(parent["layers"]):
        spec = copy.deepcopy(parent)
        spec["layers"][index]["order"] = "attn_ffn" if layer.get("order") == "ffn_attn" else "ffn_attn"
        offer(f"order-{index}", spec)
        for kind in ("attn", "ffn"):
            if layer.get(kind) is None:
                continue
            spec = copy.deepcopy(parent); remove_branch(spec["layers"][index], kind)
            offer(f"delete-{kind}{index}", spec)
            for scale in (.5, .75, 1., 1.25, {"init": .5, "learned": True}, {"init": 1., "learned": True}):
                spec = copy.deepcopy(parent); spec["layers"][index][kind + "_scale"] = scale
                label = "learned" + str(scale["init"]) if isinstance(scale, dict) else str(scale)
                offer(f"scale-{kind}{index}-{label}", spec)
            for source in range(index - 1):
                if parent["layers"][source].get(kind) is not None:
                    spec = copy.deepcopy(parent)
                    spec["layers"][index][kind] = spec["layers"][source][kind]
                    offer(f"share-{kind}{source}-{index}", spec)
        if index + 1 < len(parent["layers"]):
            for inject in (False, True):
                spec = copy.deepcopy(parent)
                repeats = spec["layers"][index].get("loop", 1) + spec["layers"][index + 1].get("loop", 1)
                del spec["layers"][index + 1]
                spec["layers"][index].update(loop=repeats, inject=inject)
                offer(f"loop-{index}-{int(inject)}", spec)
    for width in (1472, 1408, 1280, 1024):
        spec = copy.deepcopy(parent)
        for key in keys:
            current = spec.get("ffns", {}).get(key, {}).get("hidden", 1536)
            set_width(spec, min(width, current), [key])
        offer(f"narrow-all-{width}", spec)
    rng = random.Random(20261111 + round_index * 1009 + int(architecture_hash(parent)[:8], 16))
    rng.shuffle(options)
    return options


@dataclass(frozen=True)
class Job:
    run_id: str
    architecture_id: str
    spec: str
    seed: int
    phase: str
    steps: int = 5000

    def to_dict(self):
        return dict(run_id=self.run_id, architecture_id=self.architecture_id, spec=self.spec,
                    seed=self.seed, phase=self.phase, steps=self.steps)


class Scheduler:
    def __init__(self, args, deadline):
        self.args, self.deadline = args, deadline
        self.cutoff = min(args.training_cutoff_utc or deadline - timedelta(seconds=args.shutdown_buffer_seconds), deadline)
        self.search_cutoff = self.cutoff - timedelta(seconds=args.final_reserve_seconds)
        self.running, self.records, self.completed, self.architectures, self.specs, self.job_keys = {}, {}, {}, {}, {}, {}
        self.stop_requested, self.stage, self.locks = False, "initializing", []
        self.consecutive_failures = 0
        self.streams = {name: (args.out / name).open("x", buffering=1)
                        for name in ("events.jsonl", "architecture_registry.jsonl", "run_registry.jsonl")}
        self.source_hashes = {name: file_hash(source_path(name, ROOT)) for name in
                              (*PROVENANCE_SUPPORT, "cuda_search.py", "cuda_model.py", "cuda_train.py", "data.py", "data/train.json", "data/tokenizer.json") if (source_path(name, ROOT)).is_file()}
        self.source_hash = hashlib.sha256(json.dumps(self.source_hashes, sort_keys=True).encode()).hexdigest()
        self.host = {"hostname": socket.gethostname(), "public_ip": args.host_public_ip}
        try:
            self.host["local_ip"] = socket.gethostbyname(socket.gethostname())
        except OSError:
            self.host["local_ip"] = None
        self.gpu_metadata = self.discover_gpus()
        atomic_json(args.out / "provenance.json", {"created_utc": iso(utc_now()), "host": self.host,
                    "gpus": self.gpu_metadata, "source_sha256": self.source_hashes, "combined_source_sha256": self.source_hash})
        self.save()

    def discover_gpus(self):
        result = {str(gpu): {"physical_index": gpu, "uuid": None, "name": None} for gpu in self.args.gpus}
        try:
            output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True, timeout=10)
            for line in output.splitlines():
                index, uuid, name = [part.strip() for part in line.split(",", 2)]
                if index in result:
                    result[index].update(uuid=uuid, name=name)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        return result

    def append(self, filename, value):
        self.streams[filename].write(json.dumps(value, allow_nan=False) + "\n")
        self.streams[filename].flush()
        os.fsync(self.streams[filename].fileno())

    def emit(self, event, **details):
        value = {"time_utc": iso(utc_now()), "event": event, **details}
        self.append("events.jsonl", value)
        print(json.dumps(value, allow_nan=False), flush=True)

    def record(self, job, status, **details):
        current = self.records.setdefault(job.run_id, {"job": job.to_dict(), "architecture_sha256": job.architecture_id,
                    "source_sha256": self.source_hash, "source_hashes": self.source_hashes, "host": self.host,
                    "spec_path": str(self.args.out / "specs" / (job.architecture_id + ".json")),
                    "path": str(self.args.out / "attempts" / job.run_id),
                    "parameters": parameter_count(self.specs[job.architecture_id])})
        current.update(status=status, updated_utc=iso(utc_now()), **details)
        self.append("run_registry.jsonl", copy.deepcopy(current))
        return current

    def register(self, name, spec, generation=0, parents=None, operator="breadth"):
        validate_spec(spec)
        digest = architecture_hash(spec)
        if digest in self.architectures:
            return digest, False
        if not self.args.plan_only:
            from nas.cuda_model import Config, validate, spec_parameter_count
            validate(spec, Config(**FRAME))
            assert spec_parameter_count(spec, Config(**FRAME)) == parameter_count(spec)
        clean = copy.deepcopy(spec)
        clean["name"] = name
        path = self.args.out / "specs" / (digest + ".json")
        with path.open("x") as stream:
            stream.write(json.dumps(clean, indent=2) + "\n")
        record = {"architecture_id": digest, "canonical_sha256": digest, "file_sha256": file_hash(path),
                  "name": name, "generation": generation, "parents": parents or [], "operator": operator,
                  "parameters": parameter_count(clean), "learned_scale_parameters": learned_scale_count(clean),
                  "applications": sum(layer.get("loop", 1) for layer in clean["layers"]),
                  "created_utc": iso(utc_now()), "spec_path": str(path)}
        self.architectures[digest], self.specs[digest] = record, clean
        self.append("architecture_registry.jsonl", record)
        return digest, True

    def make_job(self, architecture_id, seed, phase):
        key = (architecture_id, seed, 5000)
        if key in self.job_keys:
            previous = self.job_keys[key]
            if self.records[previous.run_id]["status"] == "not_admitted":
                self.record(previous, "queued", queued_utc=iso(utc_now()), failure_reason=None)
                return previous
            return None
        name = self.architectures[architecture_id]["name"]
        job = Job(f"r{len(self.job_keys) + 1:05d}-{architecture_id[:10]}-s{seed}", architecture_id, name, seed, phase)
        self.job_keys[key] = job
        self.record(job, "queued", queued_utc=iso(utc_now()))
        return job

    def save(self):
        atomic_json(self.args.out / "status.json", {"updated_utc": iso(utc_now()), "stage": self.stage,
                    "deadline_utc": iso(self.deadline), "training_cutoff_utc": iso(self.cutoff),
                    "search_cutoff_utc": iso(self.search_cutoff), "architectures": len(self.architectures),
                    "completed": len(self.completed), "attempts": len(self.records),
                    "gpu_assignments": {str(gpu): item["job"].run_id for gpu, item in self.running.items()},
                    "failed": sum(record["status"] in ("failed", "interrupted", "timed_out") for record in self.records.values())})

    def estimate(self):
        times = [record["wall_seconds"] for record in self.records.values()
                 if record["status"] == "completed" and record["job"]["spec"] == "baseline"]
        return statistics.median(times) if times else self.args.estimated_5000_seconds

    def acquire_gpus(self):
        for gpu in self.args.gpus:
            lock = open(f"/tmp/nas-cuda-search-gpu-{gpu}.lock", "a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                raise RuntimeError(f"GPU {gpu} belongs to another search scheduler")
            self.locks.append(lock)

    def command(self, job):
        command = [*python_command("cuda_train.py", ROOT), "--out", str(self.args.out / "attempts" / job.run_id),
                   "--spec", str(self.args.out / "specs" / (job.architecture_id + ".json")),
                   "--layers", "8", "--d-model", "384", "--heads", "6", "--d-ff", "1536", "--dropout", "0.1",
                   "--steps", "5000", "--seed", str(job.seed), "--batch-rows", "8",
                   "--lr", "0.001", "--warmup", "100", "--min-lr-frac", "0.1", "--weight-decay", "0.1", "--clip", "1.0",
                   "--eval-every", "250", "--checkpoint-every", str(self.args.checkpoint_every),
                   "--max-params", str(BASELINE_PARAMETERS + learned_scale_count(self.specs[job.architecture_id])),
                   "--precision", self.args.precision, "--device", "cuda", "--cpu-threads", str(self.args.cpu_threads),
                   "--projection-chunk-size", str(self.args.projection_chunk_size)]
        if self.args.compile:
            command.extend(["--compile", "--compile-mode", self.args.compile_mode])
        return command

    def launch(self, gpu, job):
        folder = self.args.out / "attempts" / job.run_id
        if folder.exists() or self.records[job.run_id]["status"] != "queued":
            raise RuntimeError(f"refusing to overwrite {job.run_id}")
        command = self.command(job)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED=str(job.seed),
                   OMP_NUM_THREADS=str(self.args.cpu_threads), TOKENIZERS_PARALLELISM="false")
        log = (self.args.out / "logs" / (job.run_id + ".log")).open("x")
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except Exception as exc:
            log.close()
            self.record(job, "failed", ended_utc=iso(utc_now()), failure_reason=f"launch: {exc}", physical_gpu=self.gpu_metadata[str(gpu)])
            self.consecutive_failures += 1
            return
        self.running[gpu] = {"job": job, "process": process, "log": log, "start": time.monotonic(), "terminate_at": None}
        self.record(job, "running", started_utc=iso(utc_now()), pid=process.pid, physical_gpu=self.gpu_metadata[str(gpu)], command=command)
        self.emit("launched", run_id=job.run_id, architecture=job.spec, seed=job.seed, phase=job.phase, gpu=gpu)
        self.save()

    def checkpoint_metadata(self, folder):
        result = {}
        for path in sorted(folder.glob("*.pt")):
            result[path.name] = {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size, "pruned": False}
        return result

    def read_metrics(self, folder, summary):
        metrics = {key: summary.get(key) for key in ("best_dev_loss", "best_step", "train_eval_loss_at_best_dev",
                   "generalization_gap_at_best_dev", "final_train_eval_loss", "final_dev_loss", "train_seconds")}
        rows = []
        log = folder / "log.jsonl"
        if log.is_file():
            for line in log.read_text().splitlines():
                try:
                    value = json.loads(line)
                    if value.get("dev_loss") is not None:
                        rows.append(value)
                except json.JSONDecodeError:
                    continue
        if rows:
            best = min(rows, key=lambda row: row["dev_loss"])
            for key, value in {"best_step": best.get("step"), "train_eval_loss_at_best_dev": best.get("train_eval_loss"),
                               "generalization_gap_at_best_dev": best.get("generalization_gap"),
                               "final_train_eval_loss": rows[-1].get("train_eval_loss"), "final_dev_loss": rows[-1]["dev_loss"]}.items():
                if metrics.get(key) is None:
                    metrics[key] = value
            metrics["evaluations"] = len(rows)
        return metrics

    def collect(self):
        for gpu, item in list(self.running.items()):
            process, job = item["process"], item["job"]
            elapsed = time.monotonic() - item["start"]
            code = process.poll()
            if code is None:
                if elapsed > self.args.job_timeout_seconds and item["terminate_at"] is None:
                    self.terminate_one(item, "job timeout")
                if item["terminate_at"] is not None and time.monotonic() - item["terminate_at"] > 10:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                continue
            item["log"].close()
            folder = self.args.out / "attempts" / job.run_id
            details = {"returncode": code, "ended_utc": iso(utc_now()), "wall_seconds": elapsed,
                       "checkpoints": self.checkpoint_metadata(folder) if folder.exists() else {}}
            status = "failed"
            reason = self.records[job.run_id].get("stop_reason")
            if reason:
                status = "timed_out" if reason == "job timeout" else "interrupted"
                details["failure_reason"] = reason
            elif code == 0:
                try:
                    summary = json.loads((folder / "summary.json").read_text())
                    config = json.loads((folder / "config.json").read_text())
                    if (summary["steps"] != 5000 or summary["schedule_steps"] != 5000 or summary.get("benchmark")
                            or not math.isfinite(summary["best_dev_loss"]) or summary["parameters"] != parameter_count(self.specs[job.architecture_id])
                            or config["model"] != FRAME or architecture_hash(config["spec"]) != job.architecture_id
                            or config["args"]["seed"] != job.seed or config["args"]["precision"] != self.args.precision
                            or any(config["args"].get(key) != value for key, value in RECIPE.items())):
                        raise ValueError("result/config differs from the declared architecture or fixed recipe")
                    for relative, digest in config.get("sha256", {}).items():
                        if relative in self.source_hashes and digest != self.source_hashes[relative]:
                            raise ValueError(f"source/data changed after scheduler start: {relative}")
                    details.update(summary=summary, metrics=self.read_metrics(folder, summary))
                    status = "completed"
                    self.completed[job.run_id] = {"job": job, "summary": summary, "path": str(folder), "metrics": details["metrics"]}
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    details["failure_reason"] = str(exc)
            else:
                details["failure_reason"] = f"trainer exited with status {code}"
            record = self.record(job, status, **details)
            self.consecutive_failures = 0 if status == "completed" else self.consecutive_failures + 1
            self.emit("finished", run_id=job.run_id, architecture=job.spec, seed=job.seed, status=status,
                      wall_seconds=elapsed, best_dev_loss=record.get("summary", {}).get("best_dev_loss"),
                      failure_reason=record.get("failure_reason"))
            del self.running[gpu]
            if status == "completed" and not self.args.no_plots:
                try:
                    from nas.cuda_report import render_run
                    render_run(folder)
                except Exception as exc:
                    self.emit("plot_error", run_id=job.run_id, error=str(exc))
            self.update_retention()
            self.prune_if_needed()
            self.save()

    def terminate_one(self, item, reason):
        if item["terminate_at"] is not None:
            return
        item["terminate_at"] = time.monotonic()
        self.record(item["job"], "running", stop_reason=reason)
        try:
            os.killpg(item["process"].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def stop_children(self, reason):
        for item in self.running.values():
            self.terminate_one(item, reason)
        deadline = time.monotonic() + 22
        while self.running and time.monotonic() < deadline:
            self.collect()
            if self.running:
                time.sleep(.1)
        for item in self.running.values():
            try:
                os.killpg(item["process"].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            item["process"].wait(timeout=5)
        self.collect()

    def run_stage(self, jobs, phase, finish_by=None):
        self.stage = phase
        queue = [job for job in jobs if job is not None and self.records[job.run_id]["status"] == "queued"]
        finish_by = min(finish_by or self.cutoff, self.cutoff)
        self.emit("phase_started", phase=phase, jobs=len(queue), finish_by_utc=iso(finish_by))
        last_heartbeat = time.monotonic()
        while queue or self.running:
            self.collect()
            if self.stop_requested or utc_now() >= self.cutoff or self.consecutive_failures >= self.args.max_consecutive_failures:
                reason = "signal" if self.stop_requested else ("training cutoff" if utc_now() >= self.cutoff else "consecutive failure limit")
                self.stop_children(reason)
                if reason == "consecutive failure limit":
                    self.stop_requested = True
                break
            for gpu in self.args.gpus:
                if gpu in self.running or not queue:
                    continue
                job = queue.pop(0)
                if utc_now() + timedelta(seconds=self.estimate() * self.args.runtime_margin) > finish_by:
                    self.record(job, "not_admitted", failure_reason="estimated runtime exceeds phase allowance")
                    continue
                self.launch(gpu, job)
            if time.monotonic() - last_heartbeat >= 30:
                self.emit("heartbeat", phase=phase, queued=len(queue), running=len(self.running),
                          seconds_to_training_cutoff=max(0, (self.cutoff - utc_now()).total_seconds()))
                last_heartbeat = time.monotonic()
            if queue or self.running:
                time.sleep(self.args.poll_seconds)
        for job in queue:
            self.record(job, "not_admitted", failure_reason="scheduler stopped")
        self.emit("phase_finished", phase=phase)
        self.write_leaderboard()
        self.save()

    def rankings(self, reduced=False, min_seeds=1, seed0_only=False):
        grouped = {}
        for item in self.completed.values():
            job = item["job"]
            if job.phase == "final" or (seed0_only and job.seed != 0) or job.seed not in (0, 1, 2):
                continue
            if reduced and item["summary"]["parameters"] >= BASELINE_PARAMETERS:
                continue
            grouped.setdefault(job.architecture_id, {})[job.seed] = item["summary"]["best_dev_loss"]
        ranked = []
        for architecture_id, losses in grouped.items():
            if len(losses) < min_seeds:
                continue
            entry = self.architectures[architecture_id]
            ranked.append({"architecture_id": architecture_id, "name": entry["name"], "parameters": entry["parameters"],
                           "mean_dev_loss": statistics.mean(losses.values()), "seeds": sorted(losses),
                           "dev_losses": {str(seed): loss for seed, loss in losses.items()},
                           "std_dev_loss": statistics.stdev(losses.values()) if len(losses) > 1 else None})
        return sorted(ranked, key=lambda item: (item["mean_dev_loss"], item["parameters"]))

    def write_leaderboard(self):
        atomic_json(self.args.out / "rankings.json", {"updated_utc": iso(utc_now()), "validation_used": False,
                    "all": self.rankings(), "reduced": self.rankings(reduced=True),
                    "confirmed_reduced": self.rankings(reduced=True, min_seeds=3)})
        if self.completed and not self.args.no_plots:
            try:
                from nas.cuda_report import render_leaderboard
                render_leaderboard([Path(item["path"]) for item in self.completed.values()], self.args.out / "report", top=12, run_plots=False)
            except Exception as exc:
                self.emit("leaderboard_error", error=str(exc))

    def retained_run_ids(self):
        keep = {run_id for run_id, item in self.completed.items()
                if item["job"].spec == "baseline" or item["job"].phase == "final"}
        ranked = self.rankings(seed0_only=True)
        leaders = [item for item in ranked if item["parameters"] < BASELINE_PARAMETERS][:self.args.retain_top]
        leaders += [item for item in ranked if item["parameters"] >= BASELINE_PARAMETERS][:3]
        confirmed = self.rankings(min_seeds=3)
        leaders += [item for item in confirmed if item["parameters"] < BASELINE_PARAMETERS][:self.args.retain_top]
        leaders += [item for item in confirmed if item["parameters"] >= BASELINE_PARAMETERS][:3]
        leader_ids = {item["architecture_id"] for item in leaders}
        keep.update(run_id for run_id, item in self.completed.items() if item["job"].architecture_id in leader_ids)
        return keep

    def update_retention(self):
        keep = self.retained_run_ids()
        paths, details = [], []
        for run_id in sorted(keep):
            record = self.records[run_id]
            checkpoint = record.get("checkpoints", {}).get("best.pt")
            if not checkpoint or checkpoint.get("pruned"):
                continue
            path = Path(checkpoint["path"])
            if not path.is_file():
                continue
            try:
                relative = str(path.resolve().relative_to(self.args.runs_root))
            except ValueError:
                self.emit("retention_path_error", run_id=run_id, reason="checkpoint is outside runs_root")
                continue
            paths.append(relative)
            details.append({"relative_path": relative, "sha256": checkpoint["sha256"], "bytes": checkpoint["bytes"], "run_id": run_id})
        atomic_json(self.args.out / "retained_checkpoints.json", {"updated_utc": iso(utc_now()), "relative_paths": paths,
                    "total_bytes": sum(item["bytes"] for item in details), "checkpoints": details})

    def prune_if_needed(self):
        total = 0
        for path in self.args.out.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except FileNotFoundError:
                # Other workers atomically rename temporary checkpoints while
                # the scheduler measures the shared output tree.
                continue
        limit = self.args.max_disk_gib * 1024 ** 3
        if total <= limit:
            return
        keep = self.retained_run_ids()
        options = []
        for run_id, record in self.records.items():
            if record["status"] not in ("completed", "failed", "interrupted", "timed_out"):
                continue
            for name, checkpoint in record.get("checkpoints", {}).items():
                if checkpoint.get("pruned") or (run_id in keep and name == "best.pt"):
                    continue
                options.append((0 if name != "best.pt" else 1, run_id, name, checkpoint))
        for _, run_id, name, checkpoint in sorted(options):
            if total <= limit * .8:
                break
            path = Path(checkpoint["path"])
            if path.is_file():
                # Hash and byte count were durably logged before this deletion.
                path.unlink()
                total -= checkpoint["bytes"]
                checkpoint["pruned"] = True
                self.record(self.completed[run_id]["job"] if run_id in self.completed else Job(**self.records[run_id]["job"]),
                            self.records[run_id]["status"], checkpoints=self.records[run_id]["checkpoints"])
                self.emit("checkpoint_pruned", run_id=run_id, filename=name, sha256=checkpoint["sha256"], bytes=checkpoint["bytes"])
        if total > limit:
            self.emit("disk_limit_unresolved", bytes=total, limit_bytes=limit)
            self.stop_requested = True
        self.update_retention()

    def confirm_shortlist(self, count, finish_by):
        ranked = self.rankings(seed0_only=True)
        chosen = [item for item in ranked if item["parameters"] < BASELINE_PARAMETERS][:count]
        chosen += [item for item in ranked if item["parameters"] >= BASELINE_PARAMETERS and item["name"] != "baseline"][:2]
        # Reserve a genuinely smaller architecture even when tiny cuts lead.
        rescue = next((item for item in ranked if item["parameters"] <= .85 * BASELINE_PARAMETERS
                       and item["mean_dev_loss"] <= ranked[0]["mean_dev_loss"] + .15), None) if ranked else None
        if rescue and rescue not in chosen:
            chosen.append(rescue)
        jobs = [self.make_job(item["architecture_id"], seed, "confirmation") for item in chosen for seed in (1, 2)]
        self.run_stage(jobs, "confirmation", finish_by)

    def adaptive_jobs(self, round_index):
        ranked = self.rankings(seed0_only=True)
        parents = ranked[:max(2, self.args.beam_width // 2)]
        for item in [entry for entry in ranked if entry["parameters"] < BASELINE_PARAMETERS]:
            if item not in parents:
                parents.append(item)
            if len(parents) >= self.args.beam_width:
                break
        if ranked:
            rescue = next((entry for entry in ranked if entry["parameters"] <= .85 * BASELINE_PARAMETERS
                           and entry["mean_dev_loss"] <= ranked[0]["mean_dev_loss"] + .2), None)
            if rescue and rescue not in parents:
                parents[-1:] = [rescue]
        pools = [(parent, mutations(self.specs[parent["architecture_id"]], round_index)) for parent in parents]
        jobs = []
        while pools and len(jobs) < self.args.round_candidates and len(self.architectures) < self.args.max_architectures:
            next_pools = []
            for parent, pool in pools:
                while pool:
                    operator, spec = pool.pop()
                    digest = architecture_hash(spec)
                    if digest in self.architectures:
                        continue
                    name = f"r{round_index}-{parent['name'][:42]}-{operator}"
                    spec["description"] = f"Adaptive round {round_index}: {operator}; parent {parent['architecture_id']}"
                    digest, _ = self.register(name, spec, round_index, [parent["architecture_id"]], operator)
                    jobs.append(self.make_job(digest, 0, f"adaptive-{round_index}"))
                    break
                if pool:
                    next_pools.append((parent, pool))
                if len(jobs) >= self.args.round_candidates or len(self.architectures) >= self.args.max_architectures:
                    break
            pools = next_pools
        self.emit("adaptive_generation", round=round_index, jobs=len(jobs), parents=[item["architecture_id"] for item in parents])
        return jobs

    def final_confirmation(self):
        ranked = self.rankings(reduced=True, min_seeds=3)
        if not ranked:
            self.emit("final_unavailable", reason="No reduced architecture completed all selection seeds 0/1/2")
            return False
        finalists = ranked[:2]
        selection = {"frozen_utc": iso(utc_now()), "validation_used": False, "selection_seeds": [0, 1, 2],
                     "primary_architecture_id": finalists[0]["architecture_id"], "finalists": finalists,
                     "fresh_seeds": [10, 11, 12], "rule": PROTOCOL["confirmation"]}
        atomic_json(self.args.out / "selection.json", selection)
        baseline_id = architecture_hash(baseline_spec())
        jobs = [self.make_job(architecture_id, seed, "final")
                for seed in (10, 11, 12) for architecture_id in [baseline_id] + [item["architecture_id"] for item in finalists]]
        self.run_stage(jobs, "final", self.cutoff)
        comparisons = []
        for finalist in finalists:
            pairs = []
            for seed in (10, 11, 12):
                items = {item["job"].architecture_id: item for item in self.completed.values()
                         if item["job"].phase == "final" and item["job"].seed == seed}
                if baseline_id in items and finalist["architecture_id"] in items:
                    baseline_loss = items[baseline_id]["summary"]["best_dev_loss"]
                    candidate_loss = items[finalist["architecture_id"]]["summary"]["best_dev_loss"]
                    pairs.append({"seed": seed, "baseline_dev_loss": baseline_loss, "candidate_dev_loss": candidate_loss,
                                  "delta": candidate_loss - baseline_loss})
            comparisons.append({**finalist, "parameter_reduction": BASELINE_PARAMETERS - finalist["parameters"],
                                "paired_results": pairs, "complete": len(pairs) == 3,
                                "mean_paired_delta": statistics.mean(pair["delta"] for pair in pairs) if pairs else None})
        atomic_json(self.args.out / "final_comparison.json", {"frozen_selection": selection, "comparisons": comparisons,
                    "validation_used": False, "updated_utc": iso(utc_now())})
        primary_id = selection["primary_architecture_id"]
        baseline_runs, winner_runs = [], []
        for seed in (10, 11, 12):
            by_architecture = {item["job"].architecture_id: item for item in self.completed.values()
                               if item["job"].phase == "final" and item["job"].seed == seed}
            baseline_runs.append(by_architecture.get(baseline_id, {}).get("path"))
            winner_runs.append(by_architecture.get(primary_id, {}).get("path"))
        complete = all(baseline_runs) and all(winner_runs)
        primary = self.architectures[primary_id]
        atomic_json(self.args.out / "final_selection.json", {
            "status": "complete" if complete else "incomplete", "updated_utc": iso(utc_now()),
            "winner_spec": {"name": primary["name"], "hash": primary_id, "path": primary["spec_path"],
                            "file_sha256": primary["file_sha256"], "parameters": primary["parameters"]},
            "winner_spec_hash": primary_id, "baseline_parameters": BASELINE_PARAMETERS,
            "parameter_reduction": BASELINE_PARAMETERS - primary["parameters"],
            "seeds": [10, 11, 12], "baseline_runs": baseline_runs, "winner_runs": winner_runs,
            "selection_rule": PROTOCOL["confirmation"], "selection_seeds": [0, 1, 2],
            "fresh_seeds_used_for_architecture_selection": False,
            "validation_accessed": False, "frozen_selection": selection,
            "fresh_comparisons": comparisons, "all_architecture_metrics": self.rankings(),
            "run_statuses": [{"run_id": run_id, "status": record["status"], "architecture_id": record["job"]["architecture_id"],
                              "seed": record["job"]["seed"], "phase": record["job"]["phase"],
                              "failure_reason": record.get("failure_reason")}
                             for run_id, record in self.records.items()]})
        self.emit("final_comparison_written", finalists=[item["name"] for item in finalists])
        return complete

    def import_reference(self, folder):
        """Historical baseline provenance is retained but never substitutes for new logger baselines."""
        summary = json.loads((folder / "summary.json").read_text())
        config = json.loads((folder / "config.json").read_text())
        if (config["model"] != FRAME or architecture_hash(config["spec"]) != architecture_hash(baseline_spec())
                or summary.get("parameters") != BASELINE_PARAMETERS or summary.get("steps") != 5000
                or summary.get("benchmark") or not math.isfinite(summary["best_dev_loss"])
                or any(config["args"].get(key) != value for key, value in RECIPE.items())):
            raise ValueError(f"not an original-frame completed baseline: {folder}")
        self.append("run_registry.jsonl", {"status": "historical_reference", "updated_utc": iso(utc_now()),
                    "path": str(folder), "summary": summary, "seed": config["args"]["seed"],
                    "source_hashes": config.get("sha256"), "architecture_sha256": architecture_hash(config["spec"]),
                    "checkpoints": self.checkpoint_metadata(folder), "eligible_for_selection": False,
                    "reason": "Preserved provenance; new baselines establish logger/schema parity."})

    def close(self):
        self.stop_children("scheduler exit")
        for record in list(self.records.values()):
            if record["status"] == "queued":
                self.record(Job(**record["job"]), "not_admitted", failure_reason="scheduler exited before launch")
        self.update_retention()
        self.write_leaderboard()
        self.save()
        atomic_json(self.args.out / "results.json", {"stage": self.stage, "validation_used": False,
                    "updated_utc": iso(utc_now()), "attempts": self.records,
                    "architectures": len(self.architectures), "completed": len(self.completed)})
        if self.stage != "plan-only" and not (self.args.out / "final_selection.json").exists():
            atomic_json(self.args.out / "final_selection.json", {"status": "failed", "updated_utc": iso(utc_now()),
                        "failure_reason": f"No complete final selection; scheduler stage={self.stage}",
                        "baseline_runs": [], "winner_runs": [], "validation_accessed": False,
                        "selection_rule": PROTOCOL["confirmation"], "all_architecture_metrics": self.rankings()})
        atomic_json(self.args.out / "terminal_status.json", {"status": "planned" if self.stage == "plan-only" else
                    "complete" if self.stage == "completed" else "failed", "stage": self.stage,
                    "pid": os.getpid(), "updated_utc": iso(utc_now()),
                    "final_selection_path": str(self.args.out / "final_selection.json")})
        for stream in self.streams.values():
            stream.close()
        for lock in self.locks:
            lock.close()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    window = parser.add_mutually_exclusive_group(required=True)
    window.add_argument("--deadline-utc", type=parse_utc)
    window.add_argument("--hours", type=float)
    parser.add_argument("--training-cutoff-utc", type=parse_utc)
    parser.add_argument("--shutdown-buffer-seconds", type=float, default=1800)
    parser.add_argument("--final-reserve-seconds", type=float, default=1200)
    parser.add_argument("--estimated-5000-seconds", type=float, default=120)
    parser.add_argument("--runtime-margin", type=float, default=1.3)
    parser.add_argument("--job-timeout-seconds", type=float, default=900)
    parser.add_argument("--max-consecutive-failures", type=int, default=6)
    parser.add_argument("--max-architectures", type=int, default=800)
    parser.add_argument("--adaptive-rounds", type=int, default=8)
    parser.add_argument("--round-candidates", type=int, default=64)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--shortlist", type=int, default=10)
    parser.add_argument("--stagnation-rounds", type=int, default=3)
    parser.add_argument("--minimum-improvement", type=float, default=.002)
    parser.add_argument("--retain-top", type=int, default=10)
    parser.add_argument("--max-disk-gib", type=float, default=120)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--host-public-ip")
    parser.add_argument("--import-reference", type=Path, action="append", default=[])
    parser.add_argument("--budget-remaining-usd", type=float, default=200)
    parser.add_argument("--total-hourly-rate-usd", type=float, default=16.36)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    parser.add_argument("--projection-chunk-size", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=1)
    return parser


def main():
    args = make_parser().parse_args()
    try:
        args.gpus = [int(value.strip()) for value in args.gpus.split(",")]
    except ValueError as exc:
        raise SystemExit("--gpus must contain unique nonnegative integer indexes") from exc
    if not args.gpus or min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus):
        raise SystemExit("invalid GPU indexes")
    positive = [args.estimated_5000_seconds, args.cpu_threads, args.job_timeout_seconds, args.max_architectures,
                args.round_candidates, args.beam_width, args.shortlist, args.stagnation_rounds, args.max_disk_gib,
                args.budget_remaining_usd, args.poll_seconds, args.max_consecutive_failures]
    if min(positive) <= 0 or args.runtime_margin < 1 or args.adaptive_rounds < 0 or args.shutdown_buffer_seconds < 0 or args.final_reserve_seconds < 0:
        raise SystemExit("invalid runtime/search limits")
    if args.budget_remaining_usd > 200 or args.total_hourly_rate_usd < 0 or args.checkpoint_every < 0 or args.projection_chunk_size < 0:
        raise SystemExit("invalid budget/checkpoint settings")
    start = utc_now()
    deadline = args.deadline_utc or start + timedelta(hours=args.hours)
    if not start < deadline <= start + timedelta(hours=12):
        raise SystemExit("deadline must be in the next 12 hours")
    if args.total_hourly_rate_usd:
        deadline = min(deadline, start + timedelta(hours=args.budget_remaining_usd / args.total_hourly_rate_usd))
    cutoff = min(args.training_cutoff_utc or deadline - timedelta(seconds=args.shutdown_buffer_seconds), deadline)
    if cutoff <= start or cutoff - timedelta(seconds=args.final_reserve_seconds) <= start:
        raise SystemExit("deadline/budget leaves no search window before final confirmation reserve")
    args.out, args.runs_root = args.out.expanduser().resolve(), args.runs_root.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    for directory in ("specs", "attempts", "logs"):
        (args.out / directory).mkdir()
    scheduler = Scheduler(args, deadline)
    try:
        registry = {}
        for name, spec in candidates().items():
            if len(registry) >= args.max_architectures:
                break
            registry[name] = scheduler.register(name, spec)[0]
        baseline_id = registry["baseline"]
        first = [name for name in ("baseline", "relu2", "relu2-narrow1472", "relu2-narrow1472-learned-ffn") if name in registry]
        jobs = [scheduler.make_job(registry[name], 0, "breadth") for name in first]
        jobs += [scheduler.make_job(baseline_id, seed, "baselines") for seed in (1, 2)]
        jobs += [scheduler.make_job(digest, 0, "breadth") for name, digest in registry.items() if name not in first]
        plan = {"created_utc": iso(start), "deadline_utc": iso(deadline), "training_cutoff_utc": iso(scheduler.cutoff),
                "search_cutoff_utc": iso(scheduler.search_cutoff), "protocol": PROTOCOL,
                "initial_queue": [job.to_dict() for job in jobs], "architecture_count": len(registry),
                "parameters": {name: parameter_count(scheduler.specs[digest]) for name, digest in registry.items()},
                "budget_remaining_usd": args.budget_remaining_usd, "total_hourly_rate_usd": args.total_hourly_rate_usd,
                "args": {key: [str(v) for v in value] if isinstance(value, list) and value and isinstance(value[0], Path)
                         else str(value) if isinstance(value, (Path, datetime)) else value for key, value in vars(args).items()}}
        atomic_json(args.out / "plan.json", plan)
        if args.plan_only:
            scheduler.stage = "plan-only"
            print(json.dumps({"plan": str(args.out / "plan.json"), "architectures": len(registry), "jobs": len(jobs)}), flush=True)
            return 0
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda signum, frame: setattr(scheduler, "stop_requested", True))
        scheduler.acquire_gpus()
        for folder in args.import_reference:
            scheduler.import_reference(folder.expanduser().resolve())
        # Check the fresh baseline trio before committing the remaining breadth budget.
        bootstrap_size = len(first) + 2
        scheduler.run_stage(jobs[:bootstrap_size], "bootstrap", scheduler.search_cutoff)
        if scheduler.stop_requested:
            scheduler.stage = "interrupted"
            return 130
        baseline_seeds = {item["job"].seed for item in scheduler.completed.values() if item["job"].architecture_id == baseline_id}
        if not {0, 1, 2} <= baseline_seeds:
            scheduler.stage = "incomplete-baselines"
            scheduler.emit("search_stopped", reason="new baseline selection seeds 0/1/2 did not all complete")
            return 2
        scheduler.run_stage(jobs[bootstrap_size:], "breadth", scheduler.search_cutoff)
        scheduler.confirm_shortlist(args.shortlist, scheduler.search_cutoff)
        best = scheduler.rankings(reduced=True, min_seeds=3)
        previous = best[0]["mean_dev_loss"] if best else math.inf
        stagnant = 0
        for round_index in range(1, args.adaptive_rounds + 1):
            if scheduler.stop_requested or utc_now() >= scheduler.search_cutoff or len(scheduler.architectures) >= args.max_architectures:
                break
            jobs = scheduler.adaptive_jobs(round_index)
            if not jobs:
                break
            scheduler.run_stage(jobs, f"adaptive-{round_index}", scheduler.search_cutoff)
            scheduler.confirm_shortlist(args.shortlist, scheduler.search_cutoff)
            best = scheduler.rankings(reduced=True, min_seeds=3)
            current = best[0]["mean_dev_loss"] if best else math.inf
            improvement = previous - current
            stagnant = 0 if improvement >= args.minimum_improvement else stagnant + 1
            previous = min(previous, current)
            scheduler.emit("adaptive_round_result", round=round_index, best_confirmed_reduced_loss=current if math.isfinite(current) else None,
                           improvement=improvement if math.isfinite(improvement) else None, stagnant_rounds=stagnant)
            if stagnant >= args.stagnation_rounds:
                scheduler.emit("search_early_stop", reason="confirmed loss improvement below threshold for consecutive adaptive rounds")
                break
        final_complete = False
        if not scheduler.stop_requested:
            # Confirmation can use the protected final reserve if earlier phases exhausted their allowance.
            scheduler.confirm_shortlist(args.shortlist, scheduler.cutoff - timedelta(seconds=3 * scheduler.estimate() * args.runtime_margin))
            final_complete = scheduler.final_confirmation()
        scheduler.stage = "interrupted" if scheduler.stop_requested else "completed" if final_complete else "incomplete-final"
        return 130 if scheduler.stop_requested else 0 if final_complete else 2
    except BaseException as exc:
        scheduler.stage = "error"
        scheduler.emit("scheduler_error", error=repr(exc))
        raise
    finally:
        scheduler.close()


if __name__ == "__main__":
    raise SystemExit(main())
