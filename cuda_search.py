"""Bounded multi-GPU architecture search; only the training/dev splits are used.

Example (the enclosing cloud controller must also terminate the rented host):
  python cuda_search.py --gpus 0,1 --hours 10 --out runs/overnight \
      --estimated-5000-seconds 900 --total-hourly-rate-usd 6

The scheduler itself uses only the standard library, including in --plan-only
mode. Each physical GPU gets one training subprocess. Existing run directories
are never reused. Baselines run at K=5000 and seeds 0/1, screens use a separate
K=2000 cosine schedule, finalists return to K=5000, and a selected reduced model
and the baseline receive fresh seed-2 confirmation. The reserved validation
split remains untouched. Cloud termination and total rental billing are the
responsibility of the enclosing controller; killing training does not end a
Lambda rental.
"""
from __future__ import annotations

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
import signal
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
FRAME = {"vocab_size": 32768, "max_len": 1024, "n_layers": 10,
         "d_model": 512, "n_heads": 8, "d_ff": 2048, "dropout": 0.1}
BASELINE_PARAMETERS = 48_780_288
PROTOCOL = {
    "objective": "Lowest dev loss; the final candidate must have fewer parameters than the baseline.",
    "frame": FRAME,
    "baseline_parameters": BASELINE_PARAMETERS,
    "baseline_seeds": [0, 1],
    "baseline_steps": 5000,
    "screen_seed": 0,
    "screen_steps": 2000,
    "full_trial_seed": 0,
    "full_trial_steps": 5000,
    "confirmation_seed": 2,
    "confirmation_steps": 5000,
    "selection_metric": "best dev loss across the fixed every-250-step evaluations",
    "screen_comparator": "The K=2000 baseline only; K=5000 has a different LR schedule.",
    "screen_caveat": "Short runs understate regularization benefits. Reserve one full trial for the best screened >=15% parameter cut if within 0.30 dev loss of the screen baseline.",
    "no_screen_fallback": "If time prevents usable screens, run the predeclared ReLU2 + all-FFN width1920 candidate at K=5000.",
    "confirmation": "Freeze the lowest-loss reduced K=5000 seed0 candidate, then run that candidate and the baseline at fresh seed2. Report both seed pairs, including negative results.",
    "validation": "Never accessed by this scheduler. Final held-out evaluation is a separate action after model selection.",
    "parallelism": "One subprocess per physical GPU; all attempts use the same trainer recipe and precision.",
    "bootstrap": "With spare GPUs, start the K2000 baseline and K5000 ReLU2 anchor alongside the two K5000 baselines.",
    "deadline": "Admit jobs only when estimated runtime fits; terminate children at the training cutoff before the absolute deadline.",
}


def utc_now():
    return datetime.now(timezone.utc)


def iso(value):
    return value.astimezone(timezone.utc).isoformat()


def parse_utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("include a timezone, for example 2026-11-11T18:00:00Z")
    return parsed.astimezone(timezone.utc)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def baseline_spec():
    return {"name": "baseline", "description": "10-layer 48.78M-parameter GELU baseline",
            "layers": [{"attn": f"A{i}", "ffn": f"F{i}"} for i in range(10)], "ffns": {}}


def parameter_count(spec):
    """Standard-library count for generated specs; cuda_model validates at launch."""
    width = FRAME["d_model"]
    total = (FRAME["vocab_size"] + FRAME["max_len"]) * width + 2 * width
    attentions = {layer.get("attn") for layer in spec["layers"]} - {None}
    ffns = {layer.get("ffn") for layer in spec["layers"]} - {None}
    total += len(attentions) * (4 * width * width + 2 * width)
    for key in ffns:
        options = spec.get("ffns", {}).get(key, {})
        total += width * options.get("hidden", 2048) * (1 if options.get("transpose_tie") else 2) + 2 * width
    return total


def candidates():
    """Deterministic breadth around the strongest previous MLX result: ReLU2."""
    specs = {"baseline": baseline_spec()}

    def relu2():
        spec = baseline_spec()
        spec["ffns"] = {f"F{i}": {"act": "relu2"} for i in range(10)}
        return spec

    def add(name, spec, description):
        spec["name"], spec["description"] = name, description
        # Internal checks keep plan-only usable without importing torch.
        applications = sum(layer.get("loop", 1) for layer in spec["layers"])
        assert 0 < applications <= 10
        assert parameter_count(spec) <= BASELINE_PARAMETERS
        for kind in ("attn", "ffn"):
            for previous, current in zip(spec["layers"], spec["layers"][1:]):
                assert previous.get(kind) is None or previous.get(kind) != current.get(kind)
        specs[name] = spec

    add("relu2", relu2(), "ReLU2 throughout; unchanged parameter count, activation anchor")
    for hidden in (1984, 1920, 1792, 1536, 1024):
        spec = relu2()
        for options in spec["ffns"].values():
            options["hidden"] = hidden
        add(f"relu2-narrow{hidden}", spec, f"ReLU2 with every FFN narrowed to {hidden}")
    for label, indexes in (("early", range(3)), ("middle", range(3, 7)), ("late", range(7, 10))):
        spec = relu2()
        for index in indexes:
            spec["ffns"][f"F{index}"]["hidden"] = 1536
        add(f"relu2-narrow-{label}", spec, f"ReLU2, only {label} FFNs narrowed to1536")
    for kind in ("attn", "ffn", "both"):
        spec = relu2()
        for sublayer in (("attn", "ffn") if kind == "both" else (kind,)):
            spec["layers"][7][sublayer] = spec["layers"][3][sublayer]
        add(f"relu2-share-{kind}3-7", spec, f"ReLU2; nonadjacent layers3/7 share {kind}, including norms")
    for kind, indexes in (("attn", (5,)), ("attn", (2, 7)), ("ffn", (5,)), ("ffn", (8,))):
        spec = relu2()
        for index in indexes:
            spec["layers"][index][kind] = None
        add(f"relu2-delete-{kind}" + "-".join(map(str, indexes)), spec,
            f"ReLU2; delete {kind} at layer indexes {indexes}")
    for inject in (False, True):
        spec = relu2()
        del spec["layers"][4]
        spec["layers"][3].update(loop=2, inject=inject)
        add("relu2-loop-middle" + ("-inject" if inject else ""), spec,
            f"ReLU2; replace layers3/4 by two applications of layer3; input injection={inject}")
    for inject in (False, True):
        spec = relu2()
        spec["layers"] = [{"attn": f"A{i}", "ffn": f"F{i}", "loop": 2, "inject": inject} for i in range(5)]
        add("relu2-loop5x2" + ("-inject" if inject else ""), spec,
            f"ReLU2; five unique layers applied twice each; input injection={inject}")
    spec = relu2()
    spec["layers"] = [{"attn": f"A{i % 5}", "ffn": f"F{i % 5}"} for i in range(10)]
    add("relu2-cycle5", spec, "ReLU2; five unique layers cycled twice, ABCDEABCDE")
    for operation in ("share", "delete"):
        spec = copy.deepcopy(specs["relu2-narrow1792"])
        if operation == "share":
            spec["layers"][7]["attn"] = "A3"
        else:
            spec["layers"][5]["attn"] = None
        add(f"relu2-narrow1792-{operation}-attn", spec, f"ReLU2 width1792 plus one attention {operation}")
    spec = baseline_spec()
    spec["ffns"] = {f"F{i}": {"hidden": 1792} for i in range(10)}
    add("gelu-narrow1792", spec, "GELU width1792 comparator: isolate the narrowing effect")
    for half, indexes in (("early", range(5)), ("late", range(5, 10))):
        spec = copy.deepcopy(specs["relu2-narrow1920"])
        for index in indexes:
            spec["ffns"][f"F{index}"]["act"] = "gelu"
        add(f"mixed-gelu-{half}", spec, f"Width1920; GELU in {half} half, ReLU2 in the other half")
    spec = copy.deepcopy(specs["relu2-narrow1920"])
    spec["ffns"]["F8"]["act"] = "gelu_tanh"
    spec["ffns"]["F9"]["act"] = "silu"
    add("mixed-smooth-late", spec, "Width1920 ReLU2 with tanh-GELU/SiLU in the final two FFNs")
    spec = relu2()
    spec["ffns"]["F9"].update(transpose_tie=True, hidden=1024)
    add("relu2-transpose-last", spec, "ReLU2; transpose-tie and narrow only the final FFN")
    return specs


@dataclass(frozen=True)
class Job:
    name: str
    spec: str
    steps: int
    seed: int
    phase: str

    def to_dict(self):
        return dict(name=self.name, spec=self.spec, steps=self.steps, seed=self.seed, phase=self.phase)


def initial_jobs(specs, phase):
    baseline = [Job(f"baseline-k5000-s{seed}", "baseline", 5000, seed, "baselines") for seed in (0, 1)]
    screens = [Job("screen-baseline-k2000-s0", "baseline", 2000, 0, "screens"),
               Job("anchor-relu2-k5000-s0", "relu2", 5000, 0, "screens")]
    screens += [Job(f"screen-{name}-k2000-s0", name, 2000, 0, "screens")
                for name in specs if name != "baseline"]
    return ([] if phase == "search" else baseline), ([] if phase == "baselines" else screens)


class Scheduler:
    def __init__(self, args, specs, deadline):
        self.args, self.specs, self.deadline = args, specs, deadline
        self.cutoff = deadline - timedelta(seconds=args.shutdown_buffer_seconds)
        self.running = {}
        self.jobs = {}
        self.records = {}
        self.completed = {}
        self.stop_requested = False
        self.locks = []
        self.events = (args.out / "events.jsonl").open("x", buffering=1)
        self.stage = "initializing"
        self.save()

    def emit(self, event, **details):
        value = {"time_utc": iso(utc_now()), "event": event, **details}
        self.events.write(json.dumps(value, allow_nan=False) + "\n")
        print(json.dumps(value, allow_nan=False), flush=True)

    def save(self):
        atomic_json(self.args.out / "status.json", {
            "updated_utc": iso(utc_now()), "stage": self.stage,
            "deadline_utc": iso(self.deadline), "training_cutoff_utc": iso(self.cutoff),
            "gpu_assignments": {str(gpu): item["job"].name for gpu, item in self.running.items()},
            "attempts": self.records,
        })

    def estimate(self, steps=5000):
        observed = []
        for record in self.records.values():
            if record.get("status") not in ("completed", "imported"):
                continue
            # Keep baseline timing authoritative: aggressive parameter cuts can
            # make screens substantially cheaper than the final baseline rerun.
            if record["job"]["spec"] != "baseline" or record["job"]["steps"] != 5000:
                continue
            duration = record.get("wall_seconds")
            if duration is None:
                summary = record["summary"]
                duration = summary["train_seconds"] + summary.get("preparation_seconds", 0)
            observed.append(duration)
        return (statistics.median(observed) if observed else self.args.estimated_5000_seconds) * steps / 5000

    def acquire_gpus(self):
        for gpu in self.args.gpus:
            lock = open(f"/tmp/nas-cuda-search-gpu-{gpu}.lock", "a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                raise RuntimeError(f"GPU {gpu} is already assigned to another cuda_search scheduler")
            self.locks.append(lock)

    def command(self, job):
        command = [sys.executable, "-u", str(ROOT / "cuda_train.py"),
                   "--out", str(self.args.out / "attempts" / job.name),
                   "--spec", str(self.args.out / "specs" / f"{job.spec}.json"),
                   "--steps", str(job.steps), "--seed", str(job.seed),
                   "--max-params", str(BASELINE_PARAMETERS), "--precision", self.args.precision,
                   "--device", "cuda", "--cpu-threads", str(self.args.cpu_threads),
                   "--projection-chunk-size", str(self.args.projection_chunk_size)]
        if self.args.compile:
            command.extend(["--compile", "--compile-mode", self.args.compile_mode])
        return command

    def launch(self, gpu, job):
        if job.name in self.records or (self.args.out / "attempts" / job.name).exists():
            raise RuntimeError(f"refusing to overwrite attempt {job.name}")
        command = self.command(job)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED=str(job.seed),
                   OMP_NUM_THREADS=str(self.args.cpu_threads), TOKENIZERS_PARALLELISM="false")
        log = (self.args.out / "logs" / f"{job.name}.log").open("x")
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            log.close()
            raise
        self.running[gpu] = {"job": job, "process": process, "log": log,
                             "monotonic_start": time.monotonic()}
        self.jobs[job.name] = job
        self.records[job.name] = {"job": job.to_dict(), "gpu": gpu, "pid": process.pid,
                                  "started_utc": iso(utc_now()), "status": "running", "command": command}
        self.emit("launched", name=job.name, gpu=gpu, pid=process.pid, estimate_seconds=self.estimate(job.steps))
        self.save()

    def collect(self):
        for gpu, item in list(self.running.items()):
            code = item["process"].poll()
            if code is None:
                continue
            job = item["job"]
            item["log"].close()
            record = self.records[job.name]
            record.update(returncode=code, ended_utc=iso(utc_now()),
                          wall_seconds=time.monotonic() - item["monotonic_start"])
            summary_path = self.args.out / "attempts" / job.name / "summary.json"
            record["status"] = "failed"
            if code == 0 and summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text())
                    if (summary["steps"] != job.steps or summary["schedule_steps"] != job.steps
                            or summary.get("benchmark") or not math.isfinite(summary["best_dev_loss"])
                            or summary["parameters"] != parameter_count(self.specs[job.spec])):
                        raise ValueError("summary does not match the declared attempt")
                    record.update(status="completed", summary=summary)
                    self.completed[job.name] = {"job": job, "summary": summary, "path": str(summary_path.parent)}
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    record["error"] = str(exc)
            elif code == 0:
                record["error"] = "trainer returned success without summary.json"
            self.emit("finished", name=job.name, gpu=gpu, status=record["status"],
                      returncode=code, wall_seconds=record["wall_seconds"],
                      best_dev_loss=record.get("summary", {}).get("best_dev_loss"))
            del self.running[gpu]
            self.save()

    def stop_children(self, reason):
        if not self.running:
            return
        self.emit("stopping_children", reason=reason, names=[item["job"].name for item in self.running.values()])
        for item in self.running.values():
            self.records[item["job"].name]["stop_reason"] = reason
            try:
                os.killpg(item["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        grace_end = time.monotonic() + 20
        while self.running and time.monotonic() < grace_end:
            self.collect()
            if self.running:
                time.sleep(0.5)
        for item in self.running.values():
            try:
                os.killpg(item["process"].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            item["process"].wait(timeout=10)
        self.collect()

    def run_stage(self, jobs, stage, finish_by=None):
        self.stage = stage
        # Bootstrap may have already run a screen or anchor. Failed attempts
        # are retained too; retrying must use a new root, never overwrite them.
        queue = [job for job in jobs if job.name not in self.records]
        finish_by = min(finish_by or self.cutoff, self.cutoff)
        self.emit("phase_started", phase=stage, queued=[job.name for job in queue], finish_by_utc=iso(finish_by))
        self.save()
        last_heartbeat = time.monotonic()
        while queue or self.running:
            self.collect()
            if self.stop_requested or utc_now() >= self.cutoff:
                self.stop_children("signal" if self.stop_requested else "absolute training cutoff")
                break
            for gpu in self.args.gpus:
                if gpu in self.running or not queue:
                    continue
                job = queue[0]
                needed = self.estimate(job.steps) * self.args.runtime_margin
                if utc_now() + timedelta(seconds=needed) > finish_by:
                    # Keep the remaining screen queue explicit: short jobs later
                    # in the queue may still fit if a long anchor does not.
                    self.emit("not_admitted", name=job.name, reason="estimated runtime exceeds phase allowance", estimate_seconds=needed)
                    queue.pop(0)
                    continue
                queue.pop(0)
                self.launch(gpu, job)
            if time.monotonic() - last_heartbeat >= 30:
                self.emit("heartbeat", phase=stage, running=[item["job"].name for item in self.running.values()],
                          queued=len(queue), seconds_remaining=max(0, (self.cutoff - utc_now()).total_seconds()))
                last_heartbeat = time.monotonic()
            if queue or self.running:
                time.sleep(1)
        for job in queue:
            self.emit("not_admitted", name=job.name, reason="scheduler stopping")
        self.emit("phase_finished", phase=stage)
        self.save()

    def import_baselines(self, root):
        for seed in (0, 1):
            name = f"baseline-k5000-s{seed}"
            folder = root / "attempts" / name
            summary = json.loads((folder / "summary.json").read_text())
            config = json.loads((folder / "config.json").read_text())
            fixed_recipe = {"steps": 5000, "batch_rows": 8, "lr": 1e-3, "warmup": 100,
                            "min_lr_frac": .1, "weight_decay": .1, "clip": 1.0, "eval_every": 250}
            if (summary.get("steps") != 5000 or summary.get("schedule_steps") != 5000
                    or summary.get("benchmark") or summary.get("parameters") != BASELINE_PARAMETERS
                    or not math.isfinite(summary["best_dev_loss"]) or config["model"] != FRAME
                    or config["args"]["seed"] != seed or config["args"]["precision"] != self.args.precision
                    or config["spec"]["layers"] != self.specs["baseline"]["layers"]
                    or config["spec"].get("ffns", {})
                    or any(config["args"].get(key) != value for key, value in fixed_recipe.items())):
                raise ValueError(f"{folder} is not a matching completed baseline")
            for relative, previous in config.get("sha256", {}).items():
                if relative in ("cuda_train.py", "cuda_model.py", "data.py", "data/train.json", "data/tokenizer.json"):
                    if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != previous:
                        raise ValueError(f"{folder} used different source/data: {relative}")
            job = Job(name, "baseline", 5000, seed, "baselines")
            self.completed[name] = {"job": job, "summary": summary, "path": str(folder)}
            self.records[name] = {"job": job.to_dict(), "status": "imported", "summary": summary, "path": str(folder)}
            self.emit("baseline_imported", name=name, path=str(folder), best_dev_loss=summary["best_dev_loss"])
        self.save()

    def shortlist(self):
        screened = [item for item in self.completed.values()
                    if item["job"].phase == "screens" and item["job"].steps == 2000
                    and item["summary"]["parameters"] < BASELINE_PARAMETERS]
        ranked = sorted(screened, key=lambda item: (item["summary"]["best_dev_loss"], item["summary"]["parameters"]))
        chosen = ranked[:self.args.full_candidates]
        comparator = self.completed.get("screen-baseline-k2000-s0")
        rescue = None
        if comparator and self.args.full_candidates >= 2:
            rescue = next((item for item in ranked if item["summary"]["parameters"] <= .85 * BASELINE_PARAMETERS
                           and item["summary"]["best_dev_loss"] <= comparator["summary"]["best_dev_loss"] + .30), None)
            if rescue and rescue not in chosen:
                chosen = chosen[:-1] + [rescue]
        names = [item["job"].spec for item in chosen] or ["relu2-narrow1920"]
        atomic_json(self.args.out / "shortlist.json", {
            "chosen": names, "screen_baseline_loss": comparator["summary"]["best_dev_loss"] if comparator else None,
            "regularization_rescue": rescue["job"].spec if rescue else None,
            "ranking": [{"spec": item["job"].spec, "best_dev_loss": item["summary"]["best_dev_loss"],
                         "parameters": item["summary"]["parameters"]} for item in ranked],
            "rule": PROTOCOL["screen_caveat"],
        })
        self.emit("shortlist_selected", names=names, regularization_rescue=rescue["job"].spec if rescue else None)
        return [Job(f"full-{name}-k5000-s0", name, 5000, 0, "full") for name in names]

    def confirm(self):
        full = [item for item in self.completed.values()
                if item["job"].phase == "full" and item["job"].steps == 5000 and item["job"].seed == 0
                and item["summary"]["parameters"] < BASELINE_PARAMETERS]
        if not full:
            self.emit("confirmation_unavailable", reason="no completed reduced-parameter full trial")
            return
        winner = min(full, key=lambda item: (item["summary"]["best_dev_loss"], item["summary"]["parameters"]))
        name = winner["job"].spec
        baseline = self.completed["baseline-k5000-s0"]
        selection = {"candidate": name, "selected_from": winner["path"],
                     "parameters": winner["summary"]["parameters"],
                     "parameter_reduction": BASELINE_PARAMETERS - winner["summary"]["parameters"],
                     "seed0_baseline_dev_loss": baseline["summary"]["best_dev_loss"],
                     "seed0_candidate_dev_loss": winner["summary"]["best_dev_loss"],
                     "seed0_improves": winner["summary"]["best_dev_loss"] < baseline["summary"]["best_dev_loss"],
                     "confirmation_seed": 2, "selected_utc": iso(utc_now()),
                     "validation_used": False}
        atomic_json(self.args.out / "selection.json", selection)
        jobs = [Job("confirm-baseline-k5000-s2", "baseline", 5000, 2, "confirmation"),
                Job(f"confirm-{name}-k5000-s2", name, 5000, 2, "confirmation")]
        self.run_stage(jobs, "confirmation")
        pair = [self.completed.get(job.name) for job in jobs]
        if all(pair):
            losses = [item["summary"]["best_dev_loss"] for item in pair]
            selection.update(seed2_baseline_dev_loss=losses[0], seed2_candidate_dev_loss=losses[1],
                             seed2_delta=losses[1] - losses[0], seed2_improves=losses[1] < losses[0],
                             confirmation_completed=True)
        else:
            selection["confirmation_completed"] = False
        atomic_json(self.args.out / "selection.json", selection)
        self.emit("selection_result", **selection)

    def close(self):
        self.stop_children("scheduler exit")
        self.save()
        atomic_json(self.args.out / "results.json", {
            "stage": self.stage, "validation_used": False,
            "completed": [{"job": item["job"].to_dict(), "path": item["path"],
                           "parameters": item["summary"]["parameters"],
                           "best_dev_loss": item["summary"]["best_dev_loss"]}
                          for item in self.completed.values()],
            "failed": [{"name": name, "returncode": record.get("returncode"),
                        "error": record.get("error"), "stop_reason": record.get("stop_reason")}
                       for name, record in self.records.items() if record["status"] == "failed"],
        })
        self.events.close()
        for lock in self.locks:
            lock.close()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpus", default="0", help="comma-separated physical GPU indexes, e.g. 0,1")
    window = parser.add_mutually_exclusive_group(required=True)
    window.add_argument("--deadline-utc", type=parse_utc)
    window.add_argument("--hours", type=float)
    parser.add_argument("--phase", choices=("all", "baselines", "search"), default="all")
    parser.add_argument("--baseline-root", type=Path, help="required for --phase search; a completed baselines scheduler root")
    parser.add_argument("--plan-only", action="store_true", help="write specs and explicit initial queue without importing torch or starting jobs")
    parser.add_argument("--estimated-5000-seconds", type=float, default=1800,
                        help="full run including dev/startup/checkpoints; replaced by observed run durations")
    parser.add_argument("--runtime-margin", type=float, default=1.25)
    parser.add_argument("--full-candidates", type=int, default=4)
    parser.add_argument("--shutdown-buffer-seconds", type=float, default=90)
    parser.add_argument("--budget-remaining-usd", type=float, default=200)
    parser.add_argument("--total-hourly-rate-usd", type=float, default=0,
                        help="optional total cost of ALL rented GPUs; shortens cutoff to remaining budget")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    parser.add_argument("--projection-chunk-size", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def main():
    args = make_parser().parse_args()
    try:
        args.gpus = [int(value.strip()) for value in args.gpus.split(",")]
    except ValueError as exc:
        raise SystemExit("--gpus must be comma-separated nonnegative integer indexes") from exc
    if not args.gpus or min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus):
        raise SystemExit("--gpus must be unique nonnegative integer indexes")
    if args.hours is not None and not 0 < args.hours <= 12:
        raise SystemExit("--hours must be positive and no more than 12")
    if args.phase == "search" and args.baseline_root is None:
        raise SystemExit("--phase search requires --baseline-root with completed K5000 seed0/1 baselines")
    if args.phase != "search" and args.baseline_root is not None:
        raise SystemExit("--baseline-root only applies to --phase search")
    if (min(args.estimated_5000_seconds, args.cpu_threads, args.full_candidates) <= 0
            or args.runtime_margin < 1 or args.shutdown_buffer_seconds < 30 or args.projection_chunk_size < 0
            or not 0 < args.budget_remaining_usd <= 200 or args.total_hourly_rate_usd < 0):
        raise SystemExit("invalid runtime, budget, buffer, or worker settings")
    start = utc_now()
    deadline = args.deadline_utc or start + timedelta(hours=args.hours)
    if not start < deadline <= start + timedelta(hours=12):
        raise SystemExit("deadline must be in the future and within 12 hours")
    if args.total_hourly_rate_usd:
        deadline = min(deadline, start + timedelta(hours=args.budget_remaining_usd / args.total_hourly_rate_usd))
    if (deadline - start).total_seconds() <= args.shutdown_buffer_seconds:
        raise SystemExit("deadline/budget leaves no time before the shutdown buffer")
    args.out = args.out.expanduser().resolve()
    if args.baseline_root:
        args.baseline_root = args.baseline_root.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ("specs", "attempts", "logs"):
        (args.out / name).mkdir()
    specs = candidates()
    for name, spec in specs.items():
        atomic_json(args.out / "specs" / f"{name}.json", spec)
    baselines, screens = initial_jobs(specs, args.phase)
    plan = {"created_utc": iso(start), "deadline_utc": iso(deadline),
            "training_cutoff_utc": iso(deadline - timedelta(seconds=args.shutdown_buffer_seconds)),
            "phase": args.phase, "gpus": args.gpus, "protocol": PROTOCOL,
            "budget_remaining_usd": args.budget_remaining_usd,
            "total_hourly_rate_usd": args.total_hourly_rate_usd or None,
            "baseline_root": str(args.baseline_root) if args.baseline_root else None,
            "initial_queue": [job.to_dict() for job in baselines + screens],
            "adaptive_full_trials": args.full_candidates,
            "adaptive_confirmation": {"baseline": "baseline", "candidate": "lowest dev loss reduced K5000 seed0 candidate", "seed": 2, "steps": 5000},
            "spec_parameters": {name: parameter_count(spec) for name, spec in specs.items()},
            "args": {key: str(value) if isinstance(value, (Path, datetime)) else value for key, value in vars(args).items()}}
    atomic_json(args.out / "plan.json", plan)
    if args.plan_only:
        print(json.dumps({"plan": str(args.out / "plan.json"), "initial_jobs": len(baselines + screens),
                          "specs": len(specs), "deadline_utc": iso(deadline)}, indent=2))
        return

    scheduler = Scheduler(args, specs, deadline)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: setattr(scheduler, "stop_requested", True))
    try:
        scheduler.acquire_gpus()
        if args.phase == "search":
            scheduler.import_baselines(args.baseline_root)
        else:
            bootstrap_extras = min(2, max(0, len(args.gpus) - len(baselines)))
            scheduler.run_stage(baselines + screens[:bootstrap_extras], "bootstrap" if bootstrap_extras else "baselines")
        if not all(f"baseline-k5000-s{seed}" in scheduler.completed for seed in (0, 1)):
            scheduler.stage = "incomplete-baselines"
            scheduler.emit("search_stopped", reason="both K5000 baselines must complete successfully")
            return 2
        if args.phase == "baselines" or scheduler.stop_requested:
            scheduler.stage = "completed-baselines"
            return 130 if scheduler.stop_requested else 0
        estimate = scheduler.estimate(5000) * args.runtime_margin
        confirmation_waves = math.ceil(2 / len(args.gpus))
        full_waves = math.ceil(args.full_candidates / len(args.gpus))
        screen_finish = scheduler.cutoff - timedelta(seconds=(confirmation_waves + full_waves) * estimate)
        scheduler.run_stage(screens, "screens", finish_by=screen_finish)
        if scheduler.stop_requested:
            return 130
        full_jobs = scheduler.shortlist()
        full_finish = scheduler.cutoff - timedelta(seconds=confirmation_waves * scheduler.estimate(5000) * args.runtime_margin)
        scheduler.run_stage(full_jobs, "full", finish_by=full_finish)
        if not scheduler.stop_requested:
            scheduler.confirm()
        scheduler.stage = "completed" if not scheduler.stop_requested else "interrupted"
    except BaseException as exc:
        scheduler.stage = "error"
        scheduler.emit("scheduler_error", error=repr(exc))
        raise
    finally:
        scheduler.close()
    return 130 if scheduler.stop_requested else 0


if __name__ == "__main__":
    raise SystemExit(main())
