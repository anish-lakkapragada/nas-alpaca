"""Confirm exactly one frozen minimum-parameter architecture on five fresh seeds.

The search writes selection.json before this program starts. This program never
selects a replacement: a failed held-out comparison remains a failed comparison.
All ten 5000-step training runs and ten held-out evaluations enter append-only
ledgers. CUDA training and per-conversation evaluation each use up to four GPUs.

python cuda_minparams_finalize.py --out runs/cuda/minparams-20261111 \
    --gpus 0,1,2,3 --deadline-utc 2026-11-12T06:15:00Z
"""
from __future__ import annotations

from nas.paths import PROJECT_ROOT, PROVENANCE_SUPPORT, python_command, source_path

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time

from nas.cuda_finalize import atomic_json, inside, inspect_checkpoint, read_json
from nas.cuda_search import BASELINE_PARAMETERS, RECIPE, file_hash, parse_utc

ROOT = PROJECT_ROOT
SEARCH_SEEDS = [0, 1, 2]
FRESH_SEEDS = [100, 101, 102, 103, 104]
TRAIN_PROVENANCE = (*PROVENANCE_SUPPORT, "cuda_train.py", "cuda_model.py", "data.py", "data/train.json", "data/tokenizer.json")
FIXED_ARGS = {**RECIPE, "log_every": 25, "train_eval_conversations": 256,
              "train_eval_seed": 20261111, "precision": "bf16", "compile": True,
              "compile_mode": "default", "projection_chunk_size": 0}
BASELINE_FRAME = {"vocab_size": 32768, "max_len": 1024, "n_layers": 8,
                  "d_model": 384, "n_heads": 6, "d_ff": 1536, "dropout": .1}


def now():
    return datetime.now(timezone.utc)


def append_json(path, row):
    with Path(path).open("a", buffering=1) as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def exclusive_json(path, value):
    """Publish a complete JSON atomically, refusing to replace an existing file."""
    temporary = path.with_name(f".tmp-{os.getpid()}-{path.name}")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def latest_registry(root):
    records = {}
    path = root / "run_registry.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            records[row["job"]["run_id"]] = row
    return records


def spec_tools():
    # Deferred import permits protocol/report unit tests without CUDA/PyTorch.
    from nas.cuda_minparams_search import architecture_hash, parameter_count
    return architecture_hash, parameter_count


def validate_training_run(path, architecture, seed, provenance, *, checkpoints=False,
                          checkpoint_inspector=inspect_checkpoint):
    config, summary = read_json(path / "config.json"), read_json(path / "summary.json")
    hash_architecture, _ = spec_tools()
    args = config["args"]
    if (config["model"] != architecture["frame"]
            or hash_architecture(config["model"], config["spec"]) != architecture["architecture_id"]):
        raise ValueError(f"different architecture or frame: {path}")
    if args.get("seed") != seed or any(args.get(key) != value for key, value in FIXED_ARGS.items()):
        raise ValueError(f"different seed, precision, or fixed training recipe: {path}")
    if args.get("resume") or args.get("benchmark_steps"):
        raise ValueError(f"confirmation must start from scratch, without benchmark mode: {path}")
    if (summary.get("steps") != 5000 or summary.get("schedule_steps") != 5000
            or summary.get("benchmark") or summary.get("parameters") != architecture["parameters"]
            or config.get("parameters") != architecture["parameters"]
            or not math.isfinite(summary["best_dev_loss"])):
        raise ValueError(f"incomplete or incompatible 5000-step run: {path}")
    if any(config["sha256"].get(key) != digest for key, digest in provenance.items()):
        raise ValueError(f"different training source, tokenizer, or data: {path}")
    optimizer = config.get("optimizer", {})
    if (optimizer.get("name") != "AdamWNoBiasCorrection" or optimizer.get("bias_correction") is not False
            or optimizer.get("betas") != [.9, .95] or optimizer.get("eps") != 1e-8):
        raise ValueError(f"different optimizer definition: {path}")
    diagnostic = config.get("train_evaluation", {})
    if diagnostic.get("conversations") != 256 or diagnostic.get("dropout") is not False:
        raise ValueError(f"different training diagnostic: {path}")
    if type(summary.get("answer_tokens_seen")) is not int or summary["answer_tokens_seen"] <= 0:
        raise ValueError(f"missing training token count: {path}")
    row = {"run": str(path), "seed": seed, "parameters": architecture["parameters"],
           "architecture_id": architecture["architecture_id"], "frame": config["model"],
           "best_dev_loss": summary["best_dev_loss"], "best_step": summary["best_step"],
           "train_eval_loss_at_best_dev": summary.get("train_eval_loss_at_best_dev"),
           "final_train_eval_loss": summary.get("final_train_eval_loss"),
           "final_dev_loss": summary.get("final_dev_loss"),
           "train_seconds": summary.get("train_seconds"), "answer_tokens_seen": summary["answer_tokens_seen"],
           "train_evaluation": diagnostic, "hardware": config.get("hardware", {}),
           "config_sha256": file_hash(path / "config.json"), "summary_sha256": file_hash(path / "summary.json")}
    if checkpoints:
        checkpoint = path / "best.pt"
        details = checkpoint_inspector(checkpoint, config)
        if (details["step"] != summary["best_step"] or not math.isclose(
                details["best_dev_loss"], summary["best_dev_loss"], abs_tol=1e-7)):
            raise ValueError(f"best checkpoint differs from recorded dev selection: {path}")
        row.update(checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
                   learned_residual_scalars=details.get("learned_residual_scalars", {}))
    return row


def paired_controls(baselines, candidates):
    if len(baselines) != len(candidates):
        raise ValueError("unequal numbers of paired seeds")
    for baseline, candidate in zip(baselines, candidates):
        for field in ("seed", "answer_tokens_seen", "train_evaluation"):
            if baseline[field] != candidate[field]:
                raise ValueError(f"paired runs disagree on {field}: seed {baseline['seed']}")


def validate_freeze(root, project_root=ROOT):
    root, project_root = Path(root).resolve(), Path(project_root).resolve()
    path = root / "selection.json"
    frozen = read_json(path)
    digest = file_hash(path)
    if frozen.get("validation_used") is not False:
        raise ValueError("selection must be frozen before held-out test access")
    if frozen.get("selection_seeds") != SEARCH_SEEDS or frozen.get("fresh_seeds") != FRESH_SEEDS:
        raise ValueError("selection or fresh confirmation seeds differ from the approved protocol")
    if frozen.get("status") not in ("qualified", "no_qualified_candidate"):
        raise ValueError("search must be finished and its selection frozen")
    provenance = {name: file_hash(source_path(name, project_root)) for name in TRAIN_PROVENANCE}
    if frozen["status"] == "no_qualified_candidate":
        return {"freeze": frozen, "freeze_sha256": digest, "training_provenance": provenance, "qualified": False}
    hash_architecture, parameter_count = spec_tools()
    baseline, primary = frozen["baseline"], frozen["primary"]
    if baseline["frame"] != BASELINE_FRAME or baseline["parameters"] != BASELINE_PARAMETERS:
        raise ValueError("reference differs from the original 27,144,960-parameter baseline")
    from nas.cuda_search import baseline_spec, canonical
    if canonical(baseline["spec"]) != canonical(baseline_spec()):
        raise ValueError("reference architecture differs from the original GELU baseline")
    if not 0 < primary["parameters"] < BASELINE_PARAMETERS:
        raise ValueError("selected architecture must have strictly fewer total parameters")
    records = latest_registry(root)
    run_by_path = {str(Path(r["path"]).resolve()): r for r in records.values() if r.get("path")}
    groups = []
    for architecture in (baseline, primary):
        spec_path = inside(architecture["spec_path"], root)
        actual = read_json(spec_path)
        if (hash_architecture(architecture["frame"], actual) != architecture["architecture_id"]
                or hash_architecture(architecture["frame"], architecture["spec"]) != architecture["architecture_id"]
                or parameter_count(architecture["frame"], actual) != architecture["parameters"]):
            raise ValueError("frozen architecture ID/spec/frame/parameter count disagree")
        frame = architecture["frame"]
        if any(frame.get(k) != BASELINE_FRAME[k] for k in ("vocab_size", "max_len", "dropout")):
            raise ValueError("architecture changed a fixed vocabulary/context/dropout setting")
        rows = []
        for seed in SEARCH_SEEDS:
            run_path = inside(architecture["run_paths"][str(seed)], root)
            ledger = run_by_path.get(str(run_path), {})
            if ledger.get("status") != "completed" or ledger.get("job", {}).get("seed") != seed:
                raise ValueError("frozen selection cites a missing/incomplete search run")
            rows.append(validate_training_run(run_path, architecture, seed, provenance))
        measured = statistics.mean(r["best_dev_loss"] for r in rows)
        if not math.isclose(measured, architecture["mean_dev_loss"], abs_tol=1e-10):
            raise ValueError("frozen selection mean differs from the three recorded search runs")
        groups.append(rows)
    paired_controls(*groups)
    if primary["mean_dev_loss"] > baseline["mean_dev_loss"]:
        raise ValueError("frozen candidate does not match or beat baseline mean search-dev CE")
    if frozen.get("primary_architecture_id") != primary["architecture_id"]:
        raise ValueError("frozen primary IDs disagree")
    for candidate in frozen.get("qualified_candidates", []):
        if (candidate["parameters"] < primary["parameters"]
                and candidate["mean_dev_loss"] <= baseline["mean_dev_loss"]):
            raise ValueError("frozen candidate is not the smallest qualifying candidate")
    if file_hash(path) != digest:
        raise ValueError("selection changed while checking it")
    return {"freeze": frozen, "freeze_sha256": digest, "training_provenance": provenance,
            "qualified": True, "search_baseline_runs": groups[0], "search_candidate_runs": groups[1]}


def training_command(project_root, path, architecture, seed):
    frame = architecture["frame"]
    command = [*python_command("cuda_train.py", project_root), "--out", str(path),
               "--spec", architecture["spec_path"], "--seed", str(seed)]
    for option, key in (("layers", "n_layers"), ("d-model", "d_model"), ("heads", "n_heads"),
                        ("d-ff", "d_ff"), ("dropout", "dropout")):
        command.extend(["--" + option, str(frame[key])])
    for key, value in FIXED_ARGS.items():
        if key != "compile":
            command.extend(["--" + key.replace("_", "-"), str(value)])
    command.extend(["--compile", "--checkpoint-every", "0", "--cpu-threads", "4", "--device", "cuda",
                    "--max-params", str(BASELINE_PARAMETERS if architecture["parameters"] == BASELINE_PARAMETERS
                                        else BASELINE_PARAMETERS - 1)])
    return command


def stop_process(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        except ProcessLookupError:
            pass


class Confirmation:
    def __init__(self, root, verified, gpus, deadline, project_root=ROOT, job_timeout=1800):
        self.root, self.verified, self.gpus = root, verified, gpus
        self.deadline, self.project_root, self.job_timeout = deadline, project_root, job_timeout
        self.running, self.locks = {}, []
        self.records = latest_registry(root)
        self.state_path = root / "finalization.json"
        self.state = {"started_utc": now().isoformat(), "deadline_utc": deadline.isoformat(),
                      "selection_freeze_sha256": verified["freeze_sha256"], "pid": os.getpid(),
                      "hostname": socket.gethostname(), "test_used_for_selection": False,
                      "fresh_seeds_used_for_architecture_selection": False}

    def save(self, **updates):
        self.state.update(updates, updated_utc=now().isoformat())
        atomic_json(self.state_path, self.state)
        append_json(self.root / "finalization_events.jsonl", self.state)

    def unchanged(self):
        if file_hash(self.root / "selection.json") != self.verified["freeze_sha256"]:
            raise ValueError("frozen architecture selection changed")
        for name, digest in self.verified["training_provenance"].items():
            if file_hash(source_path(name, self.project_root)) != digest:
                raise ValueError(f"training source/data changed after freeze: {name}")

    def acquire(self):
        for gpu in self.gpus:
            stream = open(f"/tmp/nas-cuda-search-gpu-{gpu}.lock", "a")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                raise RuntimeError(f"GPU {gpu} is owned by an active scheduler")
            self.locks.append(stream)

    def record(self, row, **updates):
        row.update(updates, updated_utc=now().isoformat())
        self.records[row["job"]["run_id"]] = row
        append_json(self.root / "run_registry.jsonl", row)

    def jobs(self):
        queue = []
        for seed in FRESH_SEEDS:
            for role, key in (("baseline", "baseline"), ("winner", "primary")):
                architecture = self.verified["freeze"][key]
                run_id = f"f{len(queue) + 1:05d}-{architecture['architecture_id'][:10]}-s{seed}"
                queue.append({"job": {"run_id": run_id, "architecture_id": architecture["architecture_id"],
                                      "spec": architecture["name"], "seed": seed, "phase": "final", "steps": 5000},
                              "role": role, "path": str(self.root / "attempts" / run_id),
                              "architecture_sha256": architecture["architecture_id"], "architecture": architecture,
                              "parameters": architecture["parameters"], "frame": architecture["frame"],
                              "spec_path": architecture["spec_path"], "host": {"hostname": socket.gethostname()},
                              "source_hashes": self.verified["training_provenance"],
                              "selection_freeze_sha256": self.verified["freeze_sha256"]})
        return queue

    def previous_attempt(self, row):
        """Keep failed attempts, and never repeat a completed architecture/seed."""
        previous = [r for r in self.records.values() if r.get("job", {}).get("phase") == "final"
                    and r["job"].get("architecture_id") == row["job"]["architecture_id"]
                    and r["job"].get("seed") == row["job"]["seed"]]
        completed = [r for r in previous if r["status"] == "completed"]
        if len(completed) > 1:
            raise ValueError("multiple completed final runs exist for the same architecture and seed")
        if completed:
            return completed[0]
        for old in previous:
            if old.get("selection_freeze_sha256") != self.verified["freeze_sha256"]:
                raise ValueError("a previous final training attempt used a different selection")
            if old["status"] == "running":
                try:
                    os.kill(old["pid"], 0)
                except ProcessLookupError:
                    self.record(old, status="interrupted", ended_utc=now().isoformat(),
                                failure_reason="previous finalizer stopped; trainer PID no longer exists")
                else:
                    raise RuntimeError(f"previous final trainer PID {old['pid']} may still be active; refusing a duplicate")
            elif old["status"] == "queued":
                self.record(old, status="not_admitted", failure_reason="previous finalizer stopped before launch")
        if previous:
            base = row["job"]["run_id"]
            attempt = len(previous) + 1
            row["job"]["run_id"] = f"{base}-attempt{attempt}"
            row["path"] = str(self.root / "attempts" / row["job"]["run_id"])
            row["retry_of"] = [r["job"]["run_id"] for r in previous]
            row["attempt"] = attempt
        return None

    def train(self):
        self.save(status="fresh_seed_training")
        jobs, queue, completed = self.jobs(), [], []
        for row in jobs:
            existing = self.previous_attempt(row)
            if existing:
                if existing.get("selection_freeze_sha256") != self.verified["freeze_sha256"]:
                    raise ValueError("a previous final training attempt used a different selection")
                run = validate_training_run(Path(existing["path"]), row["architecture"], row["job"]["seed"],
                                            self.verified["training_provenance"], checkpoints=True)
                if existing.get("checkpoints", {}).get("best.pt", {}).get("sha256") != run["checkpoint_sha256"]:
                    raise ValueError("previous final checkpoint differs from its run ledger")
                completed.append({**run, "role": row["role"]})
                continue
            self.record(row, status="queued", queued_utc=now().isoformat())
            queue.append(row)
        while queue or self.running:
            if (self.deadline - now()).total_seconds() < 600:
                for row in queue:
                    self.record(row, status="not_admitted", failure_reason="reserved final test/report time")
                raise TimeoutError("fresh-seed training did not finish with 10 minutes reserved for test/report")
            for gpu in self.gpus:
                if gpu in self.running or not queue:
                    continue
                row = queue.pop(0)
                path = Path(row["path"])
                if path.exists():
                    raise ValueError(f"refusing to overwrite an existing run: {path}")
                command = training_command(self.project_root, path, row["architecture"], row["job"]["seed"])
                log = (self.root / "logs" / (row["job"]["run_id"] + ".log")).open("x")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED=str(row["job"]["seed"]),
                           OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
                try:
                    process = subprocess.Popen(command, cwd=self.project_root, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException as error:
                    log.close()
                    self.record(row, status="failed", failure_reason=f"launch: {error}")
                    raise
                self.running[gpu] = {"row": row, "process": process, "log": log, "started": time.monotonic()}
                self.record(row, status="running", started_utc=now().isoformat(), command=command,
                            pid=process.pid, physical_gpu={"physical_index": gpu})
            for gpu, item in list(self.running.items()):
                row, process = item["row"], item["process"]
                elapsed = time.monotonic() - item["started"]
                if elapsed > self.job_timeout and process.poll() is None:
                    stop_process(process)
                    item["log"].close()
                    self.record(row, status="timed_out", wall_seconds=elapsed, ended_utc=now().isoformat())
                    del self.running[gpu]
                    raise TimeoutError(f"confirmation training exceeded {self.job_timeout}s")
                if process.poll() is None:
                    continue
                item["log"].close()
                del self.running[gpu]
                details = {"returncode": process.returncode, "wall_seconds": elapsed, "ended_utc": now().isoformat()}
                try:
                    if process.returncode:
                        raise RuntimeError(f"trainer exited {process.returncode}")
                    self.unchanged()
                    path = Path(row["path"])
                    run = validate_training_run(path, row["architecture"], row["job"]["seed"],
                                                self.verified["training_provenance"], checkpoints=True)
                    checkpoint_metadata = {p.name: {"path": str(p), "sha256": file_hash(p),
                                                    "bytes": p.stat().st_size, "pruned": False} for p in path.glob("*.pt")}
                    self.record(row, status="completed", summary=read_json(path / "summary.json"),
                                checkpoints=checkpoint_metadata, **details)
                    completed.append({**run, "role": row["role"]})
                except BaseException as error:
                    self.record(row, status="failed", failure_reason=f"{type(error).__name__}: {error}", **details)
                    raise
                try:
                    from nas.cuda_report import render_run
                    render_run(path)
                except Exception as error:
                    append_json(self.root / "finalization_events.jsonl", {"event": "plot_failed", "run": str(path), "error": str(error)})
            self.save(completed_fresh_runs=len(completed), gpu_assignments={str(g): v["row"]["job"]["run_id"] for g, v in self.running.items()})
            if queue or self.running:
                time.sleep(2)
        completed.sort(key=lambda row: (row["role"] != "baseline", row["seed"]))
        if [r["seed"] for r in completed[:5]] != FRESH_SEEDS or [r["seed"] for r in completed[5:]] != FRESH_SEEDS:
            raise ValueError("confirmation must contain five exact fresh seed pairs")
        paired_controls(completed[:5], completed[5:])
        return completed

    def evaluate(self, runs):
        self.unchanged()
        evaluation_root = self.root / "final-evaluations"
        evaluation_root.mkdir(exist_ok=True)
        provenance = {**self.verified["training_provenance"],
                      "data/validation.json": file_hash(self.project_root / "data/validation.json"),
                      "cuda_evaluate.py": file_hash(source_path("cuda_evaluate.py", self.project_root)),
                      "cuda_minparams_finalize.py": file_hash(Path(__file__))}
        self.save(status="held_out_test_evaluation", test_started=True, data_and_source_provenance=provenance)
        queue, outputs = list(runs), {}
        while queue or self.running:
            if (self.deadline - now()).total_seconds() < 60:
                raise TimeoutError("held-out evaluator reached its reporting time reserve")
            for gpu in self.gpus:
                if gpu in self.running or not queue:
                    continue
                run = queue.pop(0)
                base = Path(run["run"]).name
                existing = sorted(evaluation_root.glob(base + "*.json"))
                output = None
                for possible in existing:
                    if possible.with_suffix(".conversations.npz").exists():
                        checked_evaluation(possible, run)
                        output = possible
                        break
                if output:
                    outputs[run["run"]] = output
                    continue
                output = evaluation_root / (base + ".json")
                attempt = 1
                while output.exists() or output.with_suffix(".conversations.npz").exists() or output.with_suffix(".log").exists():
                    attempt += 1
                    output = evaluation_root / (base + f"-attempt{attempt}.json")
                command = [*python_command("cuda_minparams_finalize.py", self.project_root), "--evaluate-run", run["run"],
                           "--evaluation-out", str(output), "--expected-checkpoint-sha256", run["checkpoint_sha256"]]
                log = output.with_suffix(".log").open("x")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
                process = subprocess.Popen(command, cwd=self.project_root, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                row = {"event": "test_evaluation", "run": run["run"], "seed": run["seed"], "role": run["role"],
                       "checkpoint_sha256": run["checkpoint_sha256"], "status": "running", "started_utc": now().isoformat(),
                       "physical_gpu": gpu, "pid": process.pid, "output": str(output), "command": command,
                       "attempt": attempt}
                append_json(self.root / "evaluation_registry.jsonl", row)
                self.running[gpu] = {"evaluation": row, "run": run, "output": output, "process": process,
                                     "log": log, "started": time.monotonic()}
            for gpu, item in list(self.running.items()):
                if item["process"].poll() is None:
                    continue
                item["log"].close()
                del self.running[gpu]
                row = item["evaluation"]
                try:
                    if item["process"].returncode:
                        raise RuntimeError(f"test evaluator exited {item['process'].returncode}")
                    checked_evaluation(item["output"], item["run"])
                    outputs[item["run"]["run"]] = item["output"]
                    row.update(status="completed", json_sha256=file_hash(item["output"]),
                               conversations_sha256=file_hash(item["output"].with_suffix(".conversations.npz")))
                except BaseException as error:
                    row.update(status="failed", error=str(error))
                    raise
                finally:
                    row.update(ended_utc=now().isoformat(), wall_seconds=time.monotonic() - item["started"])
                    append_json(self.root / "evaluation_registry.jsonl", row)
            if queue or self.running:
                time.sleep(2)
        self.unchanged()
        self.save(test_completed=True)
        return combine_evaluations(runs, outputs, self.root), provenance

    def close(self):
        for item in self.running.values():
            stop_process(item["process"])
            item["log"].close()
            if "row" in item:
                self.record(item["row"], status="interrupted", ended_utc=now().isoformat(),
                            failure_reason="finalizer stopped before completion")
            else:
                append_json(self.root / "evaluation_registry.jsonl", {**item["evaluation"], "status": "interrupted",
                                                                      "ended_utc": now().isoformat()})
        self.running.clear()
        for stream in self.locks:
            stream.close()
        self.locks.clear()


def checked_evaluation(path, run):
    import numpy as np
    row = read_json(path)
    if (row.get("run") != run["run"] or row.get("checkpoint_sha256") != run["checkpoint_sha256"]
            or row.get("config_sha256") != run["config_sha256"] or row.get("seed") != run["seed"]
            or row.get("conversations") != 2599 or row.get("split") != "validation"
            or row.get("precision") != "bf16" or not math.isfinite(row["loss"])):
        raise ValueError("test result does not describe the exact frozen checkpoint and held-out split")
    if file_hash(Path(run["checkpoint"])) != run["checkpoint_sha256"]:
        raise ValueError("checkpoint changed since freeze")
    with np.load(path.with_suffix(".conversations.npz"), allow_pickle=False) as archive:
        sums, counts = archive["sums"], archive["counts"]
        if (len(sums) != 2599 or len(counts) != 2599 or not np.isfinite(sums).all()
                or not (counts > 0).all() or int(counts.sum()) != row["answer_tokens"]
                or not math.isclose(float(sums.sum() / counts.sum()), row["loss"], abs_tol=1e-10)):
            raise ValueError("test summary and per-conversation archive disagree")
    return row


def evaluate_one(run, output, expected_checkpoint_sha256):
    """Worker: reuse exactly the existing answer-token CE implementation."""
    import numpy as np
    import torch
    from nas import data
    from nas.cuda_evaluate import load, per_conversation
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if output.exists() or output.with_suffix(".conversations.npz").exists():
        raise ValueError("held-out evaluation outputs are write-once")
    checkpoint = run / "best.pt"
    if file_hash(checkpoint) != expected_checkpoint_sha256:
        raise ValueError("checkpoint changed before held-out evaluation")
    examples = data.encode_split(data.tokenizer(), data.load_split("validation"))
    if len(examples) != 2599:
        raise ValueError("held-out evaluation must contain exactly 2599 conversations")
    model, config = load(run, "cuda", "best.pt")
    sums, counts = per_conversation(model, examples, "bf16", 0)
    if file_hash(checkpoint) != expected_checkpoint_sha256:
        raise ValueError("checkpoint changed during held-out evaluation")
    row = {"run": str(run), "seed": config["args"]["seed"], "parameters": config["parameters"],
           "checkpoint_sha256": expected_checkpoint_sha256, "config_sha256": file_hash(run / "config.json"),
           "split": "validation", "conversations": len(counts), "answer_tokens": int(counts.sum()),
           "precision": "bf16", "loss": float(sums.sum() / counts.sum()),
           "metric": "answer-token-weighted mean cross-entropy including answer EOS",
           "completed_utc": now().isoformat()}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Archive first and JSON second: no partially written JSON is accepted as complete.
    with output.with_suffix(".conversations.npz").open("xb") as stream:
        np.savez_compressed(stream, sums=sums, counts=counts)
    exclusive_json(output, row)
    print(json.dumps(row), flush=True)


def combine_evaluations(runs, outputs, root):
    import numpy as np
    from nas.cuda_evaluate import paired_multiseed_bootstrap
    per, results, archive = [], {}, {}
    for index, run in enumerate(runs):
        path = outputs[run["run"]]
        row = checked_evaluation(path, run)
        with np.load(path.with_suffix(".conversations.npz"), allow_pickle=False) as source:
            sums, counts = source["sums"].copy(), source["counts"].copy()
        per.append((sums, counts))
        archive[f"run_{index}_sums"], archive[f"run_{index}_counts"] = sums, counts
        results[run["run"]] = row
    comparison = paired_multiseed_bootstrap(per[:5], per[5:], bootstrap=5000, seed=20261111)
    results["difference_candidate_minus_baseline_across_seeds"] = comparison
    result = {"split": "validation", "display_split": "test", "baseline_count": 5,
              "precision": "bf16", "run_order": [r["run"] for r in runs], "results": results,
              "metric": "answer-token-weighted mean cross-entropy including answer EOS"}
    output = root / "final-test.json"
    if output.exists():
        if read_json(output) != result:
            raise ValueError("existing combined test result differs from immutable per-run outputs")
    else:
        archive_path = output.with_suffix(".conversations.npz")
        if archive_path.exists():
            with np.load(archive_path, allow_pickle=False) as previous:
                if set(previous.files) != set(archive) or any(not np.array_equal(previous[key], value) for key, value in archive.items()):
                    raise ValueError("previous combined archive differs from immutable per-run evaluation outputs")
        else:
            with archive_path.open("xb") as stream:
                np.savez_compressed(stream, **archive)
        exclusive_json(output, result)
    return result


def summarize(root, verified, runs, evaluated, provenance):
    def stats(rows):
        result = {"parameters": rows[0]["parameters"], "frame": rows[0]["frame"]}
        for name, values in (("dev", [r["best_dev_loss"] for r in rows]),
                             ("test", [evaluated["results"][r["run"]]["loss"] for r in rows]),
                             ("train_diagnostic", [r["train_eval_loss_at_best_dev"] for r in rows])):
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
                raise ValueError(f"nonfinite or missing {name} metric")
            result.update({name + "_mean": statistics.mean(values), name + "_std": statistics.stdev(values)})
        result["runs"] = [{**r, "test_loss": evaluated["results"][r["run"]]["loss"]} for r in rows]
        return result
    baseline, winner = stats(runs[:5]), stats(runs[5:])
    comparison = evaluated["results"]["difference_candidate_minus_baseline_across_seeds"]
    matched = comparison["point"] <= 0
    reduction = BASELINE_PARAMETERS - winner["parameters"]
    records = latest_registry(root)
    statuses = Counter(r["status"] for r in records.values())
    return {"status": "complete", "completed_utc": now().isoformat(),
            "objective": "Minimize total parameters subject to mean search-dev CE no worse than the fixed original baseline",
            "selection_rule": verified["freeze"].get("rule"),
            "winner_name": verified["freeze"]["primary"]["name"],
            "winner_architecture_id": verified["freeze"]["primary"]["architecture_id"],
            "winner_spec": verified["freeze"]["primary"]["spec"],
            "baseline": baseline, "winner": winner, "parameter_reduction": reduction,
            "parameter_reduction_fraction": reduction / BASELINE_PARAMETERS,
            "test_comparison": comparison, "test_mean_no_worse": matched,
            "test_interval_upper_no_worse": comparison["bootstrap_95"][1] <= 0,
            "public_claim_supported_by_mean": matched,
            "outcome": "smaller_model_matched_test_mean" if matched else "smaller_model_did_not_match_test_mean",
            "public_claim_rule": "No smaller-without-worse-test-loss claim if candidate mean test CE exceeds baseline; interval uncertainty is reported even when the mean matches.",
            "fresh_seed_dev_differences": [w["best_dev_loss"] - b["best_dev_loss"] for b, w in zip(runs[:5], runs[5:])],
            "search_baseline_mean_dev": verified["freeze"]["baseline"]["mean_dev_loss"],
            "search_winner_mean_dev": verified["freeze"]["primary"]["mean_dev_loss"],
            "selection_seeds": SEARCH_SEEDS, "fresh_seeds": FRESH_SEEDS,
            "fresh_seeds_used_for_architecture_selection": False, "test_used_for_selection": False,
            "reselection_after_test": False, "selection_freeze_sha256": verified["freeze_sha256"],
            "data_and_source_provenance": provenance, "run_statuses": dict(statuses),
            "completed_training_runs": statuses.get("completed", 0),
            "architectures_with_completed_runs": len({r["job"]["architecture_id"] for r in records.values() if r["status"] == "completed"}),
            "artifacts": {"selection": str(root / "selection.json"), "run_ledger": str(root / "run_registry.jsonl"),
                          "architecture_ledger": str(root / "architecture_registry.jsonl"),
                          "test": str(root / "final-test.json"), "test_conversations": str(root / "final-test.conversations.npz"),
                          "evaluation_ledger": str(root / "evaluation_registry.jsonl"),
                          "curves": str(root / "final-report/comparison.png")},
            "limitations": ["Search-dev means include architecture-selection effects; fresh confirmation seeds never trigger architecture reselection.",
                            "The interval resamples five observed paired training seeds and paired conversations; it does not establish behavior for every possible seed or dataset.",
                            "Matching the observed test mean is weaker evidence than an interval whose upper endpoint is at or below zero.",
                            "This held-out split was reported in the prior study; it is reused here after freezing this search's candidate, not a newly collected benchmark.",
                            "All training runs use 5000 optimizer steps and paired token streams. Architectural changes can alter runtime despite equal steps/data.",
                            "Train diagnostic CE covers a fixed 256-conversation subset of the 48,375 training conversations, not the entire training set."]}


def human_report(report):
    if report["status"] != "complete":
        return ("# Minimum-parameter search\n\n" + report["status"] + "\n\n" + report.get("reason", "") +
                "\n\nNo claim of a smaller model matching held-out baseline performance is supported.\n")
    baseline, winner = report["baseline"], report["winner"]
    comparison = report["test_comparison"]
    outcome = ("The smaller model matched or beat the baseline's observed mean test CE."
               if report["test_mean_no_worse"] else
               "The smaller model did not match baseline mean test CE; do not use it for the proposed tweet claim.")
    return (f"# Minimum-parameter search\n\n{outcome}\n\n"
            "| Metric | Original baseline | Frozen smaller model |\n|---|---:|---:|\n"
            f"| Parameters | {baseline['parameters']:,} | {winner['parameters']:,} |\n"
            f"| Fresh-seed mean dev CE | {baseline['dev_mean']:.6f} | {winner['dev_mean']:.6f} |\n"
            f"| Fresh-seed mean test CE | {baseline['test_mean']:.6f} | {winner['test_mean']:.6f} |\n"
            f"| Mean training diagnostic CE | {baseline['train_diagnostic_mean']:.6f} | {winner['train_diagnostic_mean']:.6f} |\n\n"
            f"Parameter reduction: **{report['parameter_reduction_fraction']:.2%}** ({report['parameter_reduction']:,}).\n\n"
            f"Test CE difference (candidate minus baseline): **{comparison['point']:+.6f}**; "
            f"paired bootstrap 95% interval [{comparison['bootstrap_95'][0]:+.6f}, {comparison['bootstrap_95'][1]:+.6f}].\n\n"
            "Architecture selection used seeds 0, 1, 2 and the 1,000-conversation dev set. "
            "The architecture was frozen before five new paired seeds (100–104) and the 2,599-conversation test evaluation. "
            "Every training run started from scratch and completed exactly 5,000 optimizer steps. "
            "Each checkpoint was selected by the same best-dev rule; no architecture was reselected after confirmation.\n\n"
            "## Paired seed results\n\n| Seed | Baseline test CE | Candidate test CE | Difference |\n|---|---:|---:|---:|\n" +
            "".join(f"| {b['seed']} | {b['test_loss']:.6f} | {w['test_loss']:.6f} | {w['test_loss']-b['test_loss']:+.6f} |\n"
                    for b, w in zip(baseline["runs"], winner["runs"])) +
            "\n## Limits\n\n" + "\n".join("- " + value for value in report["limitations"]) + "\n")


def finalize(root, gpus, deadline, project_root=ROOT, job_timeout=1800):
    root, project_root = Path(root).resolve(), Path(project_root).resolve()
    if not root.is_dir():
        raise ValueError("search output directory does not exist")
    lock_path = root / ".minparams-finalization.lock"
    lock = open(lock_path, "a")
    driver = None
    handlers = {}
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verified = validate_freeze(root, project_root)
        report_path = root / "FINAL_REPORT.json"
        if report_path.exists():
            previous = read_json(report_path)
            if previous.get("selection_freeze_sha256") != verified["freeze_sha256"]:
                raise ValueError("existing report has a different frozen selection")
            if previous["status"] in ("complete", "no_qualified_candidate"):
                return previous
            if previous["status"] != "incomplete_confirmation":
                raise ValueError("unexpected existing finalization report status")
            append_json(root / "finalization_events.jsonl", {"event": "retry_incomplete_confirmation",
                        "previous_report": previous, "time_utc": now().isoformat()})
        if not verified["qualified"]:
            report = {"status": "no_qualified_candidate", "completed_utc": now().isoformat(),
                      "reason": "No smaller architecture matched baseline mean dev CE across all three required search seeds.",
                      "selection_freeze_sha256": verified["freeze_sha256"], "test_accessed": False,
                      "public_claim_supported_by_mean": False}
        else:
            driver = Confirmation(root, verified, gpus, deadline, project_root, job_timeout)
            def interrupted(signum, frame):
                raise InterruptedError(f"finalizer received signal {signum}")
            handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
            for number in handlers:
                signal.signal(number, interrupted)
            driver.acquire()
            driver.unchanged()
            runs = driver.train()
            atomic_json(root / "final_confirmation.json", {"status": "frozen_checkpoints", "frozen_utc": now().isoformat(),
                        "selection_freeze_sha256": verified["freeze_sha256"], "runs": runs,
                        "fresh_seeds_used_for_architecture_selection": False, "test_accessed": False})
            evaluated, provenance = driver.evaluate(runs)
            report = summarize(root, verified, runs, evaluated, provenance)
            try:
                from nas.cuda_report import render_leaderboard
                directories = [r["path"] for r in latest_registry(root).values() if r["status"] == "completed"]
                render_leaderboard(directories, root / "all-run-report", top=12, run_plots=False)
                render_leaderboard([r["run"] for r in runs], root / "final-report", top=10, run_plots=False)
            except Exception as error:
                report["limitations"].append(f"Plot rendering failed: {type(error).__name__}: {error}")
            driver.unchanged()
        atomic_json(report_path, report)
        (root / "FINAL_REPORT.md").write_text(human_report(report))
        if driver:
            driver.save(status="complete", final_report_sha256=file_hash(report_path), outcome=report["outcome"])
        return report
    except BaseException as error:
        if driver:
            driver.save(status="failed", error=f"{type(error).__name__}: {error}")
            if not (root / "FINAL_REPORT.json").exists():
                report = {"status": "incomplete_confirmation", "completed_utc": now().isoformat(),
                          "reason": f"{type(error).__name__}: {error}",
                          "selection_freeze_sha256": driver.verified["freeze_sha256"],
                          "public_claim_supported_by_mean": False, "reselection_after_test": False}
                atomic_json(root / "FINAL_REPORT.json", report)
                (root / "FINAL_REPORT.md").write_text(human_report(report))
        raise
    finally:
        if driver:
            driver.close()
        for number, handler in handlers.items():
            signal.signal(number, handler)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--root", dest="out", type=Path)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--compile", action="store_true", default=True,
                        help="accepted for workflow compatibility; compilation is always required by the fixed recipe")
    parser.add_argument("--deadline-utc", "--deadline", dest="deadline_utc", type=parse_utc)
    parser.add_argument("--job-timeout-seconds", type=float, default=1800)
    parser.add_argument("--evaluate-run", type=Path)
    parser.add_argument("--evaluation-out", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    args = parser.parse_args()
    if args.evaluate_run:
        if not args.evaluation_out or not args.expected_checkpoint_sha256:
            parser.error("evaluation worker requires output and checkpoint digest")
        evaluate_one(args.evaluate_run.resolve(), args.evaluation_out.resolve(), args.expected_checkpoint_sha256)
        return
    if not args.out or not args.deadline_utc:
        parser.error("--out and --deadline-utc are required")
    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus or len(gpus) != len(set(gpus)) or min(gpus) < 0 or args.job_timeout_seconds <= 0:
        parser.error("GPU indexes must be unique nonnegative integers and timeout positive")
    report = finalize(args.out, gpus, args.deadline_utc, job_timeout=args.job_timeout_seconds)
    print(json.dumps({"event": "minparams_final_report", "status": report["status"],
                      "report": str(args.out / "FINAL_REPORT.json"),
                      "public_claim_supported_by_mean": report.get("public_claim_supported_by_mean", False)}), flush=True)


if __name__ == "__main__":
    main()
