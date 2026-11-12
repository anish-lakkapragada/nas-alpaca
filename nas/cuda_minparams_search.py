"""Durable parameter-minimization search with a fixed 5,000-step recipe.

The original 27,144,960-parameter baseline is measured on seeds 0/1/2. Search
uses only train/dev data and minimizes unique parameters subject to paired
three-seed mean dev CE <= baseline. The separately launched finalizer owns
five fresh seed pairs and the held-out test evaluation after selection freezes.
"""
from __future__ import annotations

from nas.paths import PROJECT_ROOT, PROVENANCE_SUPPORT, python_command, source_path

import argparse
import copy
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

from nas.cuda_search import (Scheduler as LegacyScheduler, Job, FRAME, RECIPE,
                         BASELINE_PARAMETERS, utc_now, iso, parse_utc,
                         atomic_json, file_hash)

ROOT = PROJECT_ROOT
SEARCH_SEEDS = (0, 1, 2)
FRESH_SEEDS = (100, 101, 102, 103, 104)
PROTOCOL = {
    "objective": "Minimize unique parameters subject to mean best-dev CE <= paired original baseline on seeds 0/1/2.",
    "baseline_frame": FRAME, "baseline_parameters": BASELINE_PARAMETERS,
    "recipe": RECIPE, "selection_seeds": list(SEARCH_SEEDS), "fresh_seeds": list(FRESH_SEEDS),
    "allowed": ["change depth", "change embedding/residual width with head dimension 64",
                "widen or narrow individual FFNs", "arbitrary attention/FFN order and repetition",
                "layer loops with optional input reinjection", "attention sharing including its LayerNorm",
                "FFN sharing including its LayerNorm", "gelu/relu/relu2/silu/gelu_tanh",
                "attention deletion", "FFN deletion", "transpose-tied FFN projections",
                "fixed or learned residual branch scalars"],
    "parameter_rule": "All candidates strictly below 27,144,960; individual components may grow; all embeddings/norms/scalars counted.",
    "data_rule": "Unchanged tokenizer/train/dev split/packing/batches/data order for each paired seed; no test data read.",
    "selection": "Exactly one smallest qualified architecture frozen before fresh seed training; CE breaks equal-parameter ties.",
    "checkpoint_rule": "Best dev checkpoint from evaluations every 250 steps; every completed training run has exactly 5000 steps.",
    "optimizer": "AdamW without bias correction, betas .9/.95, epsilon 1e-8; unchanged CUDA trainer.",
    "confirmation": "One-seed screening, then seeds1/2 on promising candidates across parameter budgets; baseline paired on same seeds.",
    "adaptation": "Preserve size/loss Pareto frontier and best candidate per size band; mutate architecture, never training recipe.",
    "provenance": "Append-only architecture, rejection and run registries; every attempt has a unique directory; no failure hidden.",
    "retention": "Preserve baseline, all feasible confirmed checkpoints, and parameter/loss frontier; only checkpoint binaries may be pruned.",
    "validation": "Never read by search scheduler; separate finalizer evaluates frozen winner and baseline on fresh seeds100..104.",
}


def baseline_spec():
    return {"name": "baseline", "description": "Original 27,144,960-parameter GPT baseline",
            "layers": [{"attn": f"A{i}", "ffn": f"F{i}"} for i in range(8)], "ffns": {}}


def canonical(frame, spec):
    """Hash effective frame and graph, including sharing, execution order and scales."""
    maps = {"attn": {}, "ffn": {}}
    layers, ffns = [], {}
    for source in spec["layers"]:
        layer = {}
        for kind in ("attn", "ffn"):
            key = source.get(kind)
            if key is None:
                layer[kind] = None
                continue
            if key not in maps[kind]:
                maps[kind][key] = ("A" if kind == "attn" else "F") + str(len(maps[kind]))
            layer[kind] = maps[kind][key]
            scale = source.get(kind + "_scale", 1.)
            if isinstance(scale, dict):
                scale = {"init": float(scale["init"]), "learned": True} if scale["learned"] else float(scale["init"])
            else:
                scale = float(scale)
            layer[kind + "_scale"] = scale
            if kind == "ffn":
                opts = spec.get("ffns", {}).get(key, {})
                ffns[layer[kind]] = {"hidden": opts.get("hidden", frame["d_ff"]),
                                     "act": opts.get("act", "gelu"),
                                     "transpose_tie": opts.get("transpose_tie", False)}
        layer["loop"] = source.get("loop", 1)
        layer["inject"] = source.get("inject", False)
        layer["order"] = source.get("order", "attn_ffn") if all(source.get(k) is not None for k in ("attn", "ffn")) else "attn_ffn"
        layers.append(layer)
    return {"frame": dict(frame), "spec": {"search_space": spec.get("search_space", "legacy"), "layers": layers, "ffns": ffns}}


def architecture_hash(frame, spec):
    return hashlib.sha256(json.dumps(canonical(frame, spec), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parameter_count(frame, spec):
    d = frame["d_model"]
    count = (frame["vocab_size"] + frame["max_len"]) * d + 2 * d
    for kind in ("attn", "ffn"):
        keys = {layer.get(kind) for layer in spec["layers"]} - {None}
        for key in keys:
            if kind == "attn":
                count += 4 * d * d + 2 * d
            else:
                opts = spec.get("ffns", {}).get(key, {})
                count += d * opts.get("hidden", frame["d_ff"]) * (1 if opts.get("transpose_tie") else 2) + 2 * d
        count += sum(isinstance(layer.get(kind + "_scale"), dict) and layer[kind + "_scale"].get("learned") is True
                     for layer in spec["layers"] if layer.get(kind) is not None)
    return count


def validate(frame, spec, baseline=False):
    if set(frame) != set(FRAME) or any(frame[key] != FRAME[key] for key in ("vocab_size", "max_len", "dropout")):
        raise ValueError("tokenizer/context/dropout frame must remain fixed")
    for key in ("n_layers", "d_model", "n_heads", "d_ff"):
        if type(frame[key]) is not int or frame[key] <= 0:
            raise ValueError("positive integer model dimensions required")
    if frame["d_model"] % frame["n_heads"]:
        raise ValueError("attention dimensions incompatible")
    layers = spec.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("nonempty layer list required")
    if not baseline and spec.get("search_space") != "expanded":
        raise ValueError("candidate must opt into expanded approved search space")
    applications = 0
    for layer in layers:
        if set(layer) - {"attn", "ffn", "loop", "inject", "order", "attn_scale", "ffn_scale"}:
            raise ValueError("unapproved layer field")
        if layer.get("attn") is None and layer.get("ffn") is None:
            raise ValueError("empty sublayer entry")
        if type(layer.get("loop", 1)) is not int or layer.get("loop", 1) < 1:
            raise ValueError("invalid loop")
        applications += layer.get("loop", 1)
        if type(layer.get("inject", False)) is not bool or (layer.get("inject") and layer.get("loop", 1) == 1):
            raise ValueError("invalid input reinjection")
        if layer.get("order", "attn_ffn") not in ("attn_ffn", "ffn_attn"):
            raise ValueError("invalid order")
        for kind in ("attn", "ffn"):
            key = layer.get(kind)
            if key is not None and (not isinstance(key, str) or not key or "." in key):
                raise ValueError("invalid weight-sharing ID")
            if key is None and kind + "_scale" in layer:
                raise ValueError("scale attached to deleted branch")
            scale = layer.get(kind + "_scale", 1.)
            if isinstance(scale, dict):
                if set(scale) != {"init", "learned"} or type(scale["learned"]) is not bool:
                    raise ValueError("invalid learned scale schema")
                scale = scale["init"]
            if isinstance(scale, bool) or not isinstance(scale, (float, int)) or not math.isfinite(scale):
                raise ValueError("invalid residual scale")
    if frame["n_layers"] != applications:
        raise ValueError("frame depth must equal executed layer-entry applications")
    for opts in spec.get("ffns", {}).values():
        if set(opts) - {"hidden", "act", "transpose_tie"}:
            raise ValueError("unapproved FFN option")
        if opts.get("act", "gelu") not in ("gelu", "relu", "relu2", "silu", "gelu_tanh"):
            raise ValueError("unapproved activation")
        if type(opts.get("hidden", frame["d_ff"])) is not int or opts.get("hidden", frame["d_ff"]) <= 0:
            raise ValueError("invalid FFN hidden width")
        if type(opts.get("transpose_tie", False)) is not bool:
            raise ValueError("invalid transpose tie")
    count = parameter_count(frame, spec)
    if baseline:
        if frame != FRAME or architecture_hash(frame, spec) != architecture_hash(FRAME, baseline_spec()) or count != BASELINE_PARAMETERS:
            raise ValueError("baseline must be the unchanged original architecture")
    elif count >= BASELINE_PARAMETERS:
        raise ValueError(f"total {count} must be strictly below {BASELINE_PARAMETERS}")


def used_ffns(spec):
    return list(dict.fromkeys(layer["ffn"] for layer in spec["layers"] if layer.get("ffn") is not None))


def normalize_frame(frame, spec):
    frame["n_layers"] = sum(layer.get("loop", 1) for layer in spec["layers"])
    return frame, spec


def standard(depth, width, hidden, act="relu2", reverse=False):
    frame = {**FRAME, "n_layers": depth, "d_model": width, "n_heads": width // 64, "d_ff": hidden}
    spec = {"search_space": "expanded", "layers": [{"attn": f"A{i}", "ffn": f"F{i}",
             "order": "ffn_attn" if reverse else "attn_ffn"} for i in range(depth)],
            "ffns": {f"F{i}": {"hidden": hidden, "act": act} for i in range(depth)}}
    return frame, spec


def breadth_candidates():
    """A bounded, diverse first batch; later proposals adapt to measured frontiers."""
    rows = []
    def add(label, pair):
        frame, spec = pair
        rows.append((label, frame, spec))
    # Loss anchors from the previous CUDA experiment, with new measurements.
    for width in (1472, 1280, 1024, 768):
        for reverse in (False, True):
            add(f"anchor-d384-f{width}-r{int(reverse)}", standard(8, 384, width, reverse=reverse))
    # Cover parameter bands and depth/width allocation, interleaved by frame.
    shapes = [(8,320,1280), (10,320,1024), (12,320,1024), (8,256,1024),
              (12,256,1024), (16,256,1024), (16,256,1408), (20,256,768),
              (12,192,768), (18,192,768), (24,192,768), (6,448,896),
              (6,384,1536), (10,384,768), (4,512,768), (6,320,1536),
              (10,256,1536), (16,320,512), (24,256,512), (8,192,1536),
              (12,128,1024), (24,128,1024), (6,256,2048), (4,384,3072)]
    for act, reverse in (("gelu",False), ("relu2",False), ("relu2",True)):
        for depth,width,hidden in shapes:
            add(f"shape-n{depth}-d{width}-f{hidden}-{act}-r{int(reverse)}", standard(depth,width,hidden,act,reverse))
    # Tie/sharing/loop schedules can increase computation at low parameter cost.
    for depth,width,hidden in ((8,384,1536),(12,320,1280),(16,256,1024),(12,256,1536),(16,192,1024)):
        for mode in ("attn-half", "ffn-half", "both-half", "loop2", "loop2-inject", "transpose", "ffa", "faa"):
            frame,spec = standard(depth,width,hidden,reverse=True)
            if mode.endswith("half"):
                for i,layer in enumerate(spec["layers"]):
                    for kind in ("attn","ffn"):
                        if mode in (kind+"-half","both-half"):
                            layer[kind] = ("A" if kind=="attn" else "F") + str(i % max(2,depth//2))
            elif mode.startswith("loop2"):
                spec["layers"] = spec["layers"][:depth//2]
                for layer in spec["layers"]:
                    layer.update(loop=2, inject=mode.endswith("inject"))
            elif mode == "transpose":
                for opts in spec["ffns"].values():
                    opts["transpose_tie"] = True
            else:
                layers=[]
                for i,layer in enumerate(spec["layers"]):
                    if mode == "ffa":
                        layers.append({"ffn":f"Fextra{i}","attn":None})
                        spec["ffns"][f"Fextra{i}"]={"hidden":max(128,hidden//2),"act":"relu2"}
                        spec["ffns"][f"F{i}"]["hidden"]=max(128,hidden//2)
                    else:
                        layers.append({"attn":f"A{i}","ffn":None})
                    layers.append(layer)
                spec["layers"]=layers
            add(f"schedule-n{depth}-d{width}-f{hidden}-{mode}", normalize_frame(frame,spec))
    # Local expanded FFN widths: redistribution may grow selected components.
    for depth,width,hidden in ((8,320,1280),(12,256,1024),(8,384,1024)):
        for pattern in ("late-wide","early-wide","alternating","learned-scales"):
            frame,spec=standard(depth,width,hidden,reverse=True)
            for i,key in enumerate(used_ffns(spec)):
                if pattern=="late-wide":
                    factor=.5 if i<depth//2 else 1.5
                elif pattern=="early-wide":
                    factor=1.5 if i<depth//2 else .5
                else:
                    factor=.5 if i%2 else 1.5
                if pattern!="learned-scales":
                    spec["ffns"][key]["hidden"]=max(64,round(hidden*factor/64)*64)
            if pattern=="learned-scales":
                for layer in spec["layers"]:
                    layer["attn_scale"]={"init":1.,"learned":True}
                    layer["ffn_scale"]={"init":1.,"learned":True}
            add(f"redistribute-n{depth}-d{width}-{pattern}",(frame,spec))
    return rows


def mutations(frame, parent, generation):
    rows=[]
    rng=random.Random(20261111+generation*1009+int(architecture_hash(frame,parent)[:12],16))
    def offer(label,f,s):
        s["search_space"]="expanded"
        normalize_frame(f,s)
        rows.append((label,f,s))
    keys=used_ffns(parent)
    # Global width changes keep every FFN proportionate, or retain FFN capacity.
    for delta in (-64,64):
        width=frame["d_model"]+delta
        if 128<=width<=512:
            for proportional in (True,False):
                f,s=copy.deepcopy(frame),copy.deepcopy(parent)
                f.update(d_model=width,n_heads=width//64)
                if proportional:
                    ratio=width/frame["d_model"]
                    f["d_ff"]=max(64,round(f["d_ff"]*ratio/64)*64)
                    for key in keys:
                        opts=s.setdefault("ffns",{}).setdefault(key,{})
                        opts["hidden"]=max(64,round(opts.get("hidden",frame["d_ff"])*ratio/64)*64)
                offer(f"width{width}-prop{int(proportional)}",f,s)
    for factor in (.625,.75,.875,1.125,1.25):
        f,s=copy.deepcopy(frame),copy.deepcopy(parent)
        for key in keys:
            opts=s.setdefault("ffns",{}).setdefault(key,{})
            opts["hidden"]=max(64,round(opts.get("hidden",f["d_ff"])*factor/64)*64)
        offer(f"ffn-all-x{factor}",f,s)
    for key in keys:
        for act in ("relu2","silu","gelu","gelu_tanh","relu"):
            f,s=copy.deepcopy(frame),copy.deepcopy(parent)
            s.setdefault("ffns",{}).setdefault(key,{})["act"]=act
            offer(f"act-{key}-{act}",f,s)
        for factor in (.5,.75,1.5,2.):
            f,s=copy.deepcopy(frame),copy.deepcopy(parent)
            opts=s.setdefault("ffns",{}).setdefault(key,{})
            opts["hidden"]=max(64,round(opts.get("hidden",f["d_ff"])*factor/64)*64)
            offer(f"ffn-{key}-x{factor}",f,s)
        f,s=copy.deepcopy(frame),copy.deepcopy(parent)
        opts=s.setdefault("ffns",{}).setdefault(key,{})
        opts["transpose_tie"]=not opts.get("transpose_tie",False)
        offer(f"transpose-{key}",f,s)
    for i,layer in enumerate(parent["layers"]):
        f,s=copy.deepcopy(frame),copy.deepcopy(parent)
        s["layers"][i]["order"]="ffn_attn" if layer.get("order","attn_ffn")=="attn_ffn" else "attn_ffn"
        offer(f"order-{i}",f,s)
        for kind in ("attn","ffn"):
            if layer.get(kind) is None:
                continue
            if len(parent["layers"])>1:
                f,s=copy.deepcopy(frame),copy.deepcopy(parent)
                s["layers"][i][kind]=None
                s["layers"][i].pop(kind+"_scale",None)
                s["layers"]=[entry for entry in s["layers"] if entry.get("attn") is not None or entry.get("ffn") is not None]
                offer(f"delete-{kind}-{i}",f,s)
            for scale in (.5,.75,1.25,{"init":.5,"learned":True},{"init":1.,"learned":True}):
                f,s=copy.deepcopy(frame),copy.deepcopy(parent)
                s["layers"][i][kind+"_scale"]=scale
                offer(f"scale-{kind}-{i}-{scale}",f,s)
            # One distant partner at a time; explicitly preserve architectural variety.
            sources=[j for j in range(len(parent["layers"])) if abs(j-i)>1 and parent["layers"][j].get(kind) is not None]
            rng.shuffle(sources)
            for j in sources[:3]:
                f,s=copy.deepcopy(frame),copy.deepcopy(parent)
                s["layers"][i][kind]=s["layers"][j][kind]
                offer(f"share-{kind}-{j}-{i}",f,s)
            # A repeated application shares weights; a new ID adds independent capacity.
            if frame["n_layers"]<32:
                for shared in (True,False):
                    f,s=copy.deepcopy(frame),copy.deepcopy(parent)
                    new={kind:layer[kind],("ffn" if kind=="attn" else "attn"):None}
                    if not shared:
                        new[kind]=f"{kind}_extra_{generation}_{i}"
                        if kind=="ffn":
                            s.setdefault("ffns",{})[new[kind]]=copy.deepcopy(s.get("ffns",{}).get(layer[kind],{}))
                    s["layers"].insert(i,new)
                    offer(f"repeat-{kind}-{i}-shared{int(shared)}",f,s)
        if len(parent["layers"])>1:
            f,s=copy.deepcopy(frame),copy.deepcopy(parent)
            del s["layers"][i]
            offer(f"drop-entry-{i}",f,s)
        if i+1<len(parent["layers"]):
            for inject in (False,True):
                f,s=copy.deepcopy(frame),copy.deepcopy(parent)
                s["layers"][i]["loop"]=s["layers"][i].get("loop",1)+s["layers"][i+1].get("loop",1)
                s["layers"][i]["inject"]=inject
                del s["layers"][i+1]
                offer(f"loop-{i}-inject{int(inject)}",f,s)
    # Widely distributed operators; caps are applied after validation and dedup.
    rng.shuffle(rows)
    groups={}
    for row in rows:
        prefix=row[0].split("-",1)[0]
        group="width" if prefix.startswith("width") else prefix
        groups.setdefault(group,[]).append(row)
    balanced=[]
    order=["ffn","width","share","delete","drop","loop","transpose","repeat","act","scale","order"]
    rng.shuffle(order)
    while any(groups.values()):
        for group in order:
            if groups.get(group):balanced.append(groups[group].pop())
    return list(reversed(balanced))


class ResumedProcess:
    """Track an orphaned trainer after scheduler restart without double-launching."""
    def __init__(self,pid,run_path):
        self.pid,self.run_path=pid,str(run_path)
    def poll(self):
        try:
            command=Path(f"/proc/{self.pid}/cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
            stat=Path(f"/proc/{self.pid}/stat").read_text().split(") ",1)[1].split()[0]
            if ("cuda_train.py" in command or "nas.cuda_train" in command) and self.run_path in command and stat!="Z":
                return None
        except OSError:
            pass
        return 0 if (Path(self.run_path)/"summary.json").is_file() else 1
    def wait(self,timeout=5):
        end=time.monotonic()+timeout
        while self.poll() is None and time.monotonic()<end:
            time.sleep(.1)
        return self.poll()


class Scheduler(LegacyScheduler):
    def __init__(self,args):
        self.args,self.deadline=args,args.deadline
        self.cutoff=self.search_cutoff=min(args.training_cutoff,args.deadline)
        self.running,self.records,self.completed,self.architectures,self.specs,self.frames,self.job_keys={},{},{},{},{},{},{}
        self.stop_requested,self.stage,self.locks=False,"initializing",[]
        self.consecutive_failures=0
        self.rejected=set()
        self.source_hashes={name:file_hash(source_path(name, ROOT)) for name in (*PROVENANCE_SUPPORT,"cuda_minparams_search.py","cuda_search.py","cuda_model.py","cuda_train.py","data.py","data/train.json","data/tokenizer.json") if (source_path(name, ROOT)).is_file()}
        self.source_hash=hashlib.sha256(json.dumps(self.source_hashes,sort_keys=True).encode()).hexdigest()
        self.host={"hostname":socket.gethostname(),"public_ip":args.host_public_ip}
        self.gpu_metadata=self.discover_gpus()
        provenance=args.out/"provenance.json"
        if provenance.exists():
            prior=json.loads(provenance.read_text())
            if prior["source_sha256"]!=self.source_hashes:
                raise ValueError("source/data changed since scheduler started; use a new search root")
            if prior["host"]["hostname"]!=self.host["hostname"]:
                raise ValueError("resume must run on the original host")
            self.restore()
        else:
            atomic_json(provenance,{"created_utc":iso(utc_now()),"host":self.host,"gpus":self.gpu_metadata,"source_sha256":self.source_hashes,"combined_source_sha256":self.source_hash})
        self.streams={name:(args.out/name).open("a",buffering=1) for name in ("events.jsonl","architecture_registry.jsonl","run_registry.jsonl")}
        # A terminated scheduler may have left an incomplete JSON line. restore
        # repairs that tail before these append-only streams are opened.
        self.save()

    def restore(self):
        def rows(path):
            if not path.exists():
                return []
            raw=path.read_bytes(); lines=raw.splitlines(keepends=True); result=[];offset=0
            for i,line in enumerate(lines):
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    if i!=len(lines)-1:
                        raise
                    with path.open("r+b") as stream:
                        stream.truncate(offset)
                    break
                offset+=len(line)
            if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
                with path.open("ab") as stream:stream.write(b"\n")
            return result
        for record in rows(self.args.out/"architecture_registry.jsonl"):
            digest=record["architecture_id"]
            if record.get("status")=="rejected":
                self.rejected.add(digest);continue
            self.architectures[digest]=record
            self.frames[digest]=record["frame"]
            self.specs[digest]=json.loads(Path(record["spec_path"]).read_text())
        for record in rows(self.args.out/"run_registry.jsonl"):
            self.records[record["job"]["run_id"]]=record
        for run_id,record in self.records.items():
            job=Job(**record["job"])
            self.job_keys[(job.architecture_id,job.seed,5000)]=job
            if record["status"]=="completed":
                self.completed[run_id]={"job":job,"summary":record["summary"],"path":record["path"],"metrics":record.get("metrics",{})}
            elif record["status"]=="running":
                gpu=record["physical_gpu"]["physical_index"]
                if gpu in self.running:
                    raise ValueError("multiple orphaned jobs on one GPU")
                elapsed=max(0,(utc_now()-parse_utc(record["started_utc"])).total_seconds())
                self.running[gpu]={"job":job,"process":ResumedProcess(record["pid"],record["path"]),
                    "log":open(os.devnull,"w"),"start":time.monotonic()-elapsed,"terminate_at":None}

    def record(self,job,status,**details):
        current=self.records.setdefault(job.run_id,{"job":job.to_dict(),"architecture_sha256":job.architecture_id,
            "source_sha256":self.source_hash,"source_hashes":self.source_hashes,"host":self.host,
            "spec_path":str(self.args.out/"specs"/(job.architecture_id+".json")),
            "path":str(self.args.out/"attempts"/job.run_id),"parameters":self.architectures[job.architecture_id]["parameters"],
            "frame":self.frames[job.architecture_id]})
        current.update(status=status,updated_utc=iso(utc_now()),**details)
        self.append("run_registry.jsonl",copy.deepcopy(current))
        return current

    def register(self,name,frame,spec,generation=0,parents=None,operator="breadth",baseline=False):
        try:
            digest=architecture_hash(frame,spec)
            hash_error=None
        except (ValueError,TypeError,KeyError,AttributeError) as exc:
            digest=hashlib.sha256(json.dumps({"frame":frame,"spec":spec},sort_keys=True,default=repr).encode()).hexdigest()
            hash_error=exc
        if digest in self.architectures:
            return digest,False
        if digest in self.rejected:
            return None,False
        try:
            if hash_error is not None:raise ValueError(f"malformed architecture: {hash_error}")
            validate(frame,spec,baseline=baseline)
            if not self.args.plan_only:
                from nas.cuda_model import Config,validate as model_validate,spec_parameter_count
                model_validate(spec,Config(**frame))
                if spec_parameter_count(spec,Config(**frame))!=parameter_count(frame,spec):
                    raise ValueError("scheduler/model parameter counts disagree")
        except (ValueError,TypeError,KeyError,AttributeError) as exc:
            self.rejected.add(digest)
            self.append("architecture_registry.jsonl",{"architecture_id":digest,"status":"rejected","name":name,
                "created_utc":iso(utc_now()),"frame":frame,"spec":spec,"generation":generation,
                "parents":parents or [],"operator":operator,"reason":str(exc)})
            return None,False
        clean=copy.deepcopy(spec);clean["name"]=name
        path=self.args.out/"specs"/(digest+".json")
        atomic_json(path,clean)
        record={"architecture_id":digest,"status":"accepted","canonical_sha256":digest,"file_sha256":file_hash(path),
                "name":name,"generation":generation,"parents":parents or [],"operator":operator,
                "parameters":parameter_count(frame,clean),"frame":copy.deepcopy(frame),
                "applications":frame["n_layers"],"created_utc":iso(utc_now()),"spec_path":str(path)}
        self.architectures[digest],self.specs[digest],self.frames[digest]=record,clean,copy.deepcopy(frame)
        self.append("architecture_registry.jsonl",record)
        return digest,True

    def make_job(self,architecture_id,seed,phase):
        if architecture_id is None:
            return None
        key=(architecture_id,seed,5000)
        previous=self.job_keys.get(key)
        if previous:
            record=self.records[previous.run_id]
            if record["status"]=="not_admitted":
                self.record(previous,"queued",failure_reason=None)
                return previous
            if record["status"]=="queued":
                return previous
            if record["status"] in ("completed","running"):
                return None
            attempts=sum(r["job"]["architecture_id"]==architecture_id and r["job"]["seed"]==seed for r in self.records.values())
            if attempts>=self.args.max_attempts:
                return None
        job=Job(f"r{len(self.records)+1:05d}-{architecture_id[:10]}-s{seed}",architecture_id,
                self.architectures[architecture_id]["name"],seed,phase)
        self.job_keys[key]=job
        self.record(job,"queued",queued_utc=iso(utc_now()))
        return job

    def command(self,job):
        frame=self.frames[job.architecture_id]
        command=[*python_command("cuda_train.py", ROOT),"--out",str(self.args.out/"attempts"/job.run_id),
            "--spec",str(self.args.out/"specs"/(job.architecture_id+".json")),
            "--layers",str(frame["n_layers"]),"--d-model",str(frame["d_model"]),"--heads",str(frame["n_heads"]),
            "--d-ff",str(frame["d_ff"]),"--dropout","0.1","--steps","5000","--seed",str(job.seed),
            "--batch-rows","8","--lr","0.001","--warmup","100","--min-lr-frac","0.1","--weight-decay","0.1",
            "--clip","1.0","--eval-every","250","--checkpoint-every","0","--max-params",
            str(BASELINE_PARAMETERS if job.spec=="baseline" else BASELINE_PARAMETERS-1),
            "--precision","bf16","--device","cuda","--cpu-threads",str(self.args.cpu_threads),"--projection-chunk-size","0"]
        if self.args.compile:
            command.extend(["--compile","--compile-mode","default"])
        return command

    def compute_cost(self,frame,spec):
        d=frame["d_model"]
        # Relative FLOP proxy includes dense projections, causal attention,
        # all layer repeats, and the tied vocabulary projection.
        cost=frame["vocab_size"]*d
        for layer in spec["layers"]:
            branch=0
            if layer.get("attn") is not None:branch+=4*d*d+2*frame["max_len"]*d
            if layer.get("ffn") is not None:
                branch+=2*d*spec.get("ffns",{}).get(layer["ffn"],{}).get("hidden",frame["d_ff"])
            cost+=layer.get("loop",1)*branch
        return cost

    def estimate_job(self,job):
        target=self.compute_cost(self.frames[job.architecture_id],self.specs[job.architecture_id])
        ratios=[]
        for item in self.completed.values():
            record=self.records[item["job"].run_id]
            cost=self.compute_cost(self.frames[item["job"].architecture_id],self.specs[item["job"].architecture_id])
            ratios.append(record["wall_seconds"]/cost)
        estimate=target*statistics.median(ratios) if ratios else self.args.estimated_5000_seconds*max(1,target/self.compute_cost(FRAME,baseline_spec()))
        # A new graph may require compilation even when a relative-FLOP proxy is cheap.
        return min(self.args.job_timeout_seconds,max(self.args.estimated_5000_seconds,estimate)+60)

    def ingest_inbox(self):
        jobs=[]
        for path in sorted((self.args.out/"inbox").glob("*.json")):
            digest=None
            try:
                proposal=json.loads(path.read_text())
                frame=proposal["frame"];spec=proposal["spec"]
                name=proposal.get("name",path.stem)
                parents=proposal.get("parents",[proposal["parent"]] if proposal.get("parent") else [])
                generation=proposal.get("generation",0)
                digest,created=self.register(name,frame,spec,generation,parents,proposal.get("operator","human-reviewed-inbox"))
                if digest:
                    job=self.make_job(digest,0,"inbox")
                    if job is not None:jobs.append(job)
                self.emit("inbox_processed",file=path.name,architecture_id=digest,status="accepted" if digest else "rejected")
            except (OSError,ValueError,KeyError,TypeError) as exc:
                # Producers must write a temporary file then atomically rename to .json.
                self.emit("inbox_rejected",file=path.name,error=str(exc))
            destination=self.args.out/"inbox-processed"/path.name
            if destination.exists():destination=destination.with_name(destination.stem+"-"+str(time.time_ns())+".json")
            path.replace(destination)
        return jobs

    def run_stage(self,jobs,phase,finish_by=None):
        self.stage=phase
        queue=[job for job in jobs if job is not None and self.records[job.run_id]["status"]=="queued"]
        # Preserve only one pending copy of a resumed job.
        queue=list({job.run_id:job for job in queue}.values())
        finish_by=min(finish_by or self.cutoff,self.cutoff)
        self.emit("phase_started",phase=phase,jobs=len(queue),finish_by_utc=iso(finish_by))
        heartbeat=time.monotonic();last_inbox=0
        while queue or self.running:
            self.collect()
            if self.stop_requested or utc_now()>=self.cutoff or self.consecutive_failures>=self.args.max_consecutive_failures:
                reason="signal" if self.stop_requested else "training cutoff" if utc_now()>=self.cutoff else "consecutive failure limit"
                self.stop_children(reason)
                if reason=="consecutive failure limit":self.stop_requested=True
                break
            if phase not in ("bootstrap","baseline-retry") and time.monotonic()-last_inbox>=10:
                incoming=self.ingest_inbox()
                # One proposed job is prioritized per GPU wave; bulk inbox inputs
                # cannot starve existing confirmations or adaptive proposals.
                if incoming:
                    queue=list({job.run_id:job for job in incoming[:1]+queue+incoming[1:]}.values())
                last_inbox=time.monotonic()
            for gpu in self.args.gpus:
                if gpu in self.running or not queue:continue
                job=queue.pop(0)
                estimate=self.estimate_job(job)*self.args.runtime_margin
                if utc_now()+timedelta(seconds=estimate)>finish_by:
                    self.record(job,"not_admitted",failure_reason="estimated full 5000-step run exceeds phase allowance",estimated_seconds=estimate)
                    continue
                self.launch(gpu,job)
            if time.monotonic()-heartbeat>=30:
                self.emit("heartbeat",phase=phase,queued=len(queue),running=len(self.running),seconds_to_training_cutoff=max(0,(self.cutoff-utc_now()).total_seconds()))
                heartbeat=time.monotonic()
            if queue or self.running:time.sleep(self.args.poll_seconds)
        for job in queue:self.record(job,"not_admitted",failure_reason="scheduler stopped")
        self.emit("phase_finished",phase=phase)
        self.write_leaderboard();self.save()

    def collect(self):
        for gpu,item in list(self.running.items()):
            process,job=item["process"],item["job"]
            elapsed=time.monotonic()-item["start"]
            code=process.poll()
            if code is None:
                if elapsed>self.args.job_timeout_seconds and item["terminate_at"] is None:
                    self.terminate_one(item,"job timeout")
                if item["terminate_at"] is not None and time.monotonic()-item["terminate_at"]>10:
                    try: os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                continue
            item["log"].close()
            folder=self.args.out/"attempts"/job.run_id
            details={"returncode":code,"ended_utc":iso(utc_now()),"wall_seconds":elapsed,
                     "checkpoints":self.checkpoint_metadata(folder) if folder.exists() else {}}
            status="failed"; reason=self.records[job.run_id].get("stop_reason")
            if reason:
                status="timed_out" if reason=="job timeout" else "interrupted"
                details["failure_reason"]=reason
            elif code==0:
                try:
                    summary=json.loads((folder/"summary.json").read_text())
                    config=json.loads((folder/"config.json").read_text())
                    if (summary["steps"]!=5000 or summary["schedule_steps"]!=5000 or summary.get("benchmark")
                        or not math.isfinite(summary["best_dev_loss"])
                        or summary["parameters"]!=self.architectures[job.architecture_id]["parameters"]
                        or config["model"]!=self.frames[job.architecture_id]
                        or architecture_hash(config["model"],config["spec"])!=job.architecture_id
                        or config["args"]["seed"]!=job.seed or config["args"]["precision"]!="bf16"
                        or any(config["args"].get(key)!=value for key,value in RECIPE.items())):
                        raise ValueError("result differs from declared architecture/fixed recipe")
                    for relative,digest in config.get("sha256",{}).items():
                        if relative in self.source_hashes and digest!=self.source_hashes[relative]:
                            raise ValueError(f"source/data changed during run: {relative}")
                    details.update(summary=summary,metrics=self.read_metrics(folder,summary))
                    status="completed"
                    self.completed[job.run_id]={"job":job,"summary":summary,"path":str(folder),"metrics":details["metrics"]}
                except (OSError,ValueError,KeyError,TypeError) as exc:
                    details["failure_reason"]=str(exc)
            else:
                details["failure_reason"]=f"trainer exited with status {code}"
            record=self.record(job,status,**details)
            self.consecutive_failures=0 if status=="completed" else self.consecutive_failures+1
            self.emit("finished",run_id=job.run_id,architecture=job.spec,seed=job.seed,status=status,
                      parameters=record["parameters"],best_dev_loss=record.get("summary",{}).get("best_dev_loss"),
                      wall_seconds=elapsed,failure_reason=record.get("failure_reason"))
            del self.running[gpu]
            if status=="completed" and not self.args.no_plots:
                try:
                    from nas.cuda_report import render_run
                    render_run(folder)
                except Exception as exc:
                    self.emit("plot_error",run_id=job.run_id,error=str(exc))
            self.update_retention();self.prune_if_needed();self.save()

    def rankings(self,min_seeds=1,**ignored):
        grouped={}
        for run_id,item in self.completed.items():
            job=item["job"]
            if job.seed not in SEARCH_SEEDS: continue
            grouped.setdefault(job.architecture_id,{})[job.seed]=(item["summary"]["best_dev_loss"],item["path"])
        baseline_id=architecture_hash(FRAME,baseline_spec())
        baseline=grouped.get(baseline_id,{})
        rows=[]
        for digest,losses in grouped.items():
            if len(losses)<min_seeds: continue
            values=[losses[seed][0] for seed in sorted(losses)]
            paired=[losses[seed][0]-baseline[seed][0] for seed in sorted(losses) if seed in baseline]
            entry=self.architectures[digest]
            rows.append({"architecture_id":digest,"name":entry["name"],"parameters":entry["parameters"],
                "mean_dev_loss":statistics.mean(values),"std_dev_loss":statistics.stdev(values) if len(values)>1 else None,
                "seeds":sorted(losses),"dev_losses":{str(seed):loss[0] for seed,loss in losses.items()},
                "run_paths":{str(seed):loss[1] for seed,loss in losses.items()},
                "mean_paired_delta":statistics.mean(paired) if len(paired)==len(losses) else None,
                "qualified":digest!=baseline_id and set(losses)==set(SEARCH_SEEDS) and len(paired)==3 and statistics.mean(paired)<=0,
                "frame":self.frames[digest],"spec_path":entry["spec_path"]})
        return sorted(rows,key=lambda row:(row["parameters"],row["mean_dev_loss"]))

    def frontier(self,rows):
        best=math.inf;front=[]
        for row in sorted(rows,key=lambda item:(item["parameters"],item["mean_dev_loss"])):
            if row["mean_dev_loss"]<best:
                best=row["mean_dev_loss"];front.append(row)
        return front

    def write_leaderboard(self):
        rows=self.rankings();confirmed=[r for r in rows if len(r["seeds"])==3]
        atomic_json(self.args.out/"rankings.json",{"updated_utc":iso(utc_now()),"validation_used":False,
            "objective":PROTOCOL["objective"],"all":rows,"confirmed":confirmed,
            "qualified_candidates":[r for r in confirmed if r["qualified"]],"pareto_frontier":self.frontier(rows)})

    def retained_run_ids(self):
        rows=self.rankings()
        keep_ids={r["architecture_id"] for r in rows if r["name"]=="baseline" or r["qualified"]}
        keep_ids.update(r["architecture_id"] for r in self.frontier(rows))
        # Size-band leaders preserve recovery candidates before all seeds finish.
        for low in range(0,28_000_000,2_000_000):
            band=[r for r in rows if low<=r["parameters"]<low+2_000_000]
            keep_ids.update(r["architecture_id"] for r in sorted(band,key=lambda r:r["mean_dev_loss"])[:2])
        selection=self.args.out/"selection.json"
        if selection.exists():
            frozen=json.loads(selection.read_text())
            if frozen.get("primary_architecture_id"):keep_ids.add(frozen["primary_architecture_id"])
        return {run_id for run_id,item in self.completed.items() if item["job"].architecture_id in keep_ids}

    def confirm_shortlist(self,count,finish_by):
        rows=[r for r in self.rankings() if r["name"]!="baseline" and len(r["seeds"])<3]
        rows=[r for r in rows if r["mean_paired_delta"] is not None and r["mean_paired_delta"]<=self.args.confirm_margin]
        chosen=[]
        # Small feasible candidates first, plus genuinely different size bands.
        feasible=sorted([r for r in rows if r["mean_paired_delta"]<=0],key=lambda r:(r["parameters"],r["mean_dev_loss"]))
        chosen.extend(feasible[:max(1,count//2)])
        for low in range(0,28_000_000,2_000_000):
            band=sorted([r for r in rows if low<=r["parameters"]<low+2_000_000],key=lambda r:r["mean_paired_delta"])
            if band and band[0] not in chosen:chosen.append(band[0])
        for row in sorted(rows,key=lambda r:(max(0,r["mean_paired_delta"]),r["parameters"])):
            if row not in chosen:chosen.append(row)
        chosen=chosen[:count]
        jobs=[self.make_job(row["architecture_id"],seed,"confirmation") for row in chosen for seed in (1,2)]
        self.run_stage(jobs,"confirmation",finish_by)

    def adaptive_jobs(self,generation):
        rows=[r for r in self.rankings() if r["name"]!="baseline"]
        if not rows:return []
        # Preserve frontiers on paired seed0 rather than averaging unmatched seeds.
        baseline=next(r for r in self.rankings() if r["name"]=="baseline")
        reference=baseline["dev_losses"]["0"]
        rows=[{**r,"screen_loss":r["dev_losses"]["0"]} for r in rows if "0" in r["dev_losses"]]
        qualified=[r for r in rows if r["qualified"]]
        parents=qualified[:max(1,self.args.beam_width//4)]
        # Best loss within each size band gives a tiny architecture room to improve.
        buckets=[]
        for low in range(6_000_000,28_000_000,2_000_000):
            band=[r for r in rows if low<=r["parameters"]<low+2_000_000]
            if band:buckets.append(min(band,key=lambda r:r["screen_loss"]))
        # Alternate threshold-proximate and ascending-size ordering across rounds.
        buckets.sort(key=lambda r:(max(0,r["screen_loss"]-reference),r["parameters"]) if generation%2 else (r["parameters"],r["screen_loss"]))
        for row in buckets:
            if row not in parents:parents.append(row)
        for row in sorted(rows,key=lambda r:r["screen_loss"]):
            if row not in parents:parents.append(row)
        parents=parents[:self.args.beam_width]
        pools=[(row,mutations(self.frames[row["architecture_id"]],self.specs[row["architecture_id"]],generation)) for row in parents]
        jobs=[]
        while pools and len(jobs)<self.args.round_candidates and len(self.architectures)<self.args.max_architectures:
            next_pools=[]
            for parent,pool in pools:
                while pool:
                    operator,frame,spec=pool.pop()
                    name=f"g{generation}-{parent['architecture_id'][:7]}-{operator}"
                    digest,created=self.register(name,frame,spec,generation,[parent["architecture_id"]],operator)
                    if created:
                        jobs.append(self.make_job(digest,0,f"adaptive-{generation}"));break
                if pool:next_pools.append((parent,pool))
                if len(jobs)>=self.args.round_candidates or len(self.architectures)>=self.args.max_architectures:break
            pools=next_pools
        self.emit("adaptive_generation",generation=generation,jobs=len(jobs),parents=[r["architecture_id"] for r in parents])
        return jobs

    def freeze_selection(self):
        path=self.args.out/"selection.json"
        if path.exists():
            self.emit("selection_already_frozen",path=str(path));return json.loads(path.read_text())
        rows=self.rankings(min_seeds=3)
        baseline=next((r for r in rows if r["name"]=="baseline"),None)
        qualified=[r for r in rows if r["qualified"]]
        def enriched(row):
            return {**row,"spec":self.specs[row["architecture_id"]]} if row else None
        primary=enriched(qualified[0]) if qualified else None
        selection={"status":"qualified" if primary else "no_qualified_candidate","frozen_utc":iso(utc_now()),
            "validation_used":False,"selection_seeds":list(SEARCH_SEEDS),"fresh_seeds":list(FRESH_SEEDS),
            "baseline":enriched(baseline),"primary":primary,"primary_architecture_id":primary["architecture_id"] if primary else None,
            "finalists":[primary] if primary else [],"qualified_candidates":qualified,"confirmed_candidates":rows,
            "rule":PROTOCOL["selection"],"baseline_parameters":BASELINE_PARAMETERS,
            "search_source_sha256":self.source_hashes,"architecture_count":len(self.architectures),"completed_runs":len(self.completed)}
        atomic_json(path,selection)
        self.emit("selection_frozen",status=selection["status"],primary_architecture_id=selection["primary_architecture_id"],
                  parameters=primary["parameters"] if primary else None,mean_dev_loss=primary["mean_dev_loss"] if primary else None)
        return selection

    def close(self):
        self.stop_children("scheduler exit")
        for record in list(self.records.values()):
            if record["status"]=="queued":self.record(Job(**record["job"]),"not_admitted",failure_reason="scheduler exited before launch")
        self.update_retention();self.write_leaderboard();self.save()
        atomic_json(self.args.out/"terminal_status.json",{"status":"complete" if self.stage=="frozen" else self.stage,
                    "stage":self.stage,"updated_utc":iso(utc_now()),"pid":os.getpid(),"selection_path":str(self.args.out/"selection.json")})
        for stream in self.streams.values():stream.close()
        for lock in self.locks:lock.close()


def make_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root","--out",dest="out",type=Path,required=True)
    p.add_argument("--gpus",default="0,1,2,3")
    p.add_argument("--deadline","--deadline-utc",type=parse_utc,required=True)
    p.add_argument("--training-cutoff","--training-cutoff-utc",type=parse_utc,required=True)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--compile",action="store_true")
    p.add_argument("--plan-only",action="store_true")
    p.add_argument("--no-plots",action="store_true")
    p.add_argument("--host-public-ip")
    p.add_argument("--runs-root",type=Path)
    p.add_argument("--estimated-5000-seconds",type=float,default=180)
    p.add_argument("--runtime-margin",type=float,default=1.5)
    p.add_argument("--job-timeout-seconds",type=float,default=900)
    p.add_argument("--confirmation-reserve-seconds",type=float,default=900)
    p.add_argument("--max-consecutive-failures",type=int,default=6)
    p.add_argument("--max-attempts",type=int,default=2)
    p.add_argument("--max-architectures",type=int,default=3000)
    p.add_argument("--adaptive-rounds",type=int,default=60)
    p.add_argument("--round-candidates",type=int,default=48)
    p.add_argument("--beam-width",type=int,default=12)
    p.add_argument("--shortlist",type=int,default=12)
    p.add_argument("--confirm-margin",type=float,default=.06)
    p.add_argument("--max-disk-gib",type=float,default=400)
    p.add_argument("--cpu-threads",type=int,default=4)
    p.add_argument("--poll-seconds",type=float,default=1)
    return p


def main():
    args=make_parser().parse_args()
    args.gpus=[int(value) for value in args.gpus.split(",")]
    if not args.gpus or min(args.gpus)<0 or len(set(args.gpus))!=len(args.gpus):raise SystemExit("invalid GPUs")
    if min(args.estimated_5000_seconds,args.runtime_margin,args.job_timeout_seconds,args.max_consecutive_failures,args.max_attempts,
           args.max_architectures,args.adaptive_rounds,args.round_candidates,args.beam_width,args.shortlist,args.max_disk_gib,args.cpu_threads,args.poll_seconds)<=0:
        raise SystemExit("limits must be positive")
    if args.training_cutoff>args.deadline or (not args.plan_only and args.training_cutoff<=utc_now()):raise SystemExit("invalid time window")
    args.out=args.out.expanduser().resolve()
    args.runs_root=(args.runs_root or args.out.parent).expanduser().resolve()
    exists=args.out.exists()
    if exists and not args.resume:raise SystemExit("search root exists; use --resume")
    args.out.mkdir(parents=True,exist_ok=True)
    for folder in ("specs","attempts","logs","inbox","inbox-processed"):(args.out/folder).mkdir(exist_ok=True)
    root_lock=(args.out/"scheduler.lock").open("a")
    try:fcntl.flock(root_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit("scheduler already active for this root")
    scheduler=Scheduler(args)
    try:
        if (args.out/"selection.json").exists():
            scheduler.stage="frozen";return 0
        if not args.plan_only:scheduler.acquire_gpus()
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda signum,frame:setattr(scheduler,"stop_requested",True))
        baseline_id,_=scheduler.register("baseline",FRAME,baseline_spec(),baseline=True)
        if baseline_id is None:raise RuntimeError("baseline rejected")
        initial=[]
        for name,frame,spec in breadth_candidates():
            digest,_=scheduler.register(name,frame,spec)
            if digest:initial.append(digest)
        atomic_json(args.out/"plan.json",{"created_utc":iso(utc_now()),"protocol":PROTOCOL,
            "deadline_utc":iso(args.deadline),"training_cutoff_utc":iso(args.training_cutoff),
            "initial_architectures":initial,"args":{k:iso(v) if isinstance(v,datetime) else str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}})
        if args.plan_only:
            scheduler.stage="plan-only";print(json.dumps({"architectures":len(scheduler.architectures),"rejected":len(scheduler.rejected)}));return 0
        # Bootstrap 3 paired baselines and one conservative reduced candidate.
        bootstrap=[scheduler.make_job(baseline_id,seed,"baseline") for seed in SEARCH_SEEDS]
        if initial:bootstrap.append(scheduler.make_job(initial[0],0,"breadth"))
        scheduler.run_stage(bootstrap,"bootstrap",scheduler.cutoff)
        baseline=next((r for r in scheduler.rankings(min_seeds=3) if r["name"]=="baseline"),None)
        if not baseline:
            # One bounded retry per failed baseline is logged as a new attempt.
            scheduler.run_stage([scheduler.make_job(baseline_id,seed,"baseline-retry") for seed in SEARCH_SEEDS],"baseline-retry",scheduler.cutoff)
            baseline=next((r for r in scheduler.rankings(min_seeds=3) if r["name"]=="baseline"),None)
        if not baseline:raise RuntimeError("all three fresh baseline seeds must finish before architecture selection")
        scheduler.emit("baseline_reference_fixed",mean_dev_loss=baseline["mean_dev_loss"],dev_losses=baseline["dev_losses"],parameters=BASELINE_PARAMETERS)
        exploration_cutoff=scheduler.cutoff-timedelta(seconds=args.confirmation_reserve_seconds)
        # Interleave confirmation with breadth to establish a feasible size frontier early.
        for start in range(0,len(initial),args.round_candidates):
            if scheduler.stop_requested or utc_now()>=exploration_cutoff:break
            jobs=[scheduler.make_job(digest,0,"breadth") for digest in initial[start:start+args.round_candidates]]
            scheduler.run_stage(jobs,"breadth",exploration_cutoff)
            scheduler.confirm_shortlist(args.shortlist,exploration_cutoff)
        # Resume pending adaptive attempts before proposing further mutations.
        pending=[Job(**record["job"]) for record in scheduler.records.values() if record["status"] in ("queued","not_admitted") and record["job"]["phase"].startswith("adaptive")]
        if pending and utc_now()<exploration_cutoff:
            scheduler.run_stage([scheduler.make_job(job.architecture_id,job.seed,job.phase) for job in pending],"resume-adaptive",exploration_cutoff)
        first_generation=1+max((entry.get("generation",0) for entry in scheduler.architectures.values()),default=0)
        for generation in range(first_generation,args.adaptive_rounds+1):
            if scheduler.stop_requested or utc_now()>=exploration_cutoff or len(scheduler.architectures)>=args.max_architectures:break
            jobs=scheduler.adaptive_jobs(generation)
            if not jobs:break
            scheduler.run_stage(jobs,f"adaptive-{generation}",exploration_cutoff)
            scheduler.confirm_shortlist(args.shortlist,exploration_cutoff)
        if scheduler.stop_requested:
            scheduler.stage="interrupted";return 130
        scheduler.confirm_shortlist(args.shortlist*2,scheduler.cutoff)
        scheduler.freeze_selection();scheduler.stage="frozen"
        return 0
    except BaseException as exc:
        scheduler.stage="error";scheduler.emit("scheduler_error",error=repr(exc));raise
    finally:
        scheduler.close();root_lock.close()


if __name__=="__main__":
    raise SystemExit(main())
