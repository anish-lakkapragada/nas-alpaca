"""Alpaca chat encoding, the fixed train/dev split and packed batches.

Format: BOS <|user|> prompt EOS <|assistant|> answer EOS. Only answer tokens and
the answer's EOS are supervised. `validation.json` is never touched here except
through `load_split("validation")`, which only the final evaluation calls.
"""
from __future__ import annotations

from nas.paths import PROJECT_ROOT

import json
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

DATA = PROJECT_ROOT / "data"
SEQ_LEN = 1024          # longest training conversation is 1,021 tokens
DEV_SIZE = 1000         # carved from train.json for all selection decisions
SPLIT_SEED = 1234
PAD, BOS, EOS, USER, ASSISTANT = 0, 1, 2, 4, 5


def tokenizer() -> Tokenizer:
    tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    tok.encode_special_tokens = True   # literal "<|eos|>" in text stays text
    return tok


def encode(tok: Tokenizer, messages, generation_prompt: bool = False):
    """Return (ids, supervised) where supervised[i] marks ids[i] as a target."""
    ids, sup = [BOS], [False]
    for m in messages:
        body = tok.encode(m["content"], add_special_tokens=False).ids + [EOS]
        ids += [USER if m["role"] == "user" else ASSISTANT] + body
        sup += [False] + [m["role"] == "assistant"] * len(body)
    if generation_prompt:
        ids.append(ASSISTANT)
        sup.append(False)
    return ids, sup


def load_split(name: str):
    """'train' and 'dev' partition train.json; 'validation' is the held-out file."""
    if name == "validation":
        return json.loads((DATA / "validation.json").read_text())
    records = json.loads((DATA / "train.json").read_text())
    order = list(range(len(records)))
    random.Random(SPLIT_SEED).shuffle(order)
    keep = order[:DEV_SIZE] if name == "dev" else order[DEV_SIZE:]
    return [records[i] for i in sorted(keep)]


def encode_split(tok, records):
    return [encode(tok, r["messages"]) for r in records]


def pack(examples, rng: random.Random | None, seq_len: int = SEQ_LEN):
    """Greedily pack whole conversations into rows (shuffled when rng is given).

    Returns int32 arrays inputs, targets, segments, positions and a float32
    weight array, each [rows, seq_len]. Segment ids give a block-diagonal causal
    mask, so packed conversations cannot attend to each other; positions restart
    at zero in every conversation. Padding has segment -1 and weight 0.
    """
    order = list(range(len(examples)))
    if rng is not None:
        rng.shuffle(order)
    rows, row, used = [], [], 0
    for i in order:
        n = len(examples[i][0])
        if used + n > seq_len + 1 and row:
            rows.append(row)
            row, used = [], 0
        row.append(i)
        used += n
    if row:
        rows.append(row)
    R = len(rows)
    inp = np.full((R, seq_len), PAD, np.int32)
    tgt = np.zeros((R, seq_len), np.int32)
    seg = np.full((R, seq_len), -1, np.int32)
    pos = np.zeros((R, seq_len), np.int32)
    wt = np.zeros((R, seq_len), np.float32)
    for r, members in enumerate(rows):
        c = 0
        for s, i in enumerate(members):
            ids, sup = examples[i]
            n = len(ids) - 1              # predict ids[1:] from ids[:-1]
            n = min(n, seq_len - c)
            inp[r, c:c + n] = ids[:n]
            tgt[r, c:c + n] = ids[1:n + 1]
            wt[r, c:c + n] = sup[1:n + 1]
            seg[r, c:c + n] = s
            pos[r, c:c + n] = np.arange(n)
            c += n
    return inp, tgt, seg, pos, wt
