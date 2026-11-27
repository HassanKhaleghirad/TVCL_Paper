#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TVCL-Real: real InTAS JSON pipeline with MAR, CSSL and EAD.

Expected layout:
    project/
      tvcl_real.py
      dataset/
        veh_0001.json
        ...
        (841 JSON files)

Each JSON file may contain a list of BSM-like records. The loader also accepts
a single record or a dict containing a list under common record keys.

Important protocol choices:
- Uses only real JSON records; no synthetic samples are generated.
- Splits by sender_id (vehicle-disjoint), not by individual messages.
- Fits normalization on TRAIN only.
- Tunes the decision threshold on VALIDATION only.
- Evaluates TEST once using the selected validation threshold.
- Excludes distance_to_road_edge and other simulator-only fields.
- Keeps MAR, CSSL and EAD independently switchable for valid ablations.

Dependencies:
    pip install numpy pandas scikit-learn scipy torch

Run:
    python tvcl_real.py --data .\\dataset --epochs 100 --seeds 0 1 2 3 4
"""

import argparse
import json
import math
import os
import random
import glob
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial import cKDTree
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    balanced_accuracy_score
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


# --------------------------- Reproducibility ---------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------- JSON ingestion ----------------------------

def as_float(value, default=np.nan):
    try:
        if value is None:
            return default
        x = float(value)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def parse_vec(value, n=3):
    """Parse 'x,y,z' strings or numeric arrays."""
    if isinstance(value, str):
        try:
            vals = [float(v.strip()) for v in value.split(",")]
        except ValueError:
            vals = []
    elif isinstance(value, (list, tuple, np.ndarray)):
        vals = [as_float(v) for v in value]
    else:
        vals = []
    vals = vals[:n] + [np.nan] * max(0, n - len(vals))
    return vals


def extract_records(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # Common wrappers used by dataset exporters.
        for key in ("messages", "data", "records", "items", "bsm", "BSM"):
            if isinstance(obj.get(key), list):
                return obj[key]
        if "sender_id" in obj:
            return [obj]
    return []


def load_json_folder(folder):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.json"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .json files found under: {folder}")

    rows = []
    bad_files = []
    bad_records = 0

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8-sig") as f:
                obj = json.load(f)
            records = extract_records(obj)
            if not records:
                bad_files.append((fp, "No record list found"))
                continue

            for rec in records:
                if not isinstance(rec, dict):
                    bad_records += 1
                    continue
                sender = rec.get("sender") or {}
                receiver = rec.get("receiver") or {}
                if not isinstance(sender, dict):
                    sender = {}
                if not isinstance(receiver, dict):
                    receiver = {}

                sid = rec.get("sender_id")
                label = rec.get("attacker")
                if sid is None or label is None:
                    bad_records += 1
                    continue

                spos = parse_vec(sender.get("pos"))
                rpos = parse_vec(receiver.get("pos"))
                snoise = parse_vec(sender.get("pos_noise"))
                rnoise = parse_vec(receiver.get("pos_noise"))

                sx, sy = spos[0], spos[1]
                rx, ry = rpos[0], rpos[1]
                dx = sx - rx if np.isfinite(sx) and np.isfinite(rx) else np.nan
                dy = sy - ry if np.isfinite(sy) and np.isfinite(ry) else np.nan

                sh = as_float(sender.get("hed"))
                rh = as_float(receiver.get("hed"))
                heading_delta = np.nan
                if np.isfinite(sh) and np.isfinite(rh):
                    heading_delta = abs((sh - rh + 180.0) % 360.0 - 180.0)

                rcv = as_float(rec.get("rcvTime"))
                snd = as_float(rec.get("sendTime"))
                # In the supplied InTAS sample timestamps are in microseconds.
                delay_s = (rcv - snd) / 1e6 if np.isfinite(rcv) and np.isfinite(snd) else np.nan

                rows.append({
                    "source_file": os.path.basename(fp),
                    "sender_id": str(sid),
                    "sender_alias": str(rec.get("sender_alias", "")),
                    "message_id": str(rec.get("messageID", "")),
                    "rcv_time": rcv,
                    "send_time": snd,
                    "label": int(float(label) > 0),
                    "sx": sx, "sy": sy,
                    "spos_noise_x": snoise[0], "spos_noise_y": snoise[1],
                    "speed": as_float(sender.get("spd")),
                    "speed_noise": as_float(sender.get("spd_noise")),
                    "accel": as_float(sender.get("acl")),
                    "accel_noise": as_float(sender.get("acl_noise")),
                    "heading": sh,
                    "heading_noise": as_float(sender.get("hed_noise")),
                    "receiver_speed": as_float(receiver.get("spd")),
                    "receiver_accel": as_float(receiver.get("acl")),
                    "receiver_heading": rh,
                    "relative_dx": dx,
                    "relative_dy": dy,
                    "relative_distance": math.hypot(dx, dy) if np.isfinite(dx) and np.isfinite(dy) else np.nan,
                    "relative_speed": (
                        as_float(sender.get("spd")) - as_float(receiver.get("spd"))
                        if np.isfinite(as_float(sender.get("spd"))) and np.isfinite(as_float(receiver.get("spd")))
                        else np.nan
                    ),
                    "relative_heading": heading_delta,
                    "receiver_pos_noise_x": rnoise[0],
                    "receiver_pos_noise_y": rnoise[1],
                    "delay_s": delay_s,
                })
        except Exception as exc:
            bad_files.append((fp, repr(exc)))

    if not rows:
        raise RuntimeError("No usable labeled records were found in the JSON files.")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(
        subset=["sender_id", "rcv_time", "message_id"], keep="first"
    )
    df = df.sort_values(["sender_id", "rcv_time"], kind="stable").reset_index(drop=True)

    print(f"JSON files discovered : {len(files)}")
    print(f"Files with errors     : {len(bad_files)}")
    print(f"Invalid records       : {bad_records}")
    print(f"Usable messages       : {len(df):,}")
    print(f"Unique senders        : {df.sender_id.nunique():,}")
    print(f"Label counts          : {df.label.value_counts().sort_index().to_dict()}")
    if bad_files[:5]:
        print("First file errors:")
        for item in bad_files[:5]:
            print("  ", item)
    return df


# ----------------------- Causal temporal features ---------------------

BASE_FEATURES = [
    "sx", "sy", "spos_noise_x", "spos_noise_y",
    "speed", "speed_noise", "accel", "accel_noise",
    "heading_sin", "heading_cos", "heading_noise",
    "relative_dx", "relative_dy", "relative_distance",
    "relative_speed", "relative_heading",
    "receiver_speed", "receiver_accel",
    "receiver_pos_noise_x", "receiver_pos_noise_y",
    "delay_s", "inter_message_gap_s",
]

def add_temporal_features(df):
    df = df.copy()
    heading_rad = np.deg2rad(df["heading"].to_numpy(dtype=float))
    df["heading_sin"] = np.sin(heading_rad)
    df["heading_cos"] = np.cos(heading_rad)

    # Time between consecutive messages from the same sender.
    gap = df.groupby("sender_id", sort=False)["rcv_time"].diff() / 1e6
    df["inter_message_gap_s"] = gap.clip(lower=0, upper=60)

    # Strictly causal history: shift(1) before rolling; current label is never used.
    temporal_cols = [
        "speed", "accel", "heading_sin", "heading_cos",
        "relative_distance", "relative_speed", "spos_noise_x", "spos_noise_y"
    ]
    grouped = df.groupby("sender_id", sort=False)
    for col in temporal_cols:
        past = grouped[col].shift(1)
        short_mean = past.groupby(df["sender_id"], sort=False).rolling(
            window=3, min_periods=1
        ).mean().reset_index(level=0, drop=True)
        long_mean = past.groupby(df["sender_id"], sort=False).rolling(
            window=20, min_periods=1
        ).mean().reset_index(level=0, drop=True)
        short_std = past.groupby(df["sender_id"], sort=False).rolling(
            window=3, min_periods=2
        ).std().reset_index(level=0, drop=True)

        df[f"{col}_short"] = short_mean
        df[f"{col}_long"] = long_mean
        df[f"{col}_short_std"] = short_std

    # Counts represent history availability and are valid reliability inputs.
    df["history_count_short"] = grouped.cumcount().clip(upper=3)
    df["history_count_long"] = grouped.cumcount().clip(upper=20)

    short_cols = [f"{c}_short" for c in temporal_cols]
    long_cols = [f"{c}_long" for c in temporal_cols]
    stability_cols = [f"{c}_short_std" for c in temporal_cols]

    for c in short_cols + long_cols + stability_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    return df


def split_by_sender(df, seed):
    """Stratified 70/15/15 split at sender level; no sender crosses partitions."""
    sender_table = df.groupby("sender_id")["label"].max().reset_index()
    groups = sender_table["sender_id"].to_numpy()
    y_group = sender_table["label"].to_numpy()

    # 70% train, 30% temporary.
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(groups, y_group))
    train_groups = set(groups[train_idx])
    temp_groups = groups[temp_idx]
    temp_y = y_group[temp_idx]

    # Split remaining 30% equally into validation and test.
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1009)
    val_rel, test_rel = next(sss2.split(temp_groups, temp_y))
    val_groups = set(temp_groups[val_rel])
    test_groups = set(temp_groups[test_rel])

    split = np.full(len(df), -1, dtype=np.int64)
    split[df["sender_id"].isin(train_groups).to_numpy()] = 0
    split[df["sender_id"].isin(val_groups).to_numpy()] = 1
    split[df["sender_id"].isin(test_groups).to_numpy()] = 2
    if (split < 0).any():
        raise RuntimeError("Some rows were not assigned to a data split.")

    # Assert strict vehicle disjointness.
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
    return split


# -------------------------- Graph construction ------------------------

def build_spatiotemporal_edges(df, split_ids, k=5, time_bin_s=1.0, radius=150.0):
    """
    Build edges only within the same partition and local time bin.
    Edges use sender positions and are not based on labels.
    """
    src_all, dst_all, edge_attr_all = [], [], []
    times = df["rcv_time"].to_numpy(dtype=float) / 1e6
    pos = df[["sx", "sy"]].to_numpy(dtype=float)

    valid_pos = np.isfinite(pos).all(axis=1) & np.isfinite(times)
    bins = np.floor(times / time_bin_s).astype(np.int64)

    for part in (0, 1, 2):
        ids = np.where((split_ids == part) & valid_pos)[0]
        if len(ids) < 2:
            continue

        # Group by temporal bin to avoid connecting distant observations in time.
        local = defaultdict(list)
        for idx in ids:
            local[bins[idx]].append(idx)

        for _, members in local.items():
            if len(members) < 2:
                continue
            members = np.asarray(members, dtype=np.int64)
            coords = pos[members]
            tree = cKDTree(coords)
            kk = min(k + 1, len(members))
            distances, neighbors = tree.query(coords, k=kk)

            for row_i, global_i in enumerate(members):
                for col in range(1, kk):
                    dist = float(distances[row_i, col])
                    if not np.isfinite(dist) or dist > radius:
                        continue
                    global_j = int(members[int(neighbors[row_i, col])])
                    if global_i == global_j:
                        continue
                    src_all.append(global_i)
                    dst_all.append(global_j)
                    # Edge attributes: distance and absolute time difference.
                    edge_attr_all.append([dist, abs(times[global_i] - times[global_j])])

    if not src_all:
        # Empty edge list is valid; EAD falls back to self representation.
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, 2), dtype=np.float32),
        )
    return (
        np.asarray([src_all, dst_all], dtype=np.int64),
        np.asarray(edge_attr_all, dtype=np.float32),
    )


# ------------------------------- Model --------------------------------

class MAR(nn.Module):
    """Reliability-aware three-scale fusion: current, short history, long history."""
    def __init__(self, dim, reliability_dim=2):
        super().__init__()
        self.scale_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
            for _ in range(3)
        ])
        self.gate = nn.Sequential(
            nn.Linear(dim * 3 + reliability_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
        )
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, current, short, long, reliability):
        views = [
            self.scale_proj[0](current),
            self.scale_proj[1](short),
            self.scale_proj[2](long),
        ]
        logits = self.gate(torch.cat([views[0], views[1], views[2], reliability], dim=-1))

        # A missing history view receives a strong negative logit.
        short_ok = (reliability[:, 0:1] > 0).float()
        long_ok = (reliability[:, 1:2] > 0).float()
        logits = torch.stack(
            [logits[:, 0], logits[:, 1] + (short_ok[:, 0] - 1) * 1e4,
             logits[:, 2] + (long_ok[:, 0] - 1) * 1e4],
            dim=1,
        )
        weights = torch.softmax(logits, dim=1)
        fused = (
            weights[:, 0:1] * views[0]
            + weights[:, 1:2] * views[1]
            + weights[:, 2:3] * views[2]
        )
        return self.out(fused + views[0]), weights


class EAD(nn.Module):
    """Edge-aware attention aggregation with residual self-information."""
    def __init__(self, dim, edge_dim=2):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.edge_score = nn.Sequential(
            nn.Linear(edge_dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        self.update = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def forward(self, h, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return h

        src, dst = edge_index[0], edge_index[1]
        q = self.query(h[dst])
        k = self.key(h[src])
        v = self.value(h[src])

        edge_attr = edge_attr.clone()
        # Scale distance and time attributes robustly.
        edge_attr[:, 0] = torch.log1p(edge_attr[:, 0].clamp(min=0))
        edge_attr[:, 1] = torch.log1p(edge_attr[:, 1].clamp(min=0))
        e = self.edge_score(edge_attr).squeeze(-1)

        score = (q * k).sum(-1) / math.sqrt(h.shape[-1]) + e
        # Stable destination-wise softmax.
        max_per_dst = torch.full(
            (h.shape[0],), -torch.inf, device=h.device, dtype=h.dtype
        )
        max_per_dst.scatter_reduce_(0, dst, score, reduce="amax", include_self=True)
        exp_score = torch.exp(score - max_per_dst[dst])
        denom = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
        denom.index_add_(0, dst, exp_score)
        alpha = exp_score / (denom[dst] + 1e-12)

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, alpha.unsqueeze(-1) * v)
        return self.update(torch.cat([h, agg], dim=-1)) + h


class TVCLReal(nn.Module):
    def __init__(self, input_dim, hidden=96, use_mar=True, use_cssl=True, use_ead=True):
        super().__init__()
        self.use_mar = use_mar
        self.use_cssl = use_cssl
        self.use_ead = use_ead

        self.current_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Dropout(0.15), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.short_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.long_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )

        self.mar = MAR(hidden, reliability_dim=2)
        self.ead = EAD(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(0.20), nn.Linear(hidden // 2, 1)
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 64)
        )

    def encode(self, x_current, x_short, x_long, reliability,
               edge_index, edge_attr):
        h0 = self.current_encoder(x_current)
        hs = self.short_encoder(x_short)
        hl = self.long_encoder(x_long)

        if self.use_mar:
            h, mar_weights = self.mar(h0, hs, hl, reliability)
        else:
            h, mar_weights = h0, None

        if self.use_ead:
            h = self.ead(h, edge_index, edge_attr)

        logits = self.classifier(h).squeeze(-1)
        z = F.normalize(self.projector(h), dim=-1)
        return logits, z, h, mar_weights

    def forward(self, *args):
        return self.encode(*args)


# -------------------------- Contrastive loss --------------------------

def contrastive_loss(z1, z2, temperature=0.2, batch_size=512):
    """Memory-bounded symmetric NT-Xent with in-batch negatives.

    The previous implementation formed one (2N x 2N) similarity matrix
    for every training message. This version computes the same NT-Xent
    objective independently over chunks, limiting peak similarity-matrix
    memory to (2 * batch_size)^2 instead of (2 * N)^2.
    """
    n = z1.shape[0]
    if n < 2:
        return z1.new_tensor(0.0)

    losses = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        a = z1[start:end]
        b = z2[start:end]
        m = end - start
        if m < 2:
            continue

        z = torch.cat([a, b], dim=0)
        sim = torch.matmul(z, z.T) / temperature
        eye = torch.eye(2 * m, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, -1e9)
        targets = torch.cat([
            torch.arange(m, 2 * m, device=z.device),
            torch.arange(0, m, device=z.device),
        ])
        losses.append(F.cross_entropy(sim, targets))

    if not losses:
        return z1.new_tensor(0.0)
    return torch.stack(losses).mean()


# ------------------------------ Training ------------------------------

def make_model_inputs(df, split, scaler=None, fit_scaler=False):
    temporal = add_temporal_features(df)
    current_cols = BASE_FEATURES
    short_cols = [f"{c}_short" for c in [
        "speed", "accel", "heading_sin", "heading_cos",
        "relative_distance", "relative_speed", "spos_noise_x", "spos_noise_y"
    ]]
    long_cols = [f"{c}_long" for c in [
        "speed", "accel", "heading_sin", "heading_cos",
        "relative_distance", "relative_speed", "spos_noise_x", "spos_noise_y"
    ]]

    # Add current feature subsets to temporal views to keep a consistent dimension.
    current = temporal[current_cols].to_numpy(dtype=np.float32)
    short_raw = temporal[short_cols].to_numpy(dtype=np.float32)
    long_raw = temporal[long_cols].to_numpy(dtype=np.float32)

    # Map the temporal summaries into the same feature dimension.
    short = np.zeros_like(current)
    long = np.zeros_like(current)
    short[:, :len(short_cols)] = short_raw
    long[:, :len(long_cols)] = long_raw

    # Missing-value handling: medians are learned on train rows only.
    train_mask = split == 0
    med_current = np.nanmedian(current[train_mask], axis=0)
    med_short = np.nanmedian(short[train_mask], axis=0)
    med_long = np.nanmedian(long[train_mask], axis=0)
    med_current = np.nan_to_num(med_current, nan=0.0, posinf=0.0, neginf=0.0)
    med_short = np.nan_to_num(med_short, nan=0.0, posinf=0.0, neginf=0.0)
    med_long = np.nan_to_num(med_long, nan=0.0, posinf=0.0, neginf=0.0)

    for arr, med in ((current, med_current), (short, med_short), (long, med_long)):
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = np.take(med, np.where(bad)[1])

    # Separate train-fitted scalers prevent leakage.
    sc_current = StandardScaler()
    sc_short = StandardScaler()
    sc_long = StandardScaler()
    sc_current.fit(current[train_mask])
    sc_short.fit(short[train_mask])
    sc_long.fit(long[train_mask])

    current = sc_current.transform(current).astype(np.float32)
    short = sc_short.transform(short).astype(np.float32)
    long = sc_long.transform(long).astype(np.float32)

    # Reliability flags: 0 means no causal history exists for that scale.
    counts = temporal[["history_count_short", "history_count_long"]].to_numpy(dtype=np.float32)
    reliability = np.column_stack([
        (counts[:, 0] > 0).astype(np.float32),
        (counts[:, 1] >= 3).astype(np.float32),
    ])

    return current, short, long, reliability, temporal


def make_temporal_pairs(df, split):
    """Previous-message index for each sender, only when both rows share a split."""
    n = len(df)
    prev = np.full(n, -1, dtype=np.int64)
    for _, idx in df.groupby("sender_id", sort=False).indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) < 2:
            continue
        for a, b in zip(idx[:-1], idx[1:]):
            if split[a] == split[b]:
                prev[b] = a
    return prev


def run_one_seed(df, seed, args, device):
    set_seed(seed)
    split = split_by_sender(df, seed)
    split_names = np.asarray(["train", "validation", "test"], dtype=object)[split]
    split_manifest = df[["source_file", "sender_id", "rcv_time", "label"]].copy()
    split_manifest["split"] = split_names
    split_manifest.to_csv(
        f"tvcl_split_seed_{seed}.csv", index=False, encoding="utf-8-sig"
    )
    current, short, long, reliability, temporal = make_model_inputs(df, split)

    y = df["label"].to_numpy(dtype=np.float32)
    edge_index_np, edge_attr_np = build_spatiotemporal_edges(
        df, split, k=args.k_neighbors,
        time_bin_s=args.time_bin, radius=args.edge_radius
    )

    Xc = torch.tensor(current, dtype=torch.float32, device=device)
    Xs = torch.tensor(short, dtype=torch.float32, device=device)
    Xl = torch.tensor(long, dtype=torch.float32, device=device)
    R = torch.tensor(reliability, dtype=torch.float32, device=device)
    Y = torch.tensor(y, dtype=torch.float32, device=device)
    E = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    EA = torch.tensor(edge_attr_np, dtype=torch.float32, device=device)

    train_idx = torch.tensor(np.where(split == 0)[0], dtype=torch.long, device=device)
    val_idx = torch.tensor(np.where(split == 1)[0], dtype=torch.long, device=device)
    test_idx = torch.tensor(np.where(split == 2)[0], dtype=torch.long, device=device)

    pos = float(y[split == 0].sum())
    neg = float((split == 0).sum() - pos)
    if pos == 0 or neg == 0:
        raise RuntimeError("Training split contains only one class.")
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    ablations = [
        ("TVCL-Full", True, True, True),
        ("w/o-MAR", False, True, True),
        ("w/o-CSSL", True, False, True),
        ("w/o-EAD", True, True, False),
        ("w/o-MAR-CSSL", False, False, True),
        ("w/o-MAR-EAD", False, True, False),
        ("w/o-CSSL-EAD", True, False, False),
        ("w/o-MAR-CSSL-EAD", False, False, False),
    ]

    results = []
    for name, use_mar, use_cssl, use_ead in ablations:
        set_seed(seed)
        model = TVCLReal(
            input_dim=Xc.shape[1], hidden=args.hidden,
            use_mar=use_mar, use_cssl=use_cssl, use_ead=use_ead
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_state = None
        best_val_ap = -1.0
        best_epoch = 0
        patience_count = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()

            logits, z, _, _ = model(Xc, Xs, Xl, R, E, EA)
            cls_loss = criterion(logits[train_idx], Y[train_idx])

            ssl_loss = logits.new_tensor(0.0)
            if use_cssl and args.lambda_cssl > 0:
                # Augmentations are deliberately mild to preserve attack evidence.
                x1 = Xc.clone()
                x2 = Xc.clone()
                x1[train_idx] = x1[train_idx] * (
                    torch.rand_like(x1[train_idx]) > args.feature_drop
                )
                x2[train_idx] = x2[train_idx] * (
                    torch.rand_like(x2[train_idx]) > args.feature_drop
                )
                x1[train_idx] += torch.randn_like(x1[train_idx]) * args.aug_noise
                x2[train_idx] += torch.randn_like(x2[train_idx]) * args.aug_noise

                # Re-encode both augmented views through the complete model.
                _, z1, _, _ = model(x1, Xs, Xl, R, E, EA)
                _, z2, _, _ = model(x2, Xs, Xl, R, E, EA)
                ssl_loss = contrastive_loss(z1[train_idx], z2[train_idx])

            loss = cls_loss + args.lambda_cssl * ssl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits, _, _, _ = model(Xc, Xs, Xl, R, E, EA)
                val_prob = torch.sigmoid(val_logits[val_idx]).cpu().numpy()
                val_y = y[split == 1]
                val_ap = average_precision_score(val_y, val_prob) if len(np.unique(val_y)) > 1 else 0.0

            if val_ap > best_val_ap + 1e-5:
                best_val_ap = val_ap
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= args.patience:
                break

        if best_state is None:
            raise RuntimeError(f"No checkpoint selected for {name}")
        model.load_state_dict(best_state)
        model.to(device)
        model.eval()

        with torch.no_grad():
            val_logits, _, _, mar_w = model(Xc, Xs, Xl, R, E, EA)
            val_prob = torch.sigmoid(val_logits[val_idx]).cpu().numpy()
            val_y = y[split == 1]

            # Select threshold on validation only; optimize F1.
            thresholds = np.linspace(0.05, 0.95, 181)
            f1s = [
                f1_score(val_y, (val_prob >= th).astype(int), zero_division=0)
                for th in thresholds
            ]
            threshold = float(thresholds[int(np.argmax(f1s))])

            test_logits, _, _, _ = model(Xc, Xs, Xl, R, E, EA)
            test_prob = torch.sigmoid(test_logits[test_idx]).cpu().numpy()
            test_y = y[split == 2]
            pred = (test_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            test_y, pred, labels=[0, 1]
        ).ravel()

        row = {
            "seed": seed,
            "model": name,
            "best_epoch": best_epoch,
            "val_pr_auc": best_val_ap,
            "threshold": threshold,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "precision": precision_score(test_y, pred, zero_division=0),
            "recall": recall_score(test_y, pred, zero_division=0),
            "f1": f1_score(test_y, pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(test_y, pred),
            "auc": roc_auc_score(test_y, test_prob) if len(np.unique(test_y)) > 1 else np.nan,
            "pr_auc": average_precision_score(test_y, test_prob),
            "test_messages": len(test_y),
            "test_senders": df.loc[split == 2, "sender_id"].nunique(),
        }
        if mar_w is not None:
            weights = mar_w[train_idx].mean(0).detach().cpu().numpy()
            row["mar_current_weight"] = float(weights[0])
            row["mar_short_weight"] = float(weights[1])
            row["mar_long_weight"] = float(weights[2])
        else:
            row["mar_current_weight"] = np.nan
            row["mar_short_weight"] = np.nan
            row["mar_long_weight"] = np.nan

        results.append(row)
        print(
            f"seed={seed:02d} {name:20s} "
            f"F1={row['f1']:.4f} PR-AUC={row['pr_auc']:.4f} "
            f"AUC={row['auc']:.4f} P/R={row['precision']:.3f}/{row['recall']:.3f} "
            f"TP/FP/TN/FN={tp}/{fp}/{tn}/{fn} "
            f"best_epoch={best_epoch}"
        )

    # Verify vehicle disjointness and report split sizes.
    for part, label in ((0, "train"), (1, "validation"), (2, "test")):
        mask = split == part
        print(
            f"  {label:10s}: messages={mask.sum():,}, "
            f"senders={df.loc[mask, 'sender_id'].nunique():,}, "
            f"attacks={int(y[mask].sum()):,}"
        )
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset")
    p.add_argument("--out", default="tvcl_real_ablation.csv")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-cssl", type=float, default=0.08)
    p.add_argument("--feature-drop", type=float, default=0.08)
    p.add_argument("--aug-noise", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=2.0)
    p.add_argument("--k-neighbors", type=int, default=5)
    p.add_argument("--time-bin", type=float, default=1.0)
    p.add_argument("--edge-radius", type=float, default=150.0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print("Device:", device)
    df = load_json_folder(args.data)
    df = add_temporal_features(df)

    all_rows = []
    for seed in args.seeds:
        print(f"\n========== SEED {seed} ==========")
        rows = run_one_seed(df, seed, args, device)
        all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")

    metric_cols = [
        "f1", "precision", "recall", "balanced_accuracy", "auc", "pr_auc",
        "val_pr_auc", "mar_current_weight", "mar_short_weight", "mar_long_weight"
    ]
    summary = out_df.groupby("model")[metric_cols].agg(["mean", "std"])
    summary.to_csv(Path(args.out).with_name("tvcl_real_summary.csv"))

    print("\n========== MEAN ± STD ACROSS SEEDS ==========")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved detailed results: {args.out}")
    print("Saved summary: tvcl_real_summary.csv")


if __name__ == "__main__":
    main()
