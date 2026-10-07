#!/usr/bin/env python3
"""Standalone inference for the data-driven cloth model.

Loads the artifacts written by the training notebook (``artifacts/``) and predicts
per-vertex cloth offsets from a new 56-D pose, WITHOUT retraining and WITHOUT PyTorch
(only NumPy is required).

NOTE: the model was trained on a *synthetic* pose -> offset generator, not on real cloth
simulations. Outputs are synthetic vertex offsets in the generator's data units.

Usage
-----
    python infer_cloth.py --random 3 --out pred.npy        # 3 random unit-quaternion poses
    python infer_cloth.py --pose my_pose.npy --out pred.npy  # (56,), (N,56), (14,4) or (N,14,4)
    python infer_cloth.py --pose my_pose.csv                 # comma-separated text, 56 values per row

Python:
    from infer_cloth import ClothPredictor
    offsets = ClothPredictor("artifacts").predict(pose)      # (2170, 3) for one pose, (N, 2170, 3) for N
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

N_JOINTS, QUAT_DIM = 14, 4
IN_DIM = N_JOINTS * QUAT_DIM          # 56
FORMAT_VERSION = 1


class ClothPredictor:
    """MLP (56 -> 128 -> 256 -> k) followed by a fixed linear PCA decoder (k -> 6510)."""

    def __init__(self, artifact_dir="artifacts"):
        d = Path(artifact_dir)
        required = ["metadata.json", "preprocessing.json", "model_weights.npz"]
        missing = [f for f in required if not (d / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing artifact file(s) {missing} in '{d.resolve()}'. "
                "Run the training notebook once (the 'Save artifacts' cell) to create them."
            )
        self.meta = json.loads((d / "metadata.json").read_text())
        self.prep = json.loads((d / "preprocessing.json").read_text())
        if self.meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported artifact format_version {self.meta.get('format_version')!r}")

        w = np.load(d / "model_weights.npz")
        n_layers = int(self.meta["n_linear_layers"])
        # torch nn.Linear stores weight as (out, in): y = x @ W.T + b
        self.layers = [(w[f"layer{i}_weight"].astype(np.float32), w[f"layer{i}_bias"].astype(np.float32))
                       for i in range(n_layers)]
        self.in_dim = int(self.meta["in_dim"])
        self.out_dim = int(self.meta["out_dim"])
        if self.layers[0][0].shape[1] != self.in_dim:
            raise ValueError("Weight shapes do not match metadata (in_dim).")

        self.k = self.meta["pca_components"]          # None => the MLP predicts the 6510 offsets directly
        if self.k is not None:
            p = np.load(d / "pca.npz")
            self.components = p["components"].astype(np.float32)   # (k, out_dim)
            self.mean = p["mean"].astype(np.float32)               # (out_dim,)
            if self.components.shape != (self.k, self.out_dim) or self.layers[-1][0].shape[0] != self.k:
                raise ValueError("PCA / weight shapes do not match metadata.")
        else:
            self.components = self.mean = None
            if self.layers[-1][0].shape[0] != self.out_dim:
                raise ValueError("Output layer shape does not match metadata (out_dim).")
        self.n_verts = int(self.meta["n_vertices"])

    # ------------------------------------------------------------------ preprocessing
    def preprocess(self, pose):
        x = np.asarray(pose, dtype=np.float32)
        if x.ndim == 1 or x.shape == (N_JOINTS, QUAT_DIM):   # single pose: (56,) or (14, 4)
            x = x.reshape(1, -1)
        if x.ndim == 3:                                  # (N, 14, 4) -> (N, 56)
            x = x.reshape(len(x), -1)
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(
                f"Expected a pose of shape (56,), (N,56), (14,4) or (N,14,4); got {np.asarray(pose).shape}.")
        if not np.all(np.isfinite(x)):
            raise ValueError("Pose contains NaN or inf.")
        q = x.reshape(len(x), N_JOINTS, QUAT_DIM)
        norms = np.linalg.norm(q, axis=-1, keepdims=True)
        if np.any(norms < 1e-8):
            raise ValueError("A joint quaternion has (near-)zero norm; cannot normalise it.")
        if np.max(np.abs(norms - 1.0)) > 1e-3:
            warnings.warn("Input quaternions were not unit length (training inputs were); renormalising.")
        return (q / norms).reshape(len(x), self.in_dim)

    # ------------------------------------------------------------------ inference
    def predict_flat(self, pose):
        """Return offsets as an (N, 6510) float32 array."""
        h = self.preprocess(pose)
        for i, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return h if self.k is None else h @ self.components + self.mean

    def predict(self, pose):
        """Return offsets as (N, 2170, 3), or (2170, 3) if a single pose was given."""
        single = np.asarray(pose).ndim == 1 or np.asarray(pose).shape == (N_JOINTS, QUAT_DIM)
        out = self.predict_flat(pose).reshape(-1, self.n_verts, 3)
        return out[0] if single else out


def _load_pose_file(path):
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    return np.loadtxt(p, delimiter="," if p.suffix == ".csv" else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default="artifacts", help="artifact directory (default: artifacts)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pose", help=".npy / .csv / .txt file holding one or more 56-D poses")
    src.add_argument("--random", type=int, metavar="N", help="predict for N random unit-quaternion poses")
    ap.add_argument("--seed", type=int, default=0, help="seed for --random (default 0)")
    ap.add_argument("--out", help="save predicted offsets to this .npy file")
    args = ap.parse_args()

    model = ClothPredictor(args.artifacts)
    if args.pose:
        pose = _load_pose_file(args.pose)
    else:
        rng = np.random.default_rng(args.seed)
        q = rng.standard_normal((args.random, N_JOINTS, QUAT_DIM))
        pose = (q / np.linalg.norm(q, axis=-1, keepdims=True)).reshape(args.random, IN_DIM)

    t = time.perf_counter()
    pred = model.predict(pose)
    dt = time.perf_counter() - t
    n = 1 if pred.ndim == 2 else len(pred)

    print(f"model      : {model.meta['selected_config']} (seed {model.meta['best_seed']})")
    print(f"prediction : shape {pred.shape}, mean |offset| {np.abs(pred).mean():.4f}, "
          f"max |offset| {np.abs(pred).max():.4f}")
    print(f"time       : {dt*1e3:.2f} ms for {n} pose(s) (NumPy, CPU, includes first-call overhead)")
    print(f"NOTE       : {model.meta['data_note']}")
    if args.out:
        np.save(args.out, pred)
        print(f"saved      : {args.out}")


if __name__ == "__main__":
    main()
