#!/usr/bin/env python3
"""
Factorial label × fusión con multi-seed y per-categoría (slr-muw v3).

El multi-seed macro mostró que las variantes de label son indistinguibles bajo
atención. Este factorial responde las dos preguntas que quedan:
  (a) ¿la fusión attention aporta sobre concat, con CIs multi-seed?
  (b) ¿el "rescate de secuestro" por la atención es real o ruido de seed?

Diseño: {circular, M=1} × {concat, attention} × 3 seeds + placebo(attention) × 3.
Registra macro Y per-categoría de cada run.

Al final restaura M=10 canónico (seed 42, attention).
Salida: data/silver/predictions/factorial_label_fusion.json
Uso: .venv-embed/bin/python scripts/exp_factorial_label_fusion.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_fusion_oracle import CATS, ORACLE_FILE, intra_spearman  # noqa: E402
from exp_label_dose_response import build, make_placebo  # noqa: E402

OUT_JSON = ROOT / "data/silver/predictions/factorial_label_fusion.json"
PY = str(ROOT / ".venv-embed/bin/python")
SEEDS = [42, 7, 123]
TEST_YEAR = 2023

PRED = {
    ("latente", "concat"): ROOT / "data/silver/predictions/stgnn_latente_h3.parquet",
    ("latente", "attention"): ROOT / "data/silver/predictions/stgnn_latente_h3_attention.parquet",
    ("hybrid", "concat"): ROOT / "data/silver/predictions/stgnn_latente_h3_hybrid.parquet",
    ("hybrid", "attention"): ROOT / "data/silver/predictions/stgnn_latente_h3_hybrid_attention.parquet",
}


def run_train(target: str, fusion: str, seed: int) -> None:
    r = subprocess.run([PY, "scripts/train_stgnn.py", "--target", target,
                        "--fusion", fusion, "--epochs", "150", "--no-mlflow",
                        "--seed", str(seed)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
        raise RuntimeError("train falló")


def evaluate(target: str, fusion: str) -> dict:
    oracle = pd.read_parquet(ORACLE_FILE)
    oracle = oracle[oracle["year"] == TEST_YEAR].copy()
    oracle["h3_index"] = oracle["h3_index"].astype(str)
    oracle["ubigeo"] = oracle["ubigeo"].astype(str).str.zfill(6)
    oracle.attrs["year"] = TEST_YEAR
    per_cat, macro = intra_spearman(PRED[(target, fusion)], oracle)
    return {"macro": macro, **{c: per_cat[c][0] for c in CATS}}


def main() -> int:
    results: dict[str, list[dict]] = {}
    cells = [
        ("circular×concat", lambda: None, "latente", "concat"),
        ("circular×attention", lambda: None, "latente", "attention"),
        ("M1×concat", lambda: build(1.0), "hybrid", "concat"),
        ("M1×attention", lambda: build(1.0), "hybrid", "attention"),
        ("placebo×attention", make_placebo, "hybrid", "attention"),
    ]
    for label, setup, target, fusion in cells:
        setup()
        results[label] = []
        for seed in SEEDS:
            run_train(target, fusion, seed)
            r = evaluate(target, fusion)
            results[label].append(r)
            print(f"{label} seed={seed}: macro={r['macro']:.4f} "
                  f"secuestro={r['secuestro']:.4f}", flush=True)
            OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # restaurar canónico
    build(10.0)
    run_train("hybrid", "attention", 42)

    print("\n=== FACTORIAL label × fusión (media ± sd de 3 seeds) ===")
    print(f"{'celda':22s} {'macro':16s} {'secuestro':16s} {'robo':16s}")
    for label, runs in results.items():
        for key in ("macro", "secuestro", "robo_hurto_callejero"):
            pass
        m = np.array([r["macro"] for r in runs])
        s = np.array([r["secuestro"] for r in runs])
        rb = np.array([r["robo_hurto_callejero"] for r in runs])
        print(f"{label:22s} {m.mean():.3f}±{m.std(ddof=1):.3f}    "
              f"{s.mean():.3f}±{s.std(ddof=1):.3f}    {rb.mean():.3f}±{rb.std(ddof=1):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
