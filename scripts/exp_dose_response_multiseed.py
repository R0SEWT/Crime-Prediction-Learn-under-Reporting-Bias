#!/usr/bin/env python3
"""
Dosis-respuesta del label con rigor multi-seed (slr-muw, versión 2).

La versión single-seed mostró banda de ruido ~±0.03 entre variantes. Aquí cada
variante de label se entrena con 3 seeds (el split distrital queda fijo) y se
reporta media ± sd:

  variantes: circular (patrón poblacional), M=1, M=10, M=50, placebo
             (patrón intra-distrital permutado, misma escala distrital)
  modelo: STGNN --target hybrid --fusion attention (circular usa --target latente)

Al final restaura la superficie M=10 y las predicciones canónicas (seed 42).
Salida: data/silver/predictions/label_dose_response_multiseed.json + tabla.
Uso: .venv-embed/bin/python scripts/exp_dose_response_multiseed.py
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
from eval_fusion_oracle import ORACLE_FILE, intra_spearman  # noqa: E402
from exp_label_dose_response import HYBRID_FILE, build, make_placebo  # noqa: E402

PRED_HYBRID = ROOT / "data/silver/predictions/stgnn_latente_h3_hybrid_attention.parquet"
PRED_CIRC = ROOT / "data/silver/predictions/stgnn_latente_h3_attention.parquet"
OUT_JSON = ROOT / "data/silver/predictions/label_dose_response_multiseed.json"
PY = str(ROOT / ".venv-embed/bin/python")
SEEDS = [42, 7, 123]
TEST_YEAR = 2023


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
        raise RuntimeError(f"fallo: {' '.join(cmd)}")


def evaluate(pred_file: Path) -> float:
    oracle = pd.read_parquet(ORACLE_FILE)
    oracle = oracle[oracle["year"] == TEST_YEAR].copy()
    oracle["h3_index"] = oracle["h3_index"].astype(str)
    oracle["ubigeo"] = oracle["ubigeo"].astype(str).str.zfill(6)
    oracle.attrs["year"] = TEST_YEAR
    _, macro = intra_spearman(pred_file, oracle)
    return macro


def train(target: str, seed: int) -> Path:
    run([PY, "scripts/train_stgnn.py", "--target", target,
         "--fusion", "attention", "--epochs", "150", "--no-mlflow",
         "--seed", str(seed)])
    return PRED_HYBRID if target == "hybrid" else PRED_CIRC


def main() -> int:
    results: dict[str, list[float]] = {}
    variants = [
        ("circular", lambda: None, "latente"),
        ("M=1", lambda: build(1.0), "hybrid"),
        ("M=10", lambda: build(10.0), "hybrid"),
        ("M=50", lambda: build(50.0), "hybrid"),
        ("placebo", make_placebo, "hybrid"),
    ]
    for label, setup, target in variants:
        setup()
        results[label] = []
        for seed in SEEDS:
            pred = train(target, seed)
            m = evaluate(pred)
            results[label].append(m)
            print(f"{label} seed={seed}: {m:.4f}", flush=True)
            OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # restaurar canónico
    build(10.0)
    train("hybrid", 42)
    print(f"canónico M=10 seed 42 restaurado: {evaluate(PRED_HYBRID):.4f}")

    print("\n=== DOSIS-RESPUESTA multi-seed (macro rho vs oráculo 2023) ===")
    for label, vals in results.items():
        v = np.array(vals)
        print(f"  {label:10s} {v.mean():.4f} ± {v.std(ddof=1):.4f}   {np.round(v, 4).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
