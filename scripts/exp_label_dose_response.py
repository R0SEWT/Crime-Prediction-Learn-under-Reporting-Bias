#!/usr/bin/env python3
"""
Dosis-respuesta del target híbrido (slr-muw, paper SIMBig).

Barre la "honestidad del label": M in {1, 5, 50} (M=10 y el circular ya están
medidos), más un PLACEBO (patrón intra-distrital permutado al azar dentro de
distrito×categoría×año, misma escala distrital). Para cada variante:
  1. construye la superficie híbrida (o placebo),
  2. reentrena STGNN --target hybrid --fusion attention (150 épocas, CPU),
  3. evalúa Spearman intra-distrital vs oráculo geocodificado 2023.

Al final RESTAURA la superficie M=10 canónica y reentrena para dejar el
parquet de predicciones consistente con lo reportado en ch5.

Salida: data/silver/predictions/label_dose_response.json + tabla en stdout.
Uso: .venv-embed/bin/python scripts/exp_label_dose_response.py
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

HYBRID_FILE = ROOT / "data/silver/crime_latent_surface_hybrid.parquet"
PRED_FILE = ROOT / "data/silver/predictions/stgnn_latente_h3_hybrid_attention.parquet"
OUT_JSON = ROOT / "data/silver/predictions/label_dose_response.json"
PY = str(ROOT / ".venv-embed/bin/python")
TEST_YEAR = 2023
RNG = np.random.default_rng(42)

# puntos ya medidos (no se re-corren): attention en ambos casos
KNOWN = {
    "circular (M=inf)": 0.2079,   # target poblacional puro, Lightning T4
    "M=10": 0.2762,               # canónico, verificado local (seed 42)
}


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise RuntimeError(f"fallo: {' '.join(cmd)}")


def evaluate() -> float:
    oracle = pd.read_parquet(ORACLE_FILE)
    oracle = oracle[oracle["year"] == TEST_YEAR].copy()
    oracle["h3_index"] = oracle["h3_index"].astype(str)
    oracle["ubigeo"] = oracle["ubigeo"].astype(str).str.zfill(6)
    oracle.attrs["year"] = TEST_YEAR
    _, macro = intra_spearman(PRED_FILE, oracle)
    return macro


def train() -> None:
    run([PY, "scripts/train_stgnn.py", "--target", "hybrid",
         "--fusion", "attention", "--epochs", "150", "--no-mlflow"])


def build(smooth: float) -> None:
    run(["python3", "scripts/build_latent_surface_h3.py",
         "--pattern", "geocoded", "--smooth", str(smooth)])


def make_placebo() -> None:
    """Permuta el patrón intra-distrital del híbrido M=10 (control negativo)."""
    build(10.0)
    df = pd.read_parquet(HYBRID_FILE)
    df["latente"] = (df.groupby(["ubigeo", "year", "crime_cat"])["latente"]
                     .transform(lambda s: s.to_numpy()[RNG.permutation(len(s))]))
    df.to_parquet(HYBRID_FILE, index=False)
    print(f"placebo: patrón permutado dentro de {df.groupby(['ubigeo','year','crime_cat']).ngroups} grupos")


def main() -> int:
    results = dict(KNOWN)
    for label, action in [
        ("M=1", lambda: build(1.0)),
        ("M=5", lambda: build(5.0)),
        ("M=50", lambda: build(50.0)),
        ("placebo (patrón permutado)", make_placebo),
    ]:
        print(f"\n=== {label} ===", flush=True)
        action()
        train()
        results[label] = evaluate()
        print(f"{label}: macro = {results[label]:.4f}", flush=True)
        OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # restaurar el canónico M=10 (superficie + predicciones)
    print("\n=== restaurando M=10 canónico ===", flush=True)
    build(10.0)
    train()
    restored = evaluate()
    print(f"M=10 restaurado: {restored:.4f} (esperado ~0.2762)")
    results["M=10 (restaurado)"] = restored
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n=== DOSIS-RESPUESTA (macro rho intra vs oráculo 2023) ===")
    order = ["placebo (patrón permutado)", "circular (M=inf)", "M=50", "M=10", "M=5", "M=1"]
    for k in order:
        if k in results:
            print(f"  {k:32s} {results[k]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
