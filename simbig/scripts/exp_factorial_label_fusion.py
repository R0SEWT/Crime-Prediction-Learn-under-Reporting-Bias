#!/usr/bin/env python3
"""
Factorial label × fusión con multi-seed y per-categoría (slr-muw v4, 20-seed).

El multi-seed macro mostró que las variantes de label son indistinguibles bajo
atención. Este factorial (2×3 COMPLETO: {circular, honest(M=1), placebo} ×
{concat, attention}) responde con CIs pareados por seed:
  (a) ¿la fusión attention aporta sobre concat bajo CADA label (incl. placebo)?
  (b) ¿el "rescate de secuestro" por la atención es real o ruido de seed?
  (c) interacción label×fusión: Δ(honest attn−concat) − Δ(placebo attn−concat).

Diseño: {circular, honest(M=1), placebo} × {concat, attention} × 20 seeds.
El placebo se reconstruye determinista (RNG=42 fresco) para que concat y
attention compartan EXACTAMENTE el mismo patrón permutado (contraste pareado
y continuidad con la celda placebo×attention de v3).

Al final restaura M=10 canónico (seed 42, attention).
Salida: factorial_label_fusion.json (+ _contrasts.json con los deltas y CIs)
Uso: .venv-embed/bin/python scripts/exp_factorial_label_fusion.py [--seeds N]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_fusion_oracle import CATS, ORACLE_FILE, intra_spearman  # noqa: E402
import exp_label_dose_response as dr  # noqa: E402
from exp_label_dose_response import build, make_placebo  # noqa: E402

OUT_JSON = ROOT / "data/silver/predictions/factorial_label_fusion.json"
CONTRASTS_JSON = ROOT / "data/silver/predictions/factorial_label_fusion_contrasts.json"
PY = str(ROOT / ".venv-embed/bin/python")
# 20 seeds por defecto (los 3 originales de v3 primero, para continuidad).
DEFAULT_SEEDS = [42, 7, 123, 11, 13, 17, 19, 23, 29, 31,
                 37, 41, 43, 47, 53, 59, 61, 67, 71, 73]
TEST_YEAR = 2023


def placebo_setup() -> None:
    """Reconstruye el placebo con RNG determinista (42) fresco, de modo que
    TODAS las llamadas produzcan el MISMO patrón permutado — así el contraste
    placebo(attention−concat) es pareado y reproduce el placebo de v3."""
    dr.RNG = np.random.default_rng(42)
    make_placebo()


def paired_ci(a, b, n_boot: int = 2000, seed: int = 0):
    """CI 95% del delta pareado media(a−b) por bootstrap sobre pares (por seed).
    nan-aware: descarta el par si cualquiera de los dos es no finito (una cabeza
    rara puede colapsar a constante en algún seed → ρ indefinido)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS),
                    help="número de seeds por celda (default 20)")
    args = ap.parse_args()
    seeds = DEFAULT_SEEDS[:args.seeds]
    print(f"Factorial 2×3 COMPLETO × {len(seeds)} seeds = {6 * len(seeds)} runs",
          flush=True)

    results: dict[str, list[dict]] = {}
    cells = [
        ("circular×concat", lambda: None, "latente", "concat"),
        ("circular×attention", lambda: None, "latente", "attention"),
        ("honest×concat", lambda: build(1.0), "hybrid", "concat"),
        ("honest×attention", lambda: build(1.0), "hybrid", "attention"),
        ("placebo×concat", placebo_setup, "hybrid", "concat"),
        ("placebo×attention", placebo_setup, "hybrid", "attention"),
    ]
    for label, setup, target, fusion in cells:
        setup()
        results[label] = []
        for seed in seeds:
            run_train(target, fusion, seed)
            r = evaluate(target, fusion)
            results[label].append(r)
            print(f"{label} seed={seed}: macro={r['macro']:.4f} "
                  f"secuestro={r['secuestro']:.4f}", flush=True)
            OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # restaurar canónico (deja las predicciones consistentes con ch5)
    build(10.0)
    run_train("hybrid", "attention", 42)

    def col(label: str, key: str) -> np.ndarray:
        return np.array([r[key] for r in results[label]], float)

    # ── tabla media ± sd por celda ──
    print(f"\n=== FACTORIAL label × fusión (media ± sd de {len(seeds)} seeds) ===")
    print(f"{'celda':22s} {'macro':16s} {'secuestro':16s} {'robo':16s}")
    def ms(x):
        return f"{np.nanmean(x):.3f}±{np.nanstd(x, ddof=1):.3f}"
    for label in results:
        print(f"{label:22s} {ms(col(label, 'macro')):16s} "
              f"{ms(col(label, 'secuestro')):16s} {ms(col(label, 'robo_hurto_callejero'))}")

    # ── contrastes pareados por seed (delta media + CI95 bootstrap) ──
    contrasts: dict[str, dict] = {}

    def add(name: str, la: str, lb: str, key: str = "macro") -> None:
        mean, lo, hi = paired_ci(col(la, key), col(lb, key))
        sig = (lo > 0) or (hi < 0)
        contrasts[name] = {"delta": mean, "ci": [lo, hi], "sig": sig, "metric": key}
        print(f"  {name:36s} Δ={mean:+.4f} [{lo:+.4f}, {hi:+.4f}] {'SIG' if sig else 'ns'}")

    print("\n=== CONTRASTES PAREADOS (macro ρ, CI95 bootstrap por seed) ===")
    add("fusion attn−concat | circular", "circular×attention", "circular×concat")
    add("fusion attn−concat | honest",   "honest×attention",   "honest×concat")
    add("fusion attn−concat | placebo",  "placebo×attention",  "placebo×concat")
    add("label honest−circular | attn",  "honest×attention",   "circular×attention")
    add("label honest−placebo | attn",   "honest×attention",   "placebo×attention")
    print("\n=== SECUESTRO (rescate de categoría rara, CI95) ===")
    add("fusion attn−concat | circular", "circular×attention", "circular×concat", "secuestro")
    add("fusion attn−concat | honest",   "honest×attention",   "honest×concat",   "secuestro")
    add("fusion attn−concat | placebo",  "placebo×attention",  "placebo×concat",  "secuestro")

    # ── interacción label×fusión, pareada por seed ──
    d_honest = col("honest×attention", "macro") - col("honest×concat", "macro")
    d_plac = col("placebo×attention", "macro") - col("placebo×concat", "macro")
    imean, ilo, ihi = paired_ci(d_honest, d_plac)
    isig = (ilo > 0) or (ihi < 0)
    contrasts["interaction (honest−placebo) of fusion gain"] = {
        "delta": imean, "ci": [ilo, ihi], "sig": isig, "metric": "macro"}
    print("\n=== INTERACCIÓN label×fusión (macro ρ) ===")
    print(f"  Δ_int = (honest attn−concat) − (placebo attn−concat) = "
          f"{imean:+.4f} [{ilo:+.4f}, {ihi:+.4f}] {'SIG' if isig else 'ns'}")

    CONTRASTS_JSON.write_text(json.dumps(contrasts, indent=2, ensure_ascii=False))
    print(f"\nContrastes → {CONTRASTS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
