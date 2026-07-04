#!/usr/bin/env python3
"""Ablación anti-revisor (slr-p1u.10): ¿el techo DL ~0.26 es artefacto de
150 épocas o de lambda_prior? Celda híbrido(M=10)×attention:
  - 600 épocas × 3 seeds (vs referencia 150ep = 0.252±0.028)
  - lambda_prior in {0.1, 1.0} a 150 épocas × 3 seeds (ref lambda=0.5)
Restaura el canónico al final. Salida: analysis/etapa2_epochs_ablation.md"""
import json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_fusion_oracle import ORACLE_FILE, intra_spearman

PRED = ROOT / "data/silver/predictions/stgnn_latente_h3_hybrid_attention.parquet"
PY = str(ROOT / ".venv-embed/bin/python")
OUT = ROOT / "analysis/etapa2_epochs_ablation.md"

def train(epochs, lam, seed):
    r = subprocess.run([PY, "scripts/train_stgnn.py", "--target", "hybrid",
        "--fusion", "attention", "--epochs", str(epochs), "--lambda-prior", str(lam),
        "--no-mlflow", "--seed", str(seed)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])

def ev():
    o = pd.read_parquet(ORACLE_FILE); o = o[o.year == 2023].copy()
    o["h3_index"] = o.h3_index.astype(str); o["ubigeo"] = o.ubigeo.astype(str).str.zfill(6)
    o.attrs["year"] = 2023
    return intra_spearman(PRED, o)[1]

res = {}
for label, ep, lam in [("600ep_lam0.5", 600, 0.5), ("150ep_lam0.1", 150, 0.1), ("150ep_lam1.0", 150, 1.0)]:
    res[label] = []
    for s in (42, 7, 123):
        train(ep, lam, s); m = ev(); res[label].append(m)
        print(f"{label} seed={s}: {m:.4f}", flush=True)
train(150, 0.5, 42)  # restaurar canónico
lines = ["# Ablación épocas/lambda (slr-p1u.10)", "",
  "Celda híbrido(M=10)×attention. Referencias 150ep/lambda=0.5: 0.252±0.028 (multiseed).", "",
  "| config | macro rho ± sd | seeds |", "|:--|:--|:--|"]
for k, v in res.items():
    a = np.array(v)
    lines.append(f"| {k} | {a.mean():.4f} ± {a.std(ddof=1):.4f} | {np.round(a,4).tolist()} |")
lines += ["", "Veredicto: si 600ep y los lambda alternos quedan dentro de la banda ~0.25±0.03,",
  "el techo DL no es artefacto de presupuesto de entrenamiento ni de la regularización elegida."]
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
