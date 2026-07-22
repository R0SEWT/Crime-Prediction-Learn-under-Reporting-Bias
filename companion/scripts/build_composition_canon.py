#!/usr/bin/env python3
"""B1: generador canónico de la composición por tipo de delito (excluyendo violencia familiar).

MOTIVO (slr-x50g.1). `data/silver/analysis/composition_excl_dv.json` era un artefacto
**huérfano**: ningún script del repo lo generaba (verificado por grep), y contenía el titular
`L1 0.9912 → 0.1092` ("el de-sesgo recupera la composición real, −89%"). Ese número es falso.
Salía de un vector π de ENAPRES que sumaba víctimas de **olas distintas**: robo solo se mide en
la ola 2024, estafa/extorsión/secuestro solo en 2019–2023, mientras observado y latente sí eran
pooled de 6 años. Una ola de robo contra cinco de estafa. Por ser huérfano, el artefacto pasó dos
auditorías previas: ambas compararon paper-vs-JSON, nunca JSON-vs-ENAPRES.

En base comparable el signo se **invierte**: la corrección aleja la composición del benchmark
(L1 0.2986 → 0.7384, +147%). Ese es el resultado negativo-útil que el paper reporta hoy.

QUÉ HACE. Lee el bloque `composicion` de `validacion.json` (que `validate_latent.py` ya emite
con el conteo de olas por categoría y la base anualizada), excluye violencia familiar --cuyos
universos MININTER y ENAPRES diferen (§3.1)--, renormaliza sobre las cuatro categorías
directamente comparables, y emite L1 y shares al registro canónico. Así la Tabla 1 del paper
deja de ser hand-copiada: al hand-copiarla se colaron tres errores de redondeo (robo latente
39.3 vs 39.2, estafa observada 7.9 vs 7.8, estafa latente 57.9 vs 58.0).

VARIANTE = base del benchmark ENAPRES. `annualised` (canónica) pondera cada categoría por las
olas en que se mide; `raw` suma los conteos crudos. La inversión de signo es robusta a la
elección: +147% anualizada, +237% cruda. La variante es justo el eje del bug original, así que
se registra explícitamente en vez de esconderse.

CAVEAT de frescura: este script rastrea su propio hash y el de `validacion.json`. Si se
regenera la superficie latente, re-correr primero `validate_latent.py` y luego este script.

Uso:  python3 scripts/build_composition_canon.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon  # noqa: E402

ROOT = canon.ROOT
VALIDACION = ROOT / "data/datasets/silver/latent_surface/validacion.json"
DEST = ROOT / "data/silver/analysis/composition_excl_dv.json"

# Universos MININTER y ENAPRES no comparables para violencia familiar (§3.1): la denuncia
# policial cubre un hecho tipificado, la encuesta un episodio auto-reportado en el hogar.
EXCLUDED = "violencia_familiar_sexual"
BASES = {"annualised": "enapres_anualizado", "raw": "enapres_crudo"}
LABELS = {
    "robo_hurto_callejero": "Robbery and Mugging",
    "estafa": "Fraud and Scams",
    "extorsion": "Extortion",
    "secuestro": "Kidnapping",
}


def renorm(vec: np.ndarray, keep: list[int]) -> np.ndarray:
    """Renormaliza a shares sobre el subconjunto comparable."""
    sub = np.asarray(vec, dtype=float)[keep]
    return sub / sub.sum()


def main() -> int:
    if not VALIDACION.exists():
        raise SystemExit(
            f"falta {VALIDACION.relative_to(ROOT)} — correr antes: "
            "python3 scripts/validate_latent.py"
        )
    comp = json.loads(VALIDACION.read_text(encoding="utf-8"))["composicion"]
    cats = comp["cats"]
    if EXCLUDED not in cats:
        raise SystemExit(f"{EXCLUDED} no está en validacion.json['composicion']['cats']")
    keep = [i for i, c in enumerate(cats) if c != EXCLUDED]
    kept = [cats[i] for i in keep]

    observed = renorm(comp["observado"], keep)
    latent = renorm(comp["latente"], keep)
    surveys = {v: renorm(comp[k], keep) for v, k in BASES.items()}
    l1 = {
        v: {"observed": float(np.abs(observed - s).sum()),
            "latent": float(np.abs(latent - s).sum())}
        for v, s in surveys.items()
    }

    # ── consola ────────────────────────────────────────────────────────────────
    print(f"Composición excl. {EXCLUDED} — {len(kept)} categorías comparables")
    print(f"  olas por categoría: {comp['olas_por_cat']}\n")
    hdr = f"  {'categoría':<24}{'observado':>11}{'latente':>10}" + "".join(
        f"{'ENAPRES ' + v:>18}" for v in BASES
    )
    print(hdr)
    for j, c in enumerate(kept):
        row = f"  {c:<24}{observed[j] * 100:10.1f}%{latent[j] * 100:9.1f}%"
        row += "".join(f"{surveys[v][j] * 100:17.1f}%" for v in BASES)
        print(row)
    print()
    for v in BASES:
        d = l1[v]
        delta = 100 * (d["latent"] / d["observed"] - 1)
        print(f"  L1 vs ENAPRES ({v:<10}): observado {d['observed']:.4f} → "
              f"latente {d['latent']:.4f}  ({delta:+.0f}%)")
    print("\n  El de-sesgo ALEJA la composición del benchmark en ambas bases (resultado negativo).")

    # ── artefacto ──────────────────────────────────────────────────────────────
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps({
        "_generated_by": "scripts/build_composition_canon.py",
        "_source": str(VALIDACION.relative_to(ROOT)),
        "_note": (
            "Reemplaza el artefacto huérfano que reportaba L1 0.9912→0.1092. Ese titular salía "
            "de sumar olas ENAPRES desiguales (robo=1 ola vs estafa=5) contra observado/latente "
            "pooled de 6 años. En base comparable el signo se invierte."
        ),
        "excluded_category": EXCLUDED,
        "cats": kept,
        "labels": [LABELS[c] for c in kept],
        "waves_per_cat": {c: comp["olas_por_cat"][c] for c in kept},
        "canonical_base": "annualised",
        "shares": {
            "observed": observed.tolist(),
            "latent": latent.tolist(),
            **{f"enapres_{v}": s.tolist() for v, s in surveys.items()},
        },
        "l1": l1,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n→ {DEST.relative_to(ROOT)}")

    # ── registro canónico ──────────────────────────────────────────────────────
    for v in BASES:
        for series in ("observed", "latent"):
            canon.emit(
                f"composition_l1.{series}", l1[v][series], variant=v,
                unit="distancia L1 de shares vs benchmark ENAPRES (4 cats, excl. VF)",
                estimator=f"shares renormalizadas, benchmark {BASES[v]}",
                inputs=[VALIDACION], script=__file__,
            )
    # La SERIE va en la familia y la base ENAPRES en la variante: así cada celda de la Tabla 1
    # se ancla citando la familia a secas, y el único eje que exige decisión explícita es la
    # base del benchmark — que es justo donde vivía el bug de olas desiguales (B1).
    # Se emiten en PORCENTAJE (no en share 0–1) para que el ancla `CANON:` del paper y la celda
    # de la Tabla 1 se lean idénticas ("88.0"): así una edición de la prosa que se desvíe queda
    # visiblemente reñida con su propia ancla.
    for j, c in enumerate(kept):
        canon.emit(f"composition_share_police.{c}", float(observed[j]) * 100, variant="pooled",
                   unit="% de denuncias MININTER (4 cats, excl. VF)",
                   estimator="conteos pooled 2019–2024 renormalizados",
                   inputs=[VALIDACION], script=__file__)
        canon.emit(f"composition_share_latent.{c}", float(latent[j]) * 100, variant="pooled",
                   unit="% de la superficie latente (4 cats, excl. VF)",
                   estimator="media posterior EB Beta-Binomial (a/(a+b)), renormalizada",
                   inputs=[VALIDACION], script=__file__)
        for v in BASES:
            canon.emit(f"composition_share_survey.{c}", float(surveys[v][j]) * 100, variant=v,
                       unit="% de victimización ENAPRES (4 cats, excl. VF)",
                       estimator=f"benchmark {BASES[v]}, renormalizado",
                       inputs=[VALIDACION], script=__file__)
    print(f"→ {canon.REGISTRY.relative_to(ROOT)} (composition_l1.*, composition_share_*.*)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
