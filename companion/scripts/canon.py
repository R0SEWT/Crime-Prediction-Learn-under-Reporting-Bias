#!/usr/bin/env python3
"""Registro de números canónicos — hardening anti número-stale (epic hardening).

Fuente única de verdad para las cifras portantes (headline) del track empírico.
El pipeline **emite** cada número aquí con procedencia; los docs (papers, reportes)
lo citan por `key` con anclas ``CANON:`` y ``scripts/check_canon.py`` verifica que
lo citado == round(canónico, decimales). Nace de un fallo real: un multiplicador
stale en ``analysis/reporting_rate.md`` (robo ×5.48) sobrevivió a un re-run del
pipeline (commit afb7c31 lo movió a ×4.6) y se coló a un paper.

Registro: ``analysis/canonical_numbers.json`` (VERSIONADO — es memoria científica,
como los reportes; los datos siguen gitignored). Dos zonas:

- ``policy``  → **hand-written**. La única parte humana: por *familia* de número,
  qué ``canonical_variant`` es canónica (víctima vs incidente), ``decimals`` y
  ``round_mode`` (``half_up`` | ``half_even`` | ``floor``). Fija la ambigüedad de
  estimador y precisión de forma explícita, versionada y diffeable. Un script
  **no puede autoproclamarse canónico**: ``canonical`` lo deriva ``emit()`` de la
  policy.
- ``entries`` → **machine-written** por ``emit()``. Determinista (sort_keys), una
  entrada solo cambia si cambia ``round(value)`` o el hash de los inputs → diffs
  limpios.

Frescura (pata C): cada entrada guarda ``script_sha256`` del emisor. Si el script
cambia sin re-emitir, ``check_canon`` marca la entrada stale. Es por CONTENIDO
(no mtime, no ancestría-git): clone-safe y sin depender del orden emit/commit.

Uso como librería (desde un emisor del pipeline)::

    import canon
    canon.emit("multiplier.robo_hurto_callejero", lat/obs,
               variant="victim", unit="latente/observado (adim.)",
               estimator="pooled Σλ*/Σy 2018-2024, r̂ EB victim-level",
               inputs=[rate_file, MININTER], script=__file__)

Uso como CLI:  ``python3 scripts/canon.py [--show]``  (resumen del registro).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN, ROUND_FLOOR
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "analysis" / "canonical_numbers.json"
SCHEMA_VERSION = 1

_ROUND = {"half_up": ROUND_HALF_UP, "half_even": ROUND_HALF_EVEN, "floor": ROUND_FLOOR}
DEFAULT_ROUND_MODE = "half_up"


# ─── utilidades de procedencia ────────────────────────────────────────────────
_SHA_CACHE: dict = {}


def _sha256(path: Path) -> str | None:
    """SHA-256 de un archivo; None si no existe (input regenerable/gitignored).

    Memoizado por (path, size, mtime_ns) para no re-hashear inputs grandes
    (ENAPRES ~238 MB) en cada emisión por categoría dentro de una misma corrida.
    """
    try:
        st = path.stat()
    except (FileNotFoundError, OSError):
        return None
    ckey = (str(path), st.st_size, st.st_mtime_ns)
    cached = _SHA_CACHE.get(ckey)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _SHA_CACHE[ckey] = digest
    return digest


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _head_commit() -> str:
    return _git("rev-parse", "--short", "HEAD") or "uncommitted"


def _is_dirty(rel: str) -> bool:
    """¿El archivo tiene cambios sin commitear respecto a HEAD?"""
    status = _git("status", "--porcelain", "--", rel)
    return bool(status)


def _relpath(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ─── redondeo canónico (respeta la policy) ────────────────────────────────────
def round_canonical(value: float, decimals: int, mode: str = DEFAULT_ROUND_MODE) -> Decimal:
    """Redondea `value` a `decimals` con el modo declarado en la policy."""
    q = Decimal(1).scaleb(-decimals)  # 10^-decimals
    return Decimal(str(value)).quantize(q, rounding=_ROUND[mode])


def display_str(value: float, decimals: int, mode: str = DEFAULT_ROUND_MODE) -> str:
    return f"{round_canonical(value, decimals, mode):.{decimals}f}"


# ─── registro ─────────────────────────────────────────────────────────────────
def load() -> dict:
    if REGISTRY.exists():
        with open(REGISTRY, encoding="utf-8") as f:
            return json.load(f)
    return {"schema_version": SCHEMA_VERSION, "policy": {}, "entries": {}}


def dump(reg: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def emit(
    family: str,
    value: float,
    *,
    variant: str,
    unit: str,
    estimator: str,
    inputs: list,
    script: str,
) -> str:
    """Registra/actualiza una entrada canónica. Devuelve la key ``family.variant``.

    Nunca crashea el pipeline: si falta la policy de la familia, escribe la entrada
    con ``needs_policy=True`` y ``canonical=False`` (check_canon la marca) en vez de
    abortar. La decisión de precisión/variante es humana, no del script.
    """
    reg = load()
    reg.setdefault("schema_version", SCHEMA_VERSION)
    reg.setdefault("policy", {})
    reg.setdefault("entries", {})

    pol = reg["policy"].get(family)
    decimals = pol.get("decimals") if pol else 4
    mode = pol.get("round_mode", DEFAULT_ROUND_MODE) if pol else DEFAULT_ROUND_MODE
    canonical = (pol.get("canonical_variant") == variant) if pol else False

    key = f"{family}.{variant}"
    script_rel = _relpath(Path(script))
    inputs_sha = {_relpath(Path(p)): _sha256(Path(p)) for p in inputs}
    display = display_str(value, decimals, mode)

    prev = reg["entries"].get(key, {})
    prev_prov = prev.get("provenance", {})
    script_sha = _sha256(Path(script))
    needs_policy = pol is None
    value_r = round(float(value), 6)
    # Si NADA material cambió (incluido el valor), preserva TODA la procedencia previa →
    # un re-run no-op no produce diff (git_commit/dirty/emitted_at estables, sin churn).
    unchanged = (
        prev.get("value") == value_r
        and prev.get("display") == display
        and prev.get("canonical") == canonical
        and prev.get("needs_policy", False) == needs_policy
        and prev_prov.get("inputs_sha256") == inputs_sha
        and prev_prov.get("script_sha256") == script_sha
    )
    if unchanged:
        git_commit = prev_prov.get("git_commit")
        dirty = prev_prov.get("dirty")
        emitted_at = prev_prov.get("emitted_at")
    else:
        git_commit = _head_commit()
        dirty = _is_dirty(script_rel)
        emitted_at = _now_iso()

    reg["entries"][key] = {
        "value": value_r,
        "display": display,
        "unit": unit,
        "estimator": estimator,
        "family": family,
        "variant": variant,
        "canonical": canonical,
        "needs_policy": needs_policy,
        "provenance": {
            "script": script_rel,
            "script_sha256": script_sha,
            "git_commit": git_commit,
            "dirty": dirty,
            "inputs_sha256": inputs_sha,
            "emitted_at": emitted_at,
        },
    }
    dump(reg)
    flag = "" if canonical else ("  [NEEDS POLICY]" if pol is None else "  [non-canonical]")
    print(f"  canon.emit {key} = {display}{flag}")
    return key


# ─── frescura (pata C, por hash de contenido del emisor) ──────────────────────
def stale_entries(reg: dict | None = None) -> dict:
    """Entradas cuyo script emisor cambió de contenido desde que se emitieron.

    Devuelve {key: motivo}. Vacío = todo fresco.
    """
    reg = reg or load()
    out = {}
    for key, e in reg.get("entries", {}).items():
        prov = e.get("provenance", {})
        script = ROOT / prov.get("script", "")
        stored = prov.get("script_sha256")
        current = _sha256(script)
        if current is None:
            out[key] = f"script emisor ausente: {prov.get('script')}"
        elif stored != current:
            out[key] = (
                f"script {prov.get('script')} cambió desde la emisión "
                f"(re-corre el emisor y commitea el registro)"
            )
    return out


def _main() -> int:
    reg = load()
    entries = reg.get("entries", {})
    pol = reg.get("policy", {})
    print(f"Registro: {_relpath(REGISTRY)}  ·  {len(entries)} entradas  ·  {len(pol)} familias en policy")
    stale = stale_entries(reg)
    for key in sorted(entries):
        e = entries[key]
        tags = []
        if e.get("canonical"):
            tags.append("canónico")
        if e.get("needs_policy"):
            tags.append("NEEDS-POLICY")
        if key in stale:
            tags.append("STALE")
        tagstr = f"  ({', '.join(tags)})" if tags else ""
        print(f"  {key:<42} {e.get('display'):>10}{tagstr}")
    if stale:
        print(f"\n⚠ {len(stale)} entrada(s) STALE — el emisor cambió sin re-emitir:")
        for key, why in sorted(stale.items()):
            print(f"  {key}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
