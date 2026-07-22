#!/usr/bin/env python3
"""Validador de números canónicos — el linter del hardening (epic hardening).

Verifica que cada número citado en un doc con ancla ``CANON:`` coincide con el
registro ``analysis/canonical_numbers.json`` (ver ``scripts/canon.py``). Tres patas:

  1. cita-vs-canónico   citado == round(canónico, decimals) según la policy
  2. variante           citar una variante NO canónica exige ``variant-ok`` explícito
  3. frescura           una key cuyo emisor cambió sin re-emitir bloquea si un doc
                        staged la cita (por hash de contenido, ver canon.stale_entries)

Anclas (el número solo se checa donde un humano declaró el binding en un comentario):

    LaTeX      % CANON: multiplier.robo_hurto_callejero = 4.6
    Typst      // CANON: r_pooled.robo_hurto_callejero = 0.224
    Markdown   <!-- CANON: multiplier.estafa = 76 -->
    non-canon  % CANON: multiplier.robo_hurto_callejero = 17 variant-ok   (incidente)

Uso:
    python3 scripts/check_canon.py --all       # todos los docs vigilados (manual)
    python3 scripts/check_canon.py --staged    # solo archivos staged (pre-commit)
    python3 scripts/check_canon.py --audit      # heurístico: literales sin ancla (nunca gatea)

Sale 0 si todo pasa, 1 si hay algún fallo. stdlib puro.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon  # noqa: E402  (helper local)

ROOT = canon.ROOT

# Globs de docs vigilados (relativos a ROOT).
WATCHED = ["analysis/*.md", "doc/*.typ", "doc/**/*.typ", "doc/ccis/*.tex"]

# CANON: <key> = <valor> [variant-ok]   — dentro de cualquier estilo de comentario.
ANCHOR_RE = re.compile(
    r"CANON:\s*(?P<key>[\w.\-]+)\s*=\s*(?P<val>[~×xX]?\s*[-\d.,]+)(?P<flag>\s+variant-ok)?",
)
# Literal numérico "portante" para --audit (multiplicadores/tasas, no años ni refs).
NUMISH_RE = re.compile(r"(?<![\w.])[×xX~]?\s*\d+\.\d+")


def watched_files() -> list[Path]:
    out: list[Path] = []
    for pat in WATCHED:
        out.extend(ROOT.glob(pat))
    return sorted(set(p for p in out if p.is_file()))


def staged_files() -> list[Path]:
    res = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=ROOT, capture_output=True, text=True,
    )
    files = [ROOT / line for line in res.stdout.splitlines() if line.strip()]
    return [p for p in files if p.is_file()]


def _parse_cited(raw: str) -> Decimal | None:
    """Normaliza el valor citado: quita ×, ~, x y espacios; parsea a Decimal."""
    s = raw.strip().lstrip("×xX~").strip().replace(" ", "")
    # coma como separador de miles (los canónicos usan punto decimal)
    if "," in s and "." in s:
        s = s.replace(",", "")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


# Una línea que "quiso" ser ancla (menciona CANON como palabra y tiene '=') pero no
# parsea → NO debe saltarse en silencio (typo: sin ':', minúsculas, '==', '≈4.6'…).
MALFORMED_RE = re.compile(r"(?i)\bcanon\b")


def _loc(f: Path, lineno: int) -> str:
    try:
        rel = f.relative_to(ROOT)
    except ValueError:
        rel = f
    return f"{rel}:{lineno}"


def find_anchors(files: list[Path]):
    """-> (anchors, malformed)

    anchors:   [(file, lineno, key, cited_raw, has_variant_ok)]
    malformed: [(file, lineno, line)] — líneas con intención de ancla que no parsean.
    """
    anchors, malformed = [], []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            matched = False
            for m in ANCHOR_RE.finditer(line):
                anchors.append((f, i, m.group("key"), m.group("val"), bool(m.group("flag"))))
                matched = True
            if not matched and "=" in line and MALFORMED_RE.search(line):
                malformed.append((f, i, line.strip()))
    return anchors, malformed


def _resolve(key: str, entries: dict, policy: dict) -> tuple[dict | None, str | None, bool]:
    """Resuelve una key de ancla a su entrada.

    - key == 'family.variant' (completa): entrada directa; `by_family=False`.
    - key == 'family': resuelve a la variante CANÓNICA de la policy; `by_family=True`.
    Devuelve (entry|None, resolved_key|None, by_family).
    """
    if key in entries:
        return entries[key], key, False
    if key in policy:
        cv = policy[key].get("canonical_variant")
        full = f"{key}.{cv}"
        if full in entries:
            return entries[full], full, True
    return None, None, False


def check(files: list[Path], reg: dict, staged_set: set[Path] | None) -> list[str]:
    """Devuelve la lista de fallos (vacía = todo OK)."""
    entries = reg.get("entries", {})
    policy = reg.get("policy", {})
    stale = canon.stale_entries(reg)
    fails: list[str] = []

    anchors, malformed = find_anchors(files)
    for f, lineno, line in malformed:
        fails.append(
            f"{_loc(f, lineno)} · ancla CANON malformada (parece intención de ancla, no parsea): {line}"
        )
    for f, lineno, key, cited_raw, variant_ok in anchors:
        loc = _loc(f, lineno)
        entry, rkey, by_family = _resolve(key, entries, policy)
        if entry is None:
            fails.append(f"{loc} · CANON key desconocida: '{key}' (no está en el registro ni en la policy)")
            continue
        if entry.get("needs_policy"):
            fails.append(f"{loc} · '{key}' sin policy declarada (agrega policy['{entry.get('family')}'] al registro)")
            continue
        # Citar por familia ya resuelve a la canónica. Citar la key completa de una variante
        # NO canónica exige 'variant-ok' explícito (fija víctima-vs-incidente a sabiendas).
        if not by_family and not entry.get("canonical") and not variant_ok:
            fails.append(
                f"{loc} · '{key}' es variante NO canónica; cita la familia canónica o marca 'variant-ok' a sabiendas"
            )
            continue
        key = rkey  # para el chequeo de frescura de abajo

        fam = policy.get(entry.get("family"), {})
        decimals = fam.get("decimals", 4)
        mode = fam.get("round_mode", canon.DEFAULT_ROUND_MODE)
        cited = _parse_cited(cited_raw)
        if cited is None:
            fails.append(f"{loc} · valor citado ilegible: '{cited_raw.strip()}'")
            continue
        canonical = canon.round_canonical(entry["value"], decimals, mode)
        # Comparación EXACTA a la precisión declarada: sin esto, un citado dentro de la
        # cuenca de redondeo (p.ej. 4.55 → 4.6) pasaría — justo el drift del incidente.
        cited_decimals = max(0, -cited.as_tuple().exponent)
        if cited_decimals > decimals:
            fails.append(
                f"{loc} · '{key}' citado con más decimales ({cited}) que la policy "
                f"({decimals} dp); usa la forma canónica exacta {canonical}"
            )
            continue
        if cited != canonical:
            prov = entry.get("provenance", {})
            fails.append(
                f"{loc} · '{key}' cita {cited} pero el canónico es {canonical} "
                f"(emitido en {prov.get('git_commit')} desde {prov.get('script')})"
            )
            continue

        # Frescura: bloqueante solo en pre-commit cuando el doc citante está staged;
        # en --all es aviso (no gatea).
        if key in stale:
            blocking = staged_set is not None and f in staged_set
            msg = f"{loc} · '{key}' STALE — {stale[key]}"
            if blocking:
                fails.append(msg)
            else:
                print(f"  ⚠ aviso (stale, no bloquea en este modo): {msg}", file=sys.stderr)

    return fails


def audit(files: list[Path]) -> None:
    """Heurístico (nunca gatea): literales numéricos sin ancla CANON cercana."""
    print("=== AUDIT: literales numéricos sin ancla CANON cercana (heurístico, ruidoso) ===")
    total = 0
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        hits = []
        for i, line in enumerate(lines):
            if NUMISH_RE.search(line) and "CANON:" not in line:
                near = "CANON:" in (lines[i - 1] if i else "") or "CANON:" in (lines[i + 1] if i + 1 < len(lines) else "")
                if not near:
                    hits.append(i + 1)
        if hits:
            total += len(hits)
            print(f"  {f.relative_to(ROOT)}: {len(hits)} líneas con literal sin ancla (p.ej. L{hits[0]})")
    print(f"→ {total} líneas candidatas. Ancla las portantes; el resto queda humano.")


def _staged_emitter_fails(reg: dict, staged: set[Path]) -> list[str]:
    """BUG-3: si un script emisor está staged pero sus entradas quedaron stale (no se
    re-emitió el registro), bloquea — es el replay exacto de afb7c31 (el pivote se
    commiteó y la detección se difirió semanas)."""
    entries = reg.get("entries", {})
    emitter_rels = {e.get("provenance", {}).get("script") for e in entries.values()}
    emitter_rels.discard(None)
    staged_rels = set()
    for p in staged:
        try:
            staged_rels.add(str(p.relative_to(ROOT)))
        except ValueError:
            pass
    staged_emitters = emitter_rels & staged_rels
    if not staged_emitters:
        return []
    stale = canon.stale_entries(reg)
    offending = sorted(
        k for k, e in entries.items()
        if e.get("provenance", {}).get("script") in staged_emitters and k in stale
    )
    if not offending:
        return []
    return [
        f"emisor staged sin re-emitir el registro ({', '.join(sorted(staged_emitters))}): "
        f"corre el pipeline y stagea analysis/canonical_numbers.json "
        f"[stale: {', '.join(offending)}]"
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="todos los docs vigilados")
    g.add_argument("--staged", action="store_true", help="solo archivos staged (pre-commit)")
    g.add_argument("--audit", action="store_true", help="heurístico de literales sin ancla")
    a = ap.parse_args()

    reg = canon.load()

    if a.audit:
        audit(watched_files())
        return 0

    extra_fails: list[str] = []
    if a.staged:
        staged = {p.resolve() for p in staged_files()}
        watched = {p.resolve() for p in watched_files()}
        # BUG-3: emisor staged sin re-emitir el registro = replay de afb7c31.
        extra_fails = _staged_emitter_fails(reg, staged)
        # Si el registro está staged, un cambio de pipeline no puede colar docs viejos:
        # se escala a validar TODOS los docs vigilados.
        if canon.REGISTRY.resolve() in staged:
            print("  registro staged → validando TODOS los docs vigilados", file=sys.stderr)
            files = sorted(watched)
        else:
            files = sorted(staged & watched)
        staged_set = staged
    else:
        files = watched_files()
        staged_set = None

    fails = extra_fails + check(files, reg, staged_set)
    if fails:
        print("✗ check_canon: números fuera de sincronía con el registro canónico:\n", file=sys.stderr)
        for msg in fails:
            print(f"  ✗ {msg}", file=sys.stderr)
        print(f"\n{len(fails)} fallo(s). Corrige el doc, re-emite el registro, o ajusta la policy.", file=sys.stderr)
        return 1
    n = len(find_anchors(files)[0])
    print(f"✓ check_canon: {n} ancla(s) CANON verificada(s) contra el registro.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
