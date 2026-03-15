from __future__ import annotations

import binascii

import sys
from pathlib import Path


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent


def resolve_resource_path(path_str: str) -> str:
    """
    Resolve a file path for both source runs and PyInstaller builds.

    Resolution order:
      1. absolute path as-is
      2. exact path under bundle/source base dir
      3. basename under bundle/source base dir
      4. cwd-relative path
      5. module-dir-relative path
    """
    raw = (path_str or "").strip()
    if not raw:
        return raw

    p = Path(raw)

    if p.is_absolute() and p.exists():
        return str(p)

    base = app_base_dir()

    candidates = [
        base / raw,
        base / p.name,
        Path.cwd() / raw,
        Path(__file__).resolve().parent / raw,
    ]

    for cand in candidates:
        try:
            if cand.exists():
                return str(cand.resolve())
        except Exception:
            if cand.exists():
                return str(cand)

    return str(base / p.name)

def le_hex_to_int(hex_text: str) -> int:
    raw = bytes.fromhex(hex_text)
    return int.from_bytes(raw, "little", signed=False)


def target_hex_to_prefilter_u64(target_hex: str) -> int:
    """
    Conservative GPU-side target widening.

    The OpenCL path is only a prefilter. It must NEVER be tighter than the CPU verifier,
    or true shares can be lost before CPU verification.

    For common 4-byte little-endian pool targets, we use the standard widened form:
        target64 = floor((2^64 - 1) / floor((2^32 - 1) / target32))

    Then we relax it slightly more to avoid false negatives from kernel approximation.
    """
    raw = bytes.fromhex((target_hex or "").strip())

    if len(raw) == 0:
        return 0xFFFFFFFFFFFFFFFF

    if len(raw) == 4:
        t32 = int.from_bytes(raw, "little", signed=False)
        if t32 == 0:
            return 0xFFFFFFFFFFFFFFFF

        max32 = 0xFFFFFFFF
        max64 = 0xFFFFFFFFFFFFFFFF
        denom = max32 // t32
        if denom <= 0:
            widened = max64
        else:
            widened = max64 // denom

        # Relax a little more to avoid accidentally filtering out a true share.
        widened = min(max64, widened + 0xFFFFFFFF)
        return widened

    if len(raw) >= 8:
        v = int.from_bytes(raw[:8], "little", signed=False)
        return min(0xFFFFFFFFFFFFFFFF, v | 0xFFFFFFFF)

    return int.from_bytes(raw.ljust(8, b"\xff"), "little", signed=False)


def nonce_to_hex_le(nonce: int) -> str:
    return int(nonce & 0xFFFFFFFF).to_bytes(4, "little", signed=False).hex()


def safe_bytes_from_hex(hex_text: str) -> bytes:
    try:
        return bytes.fromhex(hex_text)
    except (ValueError, binascii.Error):
        return b""