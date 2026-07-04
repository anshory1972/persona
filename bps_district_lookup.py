"""BPS kabupaten/kota (district/city) code -> name lookup.

Source: <repo root>/references/kab.txt, one entry per line in the
form "<4-digit code>. NAME" (e.g. "3273. BANDUNG"). The 4-digit code is the
standard national BPS code: first 2 digits = province (SUSENAS R101), last 2
digits = district within that province (SUSENAS R102, zero-padded).
"""

from pathlib import Path

KAB_TXT_PATH = Path(__file__).resolve().parent.parent / "references" / "kab.txt"

_CACHE: dict[str, str] | None = None


def load_district_lookup(path: Path = KAB_TXT_PATH) -> dict[str, str]:
    """Parses kab.txt into a {'3273': 'BANDUNG', ...} dict. Cached after first call."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    lookup: dict[str, str] = {}
    with open(path, "r", encoding="ascii") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            code, sep, name = line.partition(".")
            if not sep:
                raise ValueError(f"kab.txt line {line_no}: expected '<code>. NAME', got: {raw_line!r}")
            code = code.strip()
            name = name.strip()
            if len(code) != 4 or not code.isdigit():
                raise ValueError(f"kab.txt line {line_no}: expected a 4-digit code, got: {code!r}")
            lookup[code] = name

    _CACHE = lookup
    return lookup


def district_name(province_code: int, district_code: int, path: Path = KAB_TXT_PATH) -> str | None:
    """Combines SUSENAS R101 (province) + R102 (district) into the 4-digit BPS
    code and looks up the district/city name. Returns None if not found."""
    lookup = load_district_lookup(path)
    combined = f"{int(province_code):02d}{int(district_code):02d}"
    return lookup.get(combined)


def is_kota(combined_code: int | str) -> bool:
    """BPS convention (confirmed against kab.txt, holds nationally): within a
    province, the last 2 digits of the 4-digit code are 71+ for Kota
    (city/municipality) and lower for Kabupaten (regency) -- e.g. 3204 =
    Kabupaten Bandung, 3273 = Kota Bandung. kab.txt's raw names don't carry
    this distinction (both just say "BANDUNG"), so district names alone are
    genuinely ambiguous without it."""
    return int(combined_code) % 100 >= 71


def district_full_name(combined_code: int, path: Path = KAB_TXT_PATH) -> str | None:
    """Returns a disambiguated "Kabupaten X" / "Kota X" label (title case) for
    a 4-digit BPS code, e.g. 3204 -> "Kabupaten Bandung", 3273 -> "Kota Bandung"."""
    lookup = load_district_lookup(path)
    raw_name = lookup.get(f"{int(combined_code):04d}")
    if raw_name is None:
        return None
    prefix = "Kota" if is_kota(combined_code) else "Kabupaten"
    return f"{prefix} {raw_name.title()}"


if __name__ == "__main__":
    # Quick self-check against known codes.
    lookup = load_district_lookup()
    print(f"Loaded {len(lookup)} kabupaten/kota entries.")
    checks = {"3273": "BANDUNG", "3578": "SURABAYA", "3171": "JAKARTA SELATAN"}
    for code, expected in checks.items():
        actual = lookup.get(code)
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  {code} -> {actual!r} (expected {expected!r}) [{status}]")
    print("district_name(32, 73) ->", district_name(32, 73))

    print("\n=== Kabupaten/Kota disambiguation check ===")
    for code, expected in [(3204, "Kabupaten Bandung"), (3273, "Kota Bandung"), (3171, "Kota Jakarta Selatan")]:
        actual = district_full_name(code)
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  {code} -> {actual!r} (expected {expected!r}) [{status}]")
