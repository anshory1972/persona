"""Generalized persona-narrative builder.

Adapted from dceR/run_cvm_bandung.py's build_persona_text(), but:
- operates on the 8-field minimal persona_base row (susenas_streamlined.py),
  not the full 110-column SUSENAS row -- no employment/insurance/assets/
  bansos/FIES text, since those fields were deliberately left out.
- the geography phrase is built per-persona from that persona's own
  province/district (since personas can now be drawn from anywhere in
  Indonesia), instead of a single hardcoded "Warga Kota Bandung" for the
  whole batch. The *scenario* text (what's being valued) stays a separate,
  caller-supplied piece of context -- not this module's job.
"""

import pandas as pd

# Same income-tier language/thresholds already used in dceR's persona text,
# just no longer entangled with the education/employment/asset narrative.
_INCOME_TIER_THRESHOLDS = [
    (1_000_000, "rendah"),
    (3_000_000, "menengah"),
]
_INCOME_TIER_DEFAULT = "menengah-atas"


def _income_tier(kapita: float) -> str:
    """Tier classification uses per-capita expenditure, matching BPS's own
    convention (poverty lines, expenditure quintiles, etc. are all per-capita
    based) -- this is deliberately NOT changed by the household-total fix
    below, only the narrative text is."""
    if pd.isna(kapita):
        return "tidak diketahui"
    for threshold, tier in _INCOME_TIER_THRESHOLDS:
        if kapita < threshold:
            return tier
    return _INCOME_TIER_DEFAULT


def build_persona_text(row: pd.Series) -> str:
    """Builds an Indonesian-language persona narrative from one persona_base
    row (as produced by susenas_streamlined.load_persona_base())."""

    district = row.get("district") or f"Kabupaten/Kota #{row.get('district_code')}"
    province = row.get("province") or "tidak diketahui"
    urban_rural = (row.get("urban_rural") or "tidak diketahui").lower()

    age = row.get("age")
    age_text = f"{int(age)} tahun" if pd.notna(age) else "tidak diketahui"

    household_size = row.get("household_size")
    household_text = f"{int(household_size)} orang" if pd.notna(household_size) else "tidak diketahui"

    # KAPITA is already per-person (total household expenditure divided by
    # household size) -- showing it alongside household size without also
    # showing the household TOTAL risks the LLM double-counting (treating
    # per-capita as if it were the household total) or under-counting
    # (never scaling back up to household resources) when reasoning about
    # ability to pay for a household-level bid, which is how most CVM
    # payment vehicles are actually framed (a monthly household bill/tax).
    # Showing both figures explicitly, clearly labeled, removes the ambiguity
    # instead of relying on the model to correctly infer or compute it.
    kapita = row.get("kapita")
    if pd.notna(kapita) and pd.notna(household_size) and household_size > 0:
        income_line = (
            f"{_income_tier(kapita)} "
            f"(pengeluaran per kapita/per orang Rp {kapita:,.0f}/bulan; "
            f"total pengeluaran rumah tangga sekitar Rp {kapita * household_size:,.0f}/bulan "
            f"untuk {household_text})"
        )
    elif pd.notna(kapita):
        income_line = f"{_income_tier(kapita)} (pengeluaran per kapita/per orang Rp {kapita:,.0f}/bulan)"
    else:
        income_line = "tidak diketahui"

    persona = f"""
Profil Responden:
- Lokasi         : {district}, {province} ({urban_rural})
- Jenis kelamin  : {row.get("gender") or "tidak diketahui"}
- Usia           : {age_text}
- Pendidikan     : {row.get("education_tier") or "tidak diketahui"}
- Tingkat pendapatan: {income_line}
- Ukuran rumah tangga: {household_text}
""".strip()

    return persona


if __name__ == "__main__":
    from susenas_streamlined import load_persona_base

    base = load_persona_base()
    sample_row = base[base["district_code"] == 3273].iloc[0]
    print(build_persona_text(sample_row))
    print()
    sample_row2 = base.sample(1, random_state=7).iloc[0]
    print(build_persona_text(sample_row2))
