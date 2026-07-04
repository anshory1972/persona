"""Builds/caches a minimal, decoded "persona base" table from dceR's existing
full SUSENAS extract (850k rows x 110 cols) -- selects only the 8 confirmed
persona fields (plus the survey weight needed for sampling), decodes codes to
human-readable labels, and writes a small parquet the Streamlit app loads at
startup.

Source: <repo root>/dceR/susenas_persona_individual.parquet (national,
individual-level, age 17+, already merged with household + consumption blocks).
No need to reprocess raw SUSENAS .dta files -- this only re-derives a leaner
view of data that's already there.
"""

import json
from pathlib import Path

import pandas as pd

from bps_district_lookup import district_full_name

_REPO_ROOT = Path(__file__).resolve().parent.parent
FULL_PARQUET_PATH = _REPO_ROOT / "dceR" / "susenas_persona_individual.parquet"
CODEBOOK_PATH = _REPO_ROOT / "dceR" / "codebook.json"
STREAMLINED_PARQUET_PATH = Path(__file__).parent / "data" / "persona_base.parquet"

# Same 6-tier education collapse already proven in dceR's run_cvm_bandung.py
# (build_persona_text), reused here rather than re-invented.
_EDU_TIER_BOUNDARIES = [
    (4, "SD/sederajat"),
    (10, "SMP/sederajat"),
    (17, "SMA/SMK/sederajat"),
    (19, "Diploma (D1-D3)"),
    (22, "Sarjana/D4/S1/Profesi"),
    (24, "Pascasarjana (S2/S3)"),
]


def _education_tier(r612_code) -> str:
    if pd.isna(r612_code):
        return "Tidak diketahui"
    code = int(r612_code)
    for upper_bound, tier in _EDU_TIER_BOUNDARIES:
        if code <= upper_bound:
            return tier
    return "Tidak diketahui"


def build_streamlined_persona_base(
    full_parquet_path: Path = FULL_PARQUET_PATH,
    codebook_path: Path = CODEBOOK_PATH,
) -> pd.DataFrame:
    """Reads the full SUSENAS parquet + codebook, returns a DataFrame with just
    the 8 minimal persona fields (+ ID and sampling weight), decoded to labels."""

    cols_needed = [
        "URUT", "R101", "R102", "R105", "R407", "R405", "R301", "R612",
        "KAPITA", "KAPITA_QUINTILE", "WEIND",
    ]
    df = pd.read_parquet(full_parquet_path, columns=cols_needed)

    with open(codebook_path, encoding="utf-8") as f:
        codebook = json.load(f)

    province_map = codebook["R101"]["values"]
    urban_rural_map = codebook["R105"]["values"]
    gender_map = codebook["R405"]["values"]

    district_code = df["R101"].astype(int) * 100 + df["R102"].astype(int)
    # Disambiguated "Kabupaten X" / "Kota X" label -- kab.txt's raw names alone
    # are genuinely ambiguous (e.g. both 3204 and 3273 are just "BANDUNG").
    district_full_lookup = {code: district_full_name(code) for code in district_code.unique()}

    out = pd.DataFrame({
        "respondent_id": df["URUT"],
        "province_code": df["R101"].astype(int),
        "province": df["R101"].astype(int).astype(str).map(province_map),
        "district_code": district_code,
        "district": district_code.map(district_full_lookup),
        "urban_rural": df["R105"].astype(int).astype(str).map(urban_rural_map),
        "age": df["R407"].astype("Int64"),
        "gender": df["R405"].astype(int).astype(str).map(gender_map),
        "household_size": df["R301"].astype("Int64"),
        "education_tier": df["R612"].map(_education_tier),
        "kapita": df["KAPITA"].astype(float),
        "kapita_quintile": df["KAPITA_QUINTILE"],
        "survey_weight": df["WEIND"].astype(float),
    })

    return out


def load_persona_base(rebuild: bool = False) -> pd.DataFrame:
    """Loads the cached streamlined persona base, building it first if missing
    or if rebuild=True."""
    if not rebuild and STREAMLINED_PARQUET_PATH.exists():
        return pd.read_parquet(STREAMLINED_PARQUET_PATH)

    df = build_streamlined_persona_base()
    STREAMLINED_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(STREAMLINED_PARQUET_PATH, index=False)
    return df


if __name__ == "__main__":
    df = load_persona_base(rebuild=True)
    print(f"Built persona base: {len(df):,} rows x {len(df.columns)} cols")
    print(df.dtypes)
    print()
    print("Sample rows:")
    print(df.sample(5, random_state=42).to_string(index=False))
    print()
    print("Null counts:")
    print(df.isna().sum())
    print()
    print("Known-code spot check (district 3273 should be BANDUNG):")
    print(df.loc[df["district_code"] == 3273, "district"].unique())
