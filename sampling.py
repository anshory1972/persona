"""Consolidated weighted-stratified persona sampler.

Replaces the duplicated logic in dceR's sample_personas.py (weighted_sample())
and run_cvm_bandung.py's inline sample_bandung() -- same core algorithm
(allocate n proportionally to survey-weight mass across urban/rural x
expenditure-quintile strata, then weighted-sample without replacement within
each stratum), but operating on susenas_streamlined.py's persona_base columns
(urban_rural, kapita_quintile, survey_weight) instead of raw SUSENAS codes,
and with geography filtering as a separate, explicit step.
"""

import numpy as np
import pandas as pd


def filter_geography(
    df: pd.DataFrame,
    provinces: list[str] | None = None,
    district_codes: list[int] | None = None,
) -> pd.DataFrame:
    """Filters the persona base to a geographic scope. `provinces` matches the
    decoded province name column; `district_codes` matches the 4-digit BPS
    code column. Both optional; omit both for all of Indonesia."""
    subset = df
    if provinces:
        subset = subset[subset["province"].isin(provinces)]
    if district_codes:
        subset = subset[subset["district_code"].isin(district_codes)]
    return subset.copy()


def weighted_stratified_sample(
    df: pd.DataFrame,
    n: int,
    seed: int,
    stratify: bool = True,
    weight_col: str = "survey_weight",
    strata_cols: tuple[str, str] = ("urban_rural", "kapita_quintile"),
) -> pd.DataFrame:
    """Draws n rows from df using weight_col as sampling probability weight.

    If stratify=True (default), allocates n proportionally across
    strata_cols cells (by total weight mass in each cell), then draws a
    weighted sample without replacement within each cell -- same algorithm
    proven in dceR's sample_personas.py/run_cvm_bandung.py, consolidated here.
    If the pool is smaller than n, returns the whole pool (with a caller-
    visible shortfall, not silently padded).
    """
    rng = np.random.default_rng(seed)

    if len(df) <= n:
        return df.copy()

    if not stratify:
        w = df[weight_col].fillna(1).to_numpy()
        w = w / w.sum()
        idx = rng.choice(len(df), size=n, replace=False, p=w)
        return df.iloc[idx].copy()

    strata_cols = list(strata_cols)
    stratum_weight = df.groupby(strata_cols, observed=True)[weight_col].sum()
    total_weight = stratum_weight.sum()
    stratum_n = (stratum_weight / total_weight * n).round().astype(int)

    # Rounding can miss the target n by a few; adjust the largest strata first.
    diff = n - stratum_n.sum()
    if diff != 0:
        idx_adjust = stratum_n.nlargest(abs(diff)).index
        stratum_n[idx_adjust] += int(np.sign(diff))

    pieces = []
    for keys, group in df.groupby(strata_cols, observed=True):
        k = stratum_n.get(keys, 0)
        if k <= 0 or len(group) == 0:
            continue
        k = min(k, len(group))
        w = group[weight_col].fillna(1).to_numpy()
        w = w / w.sum()
        chosen = rng.choice(len(group), size=k, replace=False, p=w)
        pieces.append(group.iloc[chosen])

    if not pieces:
        return df.iloc[[]].copy()

    return (
        pd.concat(pieces)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def sample_personas(
    persona_base: pd.DataFrame,
    n: int,
    seed: int,
    provinces: list[str] | None = None,
    district_codes: list[int] | None = None,
    stratify: bool = True,
) -> pd.DataFrame:
    """One-call convenience wrapper: filter by geography, then weighted-
    stratified-sample n personas. Raises if the filtered pool is empty."""
    subset = filter_geography(persona_base, provinces, district_codes)
    if len(subset) == 0:
        raise ValueError(
            "No personas match the requested geography scope "
            f"(provinces={provinces}, district_codes={district_codes})."
        )
    return weighted_stratified_sample(subset, n, seed, stratify=stratify)


if __name__ == "__main__":
    from susenas_streamlined import load_persona_base

    base = load_persona_base()

    print("=== National sample, n=200, stratified ===")
    s = sample_personas(base, n=200, seed=42)
    print(f"  n sampled: {len(s)}")
    print(f"  urban/rural: {s['urban_rural'].value_counts().to_dict()}")
    print(f"  kapita_quintile: {s['kapita_quintile'].value_counts().sort_index().to_dict()}")
    print(f"  age mean (std): {s['age'].mean():.1f} ({s['age'].std():.1f})")

    print("\n=== Geo-scoped sample: Jawa Barat only, n=100 ===")
    s2 = sample_personas(base, n=100, seed=42, provinces=["Jawa Barat"])
    print(f"  n sampled: {len(s2)}, all Jawa Barat: {(s2['province'] == 'Jawa Barat').all()}")

    print("\n=== District-scoped sample: Kota Bandung (3273) only, n=30 ===")
    s3 = sample_personas(base, n=30, seed=42, district_codes=[3273])
    print(f"  n sampled: {len(s3)}, all Bandung: {(s3['district_code'] == 3273).all()}")

    print("\n=== Reproducibility check: same seed -> identical sample ===")
    s4a = sample_personas(base, n=50, seed=99, provinces=["Bali"])
    s4b = sample_personas(base, n=50, seed=99, provinces=["Bali"])
    print(f"  identical: {s4a['respondent_id'].tolist() == s4b['respondent_id'].tolist()}")
