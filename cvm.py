"""CVM (SB-DC / DB-DC) synthetic response generation.

Generalizes dceR/run_cvm_bandung.py's CVM prompt pattern: the scenario text is
no longer hardcoded (Bandung LRT) -- callers supply a `context` string with a
`{BID}` placeholder (mirroring the existing `{PERSONA}` placeholder pattern),
substituted with the actual bid value being asked. Output columns are shaped
to paste directly into CVMToolbox: `Y` (1/0, ready for DCchoice::sbchoice/
dbchoice -- no manual Ya/Tidak recode needed) + `Bid` for SB-DC, `Y1,Y2,Bid1,Bid2`
for DB-DC, plus the persona's own fields as optional covariate columns.

DB-DC uses two sequential Claude calls per persona in one multi-turn
conversation (not one call) -- see module docstring in the project plan for
why: the follow-up bid depends on the real first answer, so the model
shouldn't have to reason about an unresolved counterfactual.
"""

import random

import pandas as pd

from llm import call_claude
from persona import build_persona_text

SYSTEM_PROMPT = """Anda adalah simulator responden survei. Tugas Anda adalah menjawab survei
Contingent Valuation Method (CVM) sebagai seorang individu dengan profil sosio-ekonomi yang
diberikan. Jawab secara realistis sesuai dengan kondisi ekonomi, kebutuhan, dan preferensi
orang tersebut, serta konteks skenario yang diberikan.

Berikan jawaban dalam format JSON berikut:
{
  "wtp_response": "Ya" atau "Tidak",
  "wtp_confidence": angka 1-5 (1=sangat tidak yakin, 5=sangat yakin),
  "reasoning": "alasan singkat 2-3 kalimat dalam Bahasa Indonesia"
}

Pertimbangkan:
- Kemampuan membayar berdasarkan pendapatan dan kondisi rumah tangga
- Konteks skenario dan manfaat yang relevan bagi orang ini
- Konteks sosio-ekonomi secara keseluruhan
Jangan tambahkan teks apapun di luar JSON."""

COVARIATE_COLUMNS = [
    "province", "district", "urban_rural", "age", "gender",
    "education_tier", "kapita", "household_size",
]


def _require_bid_placeholder(context: str) -> None:
    if "{BID}" not in context:
        raise ValueError(
            "Context/narration must include a '{BID}' placeholder where the bid "
            "amount should be substituted, e.g. '...bersedia membayar Rp {BID} per bulan?'"
        )


def _rotate_bids(bid_levels: list[float], n: int, seed: int) -> list[float]:
    """Round-robin + shuffle, same pattern as dceR's bid assignment: each bid
    level gets ~n/len(bid_levels) personas, order randomized."""
    reps = n // len(bid_levels) + 1
    bids = (bid_levels * reps)[:n]
    rng = random.Random(seed + 1)  # +1 so it doesn't reuse the sampler's own seed stream
    rng.shuffle(bids)
    return bids


def _build_question(persona_text: str, context: str, bid: float) -> str:
    filled_context = context.replace("{BID}", f"{bid:,.0f}")
    return f"Profil responden:\n{persona_text}\n\nKonteks survei:\n{filled_context}"


def generate_sbdc_responses(
    personas: pd.DataFrame,
    bid_levels: list[float],
    context: str,
    client,
    model: str,
    seed: int,
    progress_callback=None,
    cancel_event=None,
) -> pd.DataFrame:
    """One Claude call per persona. `context` must contain a `{BID}` placeholder.
    `cancel_event` (a threading.Event, optional) is checked before each
    persona -- if set, generation stops early and returns whatever's been
    collected so far (a real, if partial, result -- not an error)."""
    _require_bid_placeholder(context)
    bids = _rotate_bids(bid_levels, len(personas), seed)

    rows = []
    for i, (_, persona_row) in enumerate(personas.iterrows()):
        if cancel_event is not None and cancel_event.is_set():
            break
        bid = bids[i]
        persona_text = build_persona_text(persona_row)
        question = _build_question(persona_text, context, bid)

        result = call_claude(
            client=client, model=model, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question}],
        )

        parsed = result.parsed or {}
        wtp_label = parsed.get("wtp_response", "")
        row = {
            "respondent_id": persona_row["respondent_id"],
            "Y": 1 if wtp_label == "Ya" else (0 if wtp_label == "Tidak" else None),
            "Bid": bid,
            "wtp_response_label": wtp_label,
            "wtp_confidence": parsed.get("wtp_confidence", ""),
            "reasoning": parsed.get("reasoning", ""),
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "error": result.error,
        }
        for col in COVARIATE_COLUMNS:
            row[col] = persona_row.get(col)
        rows.append(row)

        if progress_callback is not None:
            progress_callback(i + 1, len(personas))

    return pd.DataFrame(rows)


def generate_dbdc_responses(
    personas: pd.DataFrame,
    bid1_levels: list[float],
    up_multiplier: float,
    down_multiplier: float,
    context: str,
    client,
    model: str,
    seed: int,
    progress_callback=None,
    cancel_event=None,
) -> pd.DataFrame:
    """Two sequential Claude calls per persona, in one multi-turn conversation.
    Call 1 asks Bid1; Bid2 = Bid1 * up_multiplier if the real answer was "Ya",
    or Bid1 * down_multiplier if "Tidak" (standard double-bounded design).
    Call 2 replays call 1's user/assistant turns as history before asking the
    follow-up, so Claude's second answer is a genuine continuation of its own
    first answer rather than an independent, potentially inconsistent guess.
    `context` must contain a `{BID}` placeholder (reused for both bids).
    `cancel_event` (a threading.Event, optional) is checked before each
    persona -- if set, generation stops early and returns the partial result."""
    _require_bid_placeholder(context)
    bid1s = _rotate_bids(bid1_levels, len(personas), seed)

    rows = []
    for i, (_, persona_row) in enumerate(personas.iterrows()):
        if cancel_event is not None and cancel_event.is_set():
            break
        bid1 = bid1s[i]
        persona_text = build_persona_text(persona_row)
        question1 = _build_question(persona_text, context, bid1)

        result1 = call_claude(
            client=client, model=model, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question1}],
        )
        parsed1 = result1.parsed or {}
        y1_label = parsed1.get("wtp_response", "")

        bid2 = bid1 * up_multiplier if y1_label == "Ya" else bid1 * down_multiplier
        question2 = f"Pertanyaan lanjutan:\nApakah Anda bersedia membayar Rp {bid2:,.0f} untuk hal ini?"

        messages_so_far = [
            {"role": "user", "content": question1},
            {"role": "assistant", "content": result1.raw_text},
            {"role": "user", "content": question2},
        ]
        result2 = call_claude(
            client=client, model=model, system=SYSTEM_PROMPT,
            messages=messages_so_far,
        )
        parsed2 = result2.parsed or {}
        y2_label = parsed2.get("wtp_response", "")

        row = {
            "respondent_id": persona_row["respondent_id"],
            "Y1": 1 if y1_label == "Ya" else (0 if y1_label == "Tidak" else None),
            "Y2": 1 if y2_label == "Ya" else (0 if y2_label == "Tidak" else None),
            "Bid1": bid1,
            "Bid2": bid2,
            "wtp_confidence_1": parsed1.get("wtp_confidence", ""),
            "wtp_confidence_2": parsed2.get("wtp_confidence", ""),
            "reasoning_1": parsed1.get("reasoning", ""),
            "reasoning_2": parsed2.get("reasoning", ""),
            "input_tokens": result1.usage.input_tokens + result2.usage.input_tokens,
            "output_tokens": result1.usage.output_tokens + result2.usage.output_tokens,
            "error": result1.error or result2.error,
        }
        for col in COVARIATE_COLUMNS:
            row[col] = persona_row.get(col)
        rows.append(row)

        if progress_callback is not None:
            progress_callback(i + 1, len(personas))

    return pd.DataFrame(rows)
