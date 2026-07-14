"""DCE synthetic response generation.

One Claude call per persona, covering *all* questions in that persona's
assigned block at once (not one call per question) -- cheaper, faster, and
more realistic (a real respondent answers a whole block of questions in one
sitting with internal consistency) than a call-per-question design. Returns a
JSON array of {qes, choice, confidence, reasoning}.

Block assignment is round-robin across personas, matching the exact
assignment logic already proven in CVMToolbox's DCE-DataSim R script
(`blk <- blocks[((rid - 1) %% nblocks) + 1]`).

Output shape: ID,BLOCK,q1..qN (+ covariates) -- exactly what CLogit's
Respondent Data Range and DCE-DataSim's own output both use, for direct
round-trip paste-back into the toolbox.

Alternative-position (primacy) bias: LLM personas over-pick whichever
alternative is shown first, so with a fixed "Alternative 1 first" layout the
first design alternative is chosen too often. To neutralise this, the
left-to-right order of the (non-opt-out) alternatives is randomised per
respondent, balanced so each alternative is shown first in half the questions
(see `_balanced_display_orders`). The column labels still carry the design ALT
number, so the persona keeps reporting the *design* alternative it chose and
the recorded response needs no re-mapping -- CLogit lines up exactly as before.
"""

import hashlib
import random

import pandas as pd

from llm import call_claude
from persona import build_persona_text

SYSTEM_PROMPT = """Anda adalah simulator responden survei. Tugas Anda adalah menjawab survei
Discrete Choice Experiment (DCE) sebagai seorang individu dengan profil sosio-ekonomi yang
diberikan. Untuk SETIAP pertanyaan yang diberikan, pilih SATU alternatif yang paling Anda
sukai -- termasuk opsi opt-out ("Tidak memilih satupun") jika itu yang paling sesuai dengan
preferensi Anda -- berdasarkan profil dan konteks yang diberikan. Jawab semua pertanyaan
secara konsisten satu sama lain, seolah-olah Anda benar-benar mengisi satu kuesioner.

Berikan jawaban dalam format JSON array berikut, SATU objek per pertanyaan, urutan sesuai
urutan pertanyaan yang diberikan:
[
  {"qes": <nomor pertanyaan>, "choice": <nomor alternatif (ALT) yang dipilih>,
   "confidence": <1-5>, "reasoning": "<alasan singkat 1-2 kalimat>"},
  ...
]

Jangan tambahkan teks apapun di luar JSON array."""

COVARIATE_COLUMNS = [
    "province", "district", "urban_rural", "age", "gender",
    "education_tier", "kapita", "household_size",
]


def _render_question(question: dict, display_order: list[int] | None = None) -> str:
    """Renders one question's alternatives as a small text table, e.g.:

    Pertanyaan 1:
                  Alternative 1   Alternative 2   Tidak memilih satupun
    visi          menengah        panjang         -
    politk        tinggi          sedang          -
    bid           50              20              -

    `display_order`, if given, is a permutation of the indices of the
    *non-opt-out* alternatives, setting the left-to-right column order in which
    they are shown (used to randomise which alternative appears first and cancel
    the primacy bias). The opt-out column is always kept last. Column labels
    stay "Alternative {ALT}" (the design ALT number), so a swapped layout does
    not change what the persona reports or how the response is recorded.
    """
    real = [a for a in question["alternatives"] if not a["is_optout"]]
    optout = [a for a in question["alternatives"] if a["is_optout"]]
    if display_order is not None:
        real = [real[i] for i in display_order]
    alternatives = real + optout
    col_labels = []
    for alt_obj in alternatives:
        if alt_obj["is_optout"]:
            col_labels.append(f"Tidak memilih satupun (ALT={alt_obj['alt']})")
        else:
            col_labels.append(f"Alternative {alt_obj['alt']}")

    # Preserve attribute order as first seen (not alphabetical) for readability.
    attrs = []
    for alt_obj in alternatives:
        for attr in alt_obj["levels"]:
            if attr not in attrs:
                attrs.append(attr)

    lines = [f"Pertanyaan {question['qes']}:"]
    header = "\t" + "\t".join(col_labels)
    lines.append(header)
    for attr in attrs:
        row_vals = [str(alt_obj["levels"].get(attr, "-")) if not alt_obj["is_optout"] else "-"
                    for alt_obj in alternatives]
        lines.append(attr + "\t" + "\t".join(row_vals))

    return "\n".join(lines)


def _balanced_display_orders(block: dict, seed: int) -> list[list[int]]:
    """Per-respondent left-to-right order for the non-opt-out alternatives of
    every question in a block, used to randomise which alternative is shown
    first and cancel the LLM's first-option (primacy) bias.

    Seeded by the respondent's ID so the layout is reproducible. For the common
    two-alternative case it returns an *exactly balanced* schedule -- each
    alternative is shown first in half the block's questions -- rather than a
    plain coin flip, so there is no drift. For other alternative counts it
    falls back to an independent random permutation per question.
    """
    rng = random.Random(seed)
    questions = block["questions"]
    if not questions:
        return []
    n_real = sum(1 for a in questions[0]["alternatives"] if not a["is_optout"])

    if n_real == 2:
        nq = len(questions)
        swap = [i < nq // 2 for i in range(nq)]  # exactly half swapped
        rng.shuffle(swap)
        return [[1, 0] if s else [0, 1] for s in swap]

    orders = []
    for _ in questions:
        perm = list(range(n_real))
        rng.shuffle(perm)
        orders.append(perm)
    return orders


def _render_block(block: dict, context: str, display_orders: list[list[int]] | None = None) -> str:
    if display_orders is None:
        rendered = [_render_question(q) for q in block["questions"]]
    else:
        rendered = [_render_question(q, display_orders[i]) for i, q in enumerate(block["questions"])]
    rendered_questions = "\n\n".join(rendered)
    return f"Konteks survei:\n{context}\n\n{rendered_questions}"


def _assign_blocks(personas: pd.DataFrame, block_numbers: list[int]) -> list[int]:
    """Round-robin block assignment, matching DCE-DataSim's
    blk <- blocks[((rid - 1) %% nblocks) + 1] logic exactly."""
    nblocks = len(block_numbers)
    return [block_numbers[i % nblocks] for i in range(len(personas))]


def generate_dce_responses(
    personas: pd.DataFrame,
    design: dict,
    context: str,
    client,
    model: str,
    progress_callback=None,
    cancel_event=None,
) -> pd.DataFrame:
    """`design` is the structured object from design_parser.parse_design().
    Returns a DataFrame with columns ID,BLOCK,q1..qN,<covariates>,error --
    ready to paste into CLogit's Respondent Data Range. `cancel_event` (a
    threading.Event, optional) is checked before each persona -- if set,
    generation stops early and returns the partial result."""
    block_numbers = [b["block"] for b in design["blocks"]]
    blocks_by_number = {b["block"]: b for b in design["blocks"]}
    assigned_blocks = _assign_blocks(personas, block_numbers)

    all_qes_numbers = sorted({q["qes"] for b in design["blocks"] for q in b["questions"]})
    q_columns = [f"q{qn}" for qn in all_qes_numbers]

    rows = []
    for i, (_, persona_row) in enumerate(personas.iterrows()):
        if cancel_event is not None and cancel_event.is_set():
            break
        block_num = assigned_blocks[i]
        block = blocks_by_number[block_num]
        persona_text = build_persona_text(persona_row)
        # Randomise which alternative is shown first (balanced per respondent,
        # seeded by respondent_id) to neutralise the persona's first-option
        # primacy bias. Labels keep the design ALT number, so recorded choices
        # stay design-alternative numbers and CLogit lines up unchanged.
        rid = persona_row["respondent_id"]
        try:
            seed = int(rid)
        except (TypeError, ValueError):
            seed = int(hashlib.md5(str(rid).encode()).hexdigest()[:8], 16)
        display_orders = _balanced_display_orders(block, seed)
        block_text = _render_block(block, context, display_orders)
        user_message = f"Profil responden:\n{persona_text}\n\n{block_text}"

        result = call_claude(
            client=client, model=model, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=2000,
        )

        row = {
            "ID": persona_row["respondent_id"],
            "BLOCK": block_num,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "error": result.error,
        }
        for qn in all_qes_numbers:
            row[f"q{qn}"] = None

        if result.error is None and isinstance(result.parsed, list):
            for answer in result.parsed:
                qes = answer.get("qes")
                if qes in all_qes_numbers:
                    row[f"q{qes}"] = answer.get("choice")
        elif result.error is None:
            row["error"] = "unexpected_response_shape"

        for col in COVARIATE_COLUMNS:
            row[col] = persona_row.get(col)
        rows.append(row)

        if progress_callback is not None:
            progress_callback(i + 1, len(personas))

    ordered_cols = ["ID", "BLOCK"] + q_columns + COVARIATE_COLUMNS + ["input_tokens", "output_tokens", "error"]
    return pd.DataFrame(rows)[ordered_cols]


if __name__ == "__main__":
    # Render-only smoke test (no API call) using the same design fixture as
    # design_parser.py's own test.
    from design_parser import parse_raw_alternatives

    raw_alternatives_text = (
        "BLOCK\tQES\tALT\tvisi\tpolitk\tbid\n"
        "1\t1\t1\tmenengah\ttinggi\t50\n"
        "1\t1\t2\tpanjang\tsedang\t20\n"
        "1\t1\t3\t\t\t\n"
        "1\t2\t1\tpanjang\ttinggi\t30\n"
        "1\t2\t2\tpendek\tsedang\t50\n"
        "1\t2\t3\t\t\t\n"
    )
    design = parse_raw_alternatives(raw_alternatives_text)
    print(_render_block(design["blocks"][0], "Pemkot sedang mempertimbangkan kebijakan X."))

    print("\n=== block assignment round-robin check ===")
    import pandas as pd
    fake_personas = pd.DataFrame({"respondent_id": range(7)})
    print(_assign_blocks(fake_personas, block_numbers=[1, 2, 3]))

    print("\n=== position randomisation: balanced display orders per respondent ===")
    blk = design["blocks"][0]
    for rid in range(4):
        orders = _balanced_display_orders(blk, seed=rid)
        firsts = ["ALT1-first" if o[0] == 0 else "ALT2-first" for o in orders]
        print(f"  respondent {rid}: {orders}  -> {firsts}")

    print("\n=== same question rendered ALT1-first vs ALT2-first (labels unchanged) ===")
    q0 = blk["questions"][0]
    print("[order 0,1]\n" + _render_question(q0, [0, 1]))
    print("[order 1,0]\n" + _render_question(q0, [1, 0]))
