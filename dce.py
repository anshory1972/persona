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
"""

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


def _render_question(question: dict) -> str:
    """Renders one question's alternatives as a small text table, e.g.:

    Pertanyaan 1:
                  Alternative 1   Alternative 2   Tidak memilih satupun
    visi          menengah        panjang         -
    politk        tinggi          sedang          -
    bid           50              20              -
    """
    alternatives = question["alternatives"]
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


def _render_block(block: dict, context: str) -> str:
    rendered_questions = "\n\n".join(_render_question(q) for q in block["questions"])
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
        block_text = _render_block(block, context)
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
