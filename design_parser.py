"""DCE design ingestion.

Two paths, both producing the same structured design object:
- Primary/recommended: a deterministic parser for OA-Design's Table 2
  "Raw Alternatives" shape (BLOCK,QES,ALT,<attr1>,<attr2>,...), pasted as
  tab- or comma-separated text -- the same strict shape DCE-DataSim/CLogit
  already consume elsewhere in CVMToolbox, so no LLM call is needed here.
- Secondary/fallback: a Claude-based structured-extraction call for OA-Design's
  Table 4 "Questionnaire" card text (BLOK N / Question M cards), for users who
  only have the human-readable printout on hand.

Design object shape (both paths produce this):
{
  "attributes": ["visi", "politk", "bid"],
  "blocks": [
    {"block": 1, "questions": [
      {"qes": 1, "alternatives": [
        {"alt": 1, "levels": {"visi": "menengah", "politk": "tinggi", "bid": "50"}, "is_optout": false},
        {"alt": 2, "levels": {"visi": "panjang", "politk": "sedang", "bid": "20"}, "is_optout": false},
        {"alt": 3, "levels": {}, "is_optout": true}
      ]},
      ...
    ]},
    ...
  ]
}
"""

import json
import re

from llm import call_claude


def _split_line(line: str) -> list[str]:
    """Splits on tab if present (pasted from a spreadsheet grid), else comma
    (pasted as a plain CSV)."""
    if "\t" in line:
        return line.split("\t")
    return line.split(",")


def looks_like_raw_alternatives(text: str) -> bool:
    """Cheap check: does the first non-blank line's first 3 fields read
    BLOCK, QES, ALT (case-insensitive)? Used to decide which parser to try
    first, without committing to a full parse attempt."""
    for line in text.splitlines():
        if line.strip():
            fields = [f.strip().upper() for f in _split_line(line)]
            return fields[:3] == ["BLOCK", "QES", "ALT"]
    return False


def parse_raw_alternatives(text: str) -> dict:
    """Deterministic parser for OA-Design's Table 2 "Raw Alternatives" shape.
    Raises ValueError with a clear message if the header doesn't match."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Pasted design text is empty.")

    header = [f.strip() for f in _split_line(lines[0])]
    if len(header) < 4 or [h.upper() for h in header[:3]] != ["BLOCK", "QES", "ALT"]:
        raise ValueError(
            "Expected a header starting with BLOCK,QES,ALT,<attributes...> "
            f"(OA-Design's 'Raw Alternatives' table), got: {header}"
        )
    attributes = header[3:]

    blocks_map: dict[int, dict[int, list[dict]]] = {}

    for line_no, line in enumerate(lines[1:], start=2):
        fields = [f.strip() for f in _split_line(line)]
        if len(fields) < 3:
            raise ValueError(f"Row {line_no}: expected at least BLOCK,QES,ALT, got: {fields}")
        try:
            block = int(fields[0])
            qes = int(fields[1])
            alt = int(fields[2])
        except ValueError as e:
            raise ValueError(f"Row {line_no}: BLOCK/QES/ALT must be integers, got: {fields[:3]}") from e

        level_values = fields[3:3 + len(attributes)]
        # Pad short rows (ragged paste) with blanks rather than failing outright.
        level_values += [""] * (len(attributes) - len(level_values))
        levels = {attr: val for attr, val in zip(attributes, level_values)}
        is_optout = all(v == "" for v in level_values)

        blocks_map.setdefault(block, {}).setdefault(qes, []).append({
            "alt": alt,
            "levels": {} if is_optout else levels,
            "is_optout": is_optout,
        })

    blocks = []
    for block_num in sorted(blocks_map):
        questions = []
        for qes_num in sorted(blocks_map[block_num]):
            alternatives = sorted(blocks_map[block_num][qes_num], key=lambda a: a["alt"])
            questions.append({"qes": qes_num, "alternatives": alternatives})
        blocks.append({"block": block_num, "questions": questions})

    return {"attributes": attributes, "blocks": blocks}


_DESIGN_EXTRACTION_SYSTEM_PROMPT = """Anda adalah parser teks. Tugas Anda adalah membaca teks kartu kuesioner
Discrete Choice Experiment (DCE) yang di-paste dari Excel, dan mengubahnya menjadi struktur JSON.

Format input biasanya seperti ini (boleh sedikit bervariasi):

BLOK 1
Question 1
	Alternative 1	Alternative 2	None of these
visi	menengah	panjang
politk	tinggi	sedang
bid	50	20

Question 2
	Alternative 1	Alternative 2	None of these
...

BLOK 2
...

Aturan parsing:
- "BLOK N" menandai awal blok baru (integer N).
- "Question M" menandai pertanyaan/choice-set baru dalam blok tersebut (integer M).
- Baris header setelah "Question M" berisi label kolom: "Alternative 1", "Alternative 2", ...,
  dan kolom terakhir adalah opsi opt-out (biasanya berlabel "None of these" atau serupa) -- kolom
  ini SELALU kosong nilainya untuk setiap atribut (levels harus berupa objek kosong {}).
- Setiap baris berikutnya adalah satu atribut: sel pertama = nama atribut, sel-sel berikutnya =
  nilai atribut untuk tiap alternative (kolom opt-out selalu kosong).

Kembalikan HANYA JSON dengan struktur berikut, tanpa teks lain:
{
  "attributes": ["nama_atribut_1", "nama_atribut_2", ...],
  "blocks": [
    {"block": 1, "questions": [
      {"qes": 1, "alternatives": [
        {"alt": 1, "levels": {"nama_atribut_1": "...", ...}, "is_optout": false},
        {"alt": 2, "levels": {...}, "is_optout": false},
        {"alt": 3, "levels": {}, "is_optout": true}
      ]},
      ...
    ]},
    ...
  ]
}

Urutan "alt" mengikuti urutan kolom Alternative 1, 2, ..., lalu opt-out sebagai alt terakhir.
Jangan tambahkan teks apapun di luar JSON."""


def parse_questionnaire_cards(text: str, client, model: str) -> dict:
    """LLM-based structured extraction for OA-Design's Table 4 "Questionnaire"
    card text. One Claude call. Raises RuntimeError if extraction fails after
    retries (see llm.call_claude)."""
    result = call_claude(
        client=client,
        model=model,
        system=_DESIGN_EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
        max_tokens=4000,
    )
    if result.error is not None:
        raise RuntimeError(f"Design extraction failed: {result.error}\nRaw response: {result.raw_text[:500]}")
    if not isinstance(result.parsed, dict) or "attributes" not in result.parsed or "blocks" not in result.parsed:
        raise RuntimeError(f"Design extraction returned unexpected shape: {result.parsed!r}")
    return result.parsed


def parse_design(text: str, client=None, model: str | None = None) -> dict:
    """Tries the deterministic Table 2 parser first; falls back to the
    Claude-based Table 4 parser only if the pasted text doesn't look like
    Table 2's shape. `client`/`model` are required only for the fallback
    path -- omit them to force "Table 2 or bust" (e.g. for automated tests)."""
    if looks_like_raw_alternatives(text):
        return parse_raw_alternatives(text)

    if client is None or model is None:
        raise ValueError(
            "Pasted text doesn't look like OA-Design's 'Raw Alternatives' table "
            "(BLOCK,QES,ALT,... header), and no Claude client was provided to "
            "fall back to the Questionnaire-card parser."
        )
    return parse_questionnaire_cards(text, client, model)


if __name__ == "__main__":
    # Fixture 1: OA-Design Table 2 "Raw Alternatives" (deterministic path),
    # reconstructed from the real exercise.xlsx example (visi/politk/bid,
    # 2 alternatives + opt-out, 3 blocks x 3 questions -- first block shown).
    raw_alternatives_text = (
        "BLOCK\tQES\tALT\tvisi\tpolitk\tbid\n"
        "1\t1\t1\tmenengah\ttinggi\t50\n"
        "1\t1\t2\tpanjang\tsedang\t20\n"
        "1\t1\t3\t\t\t\n"
        "1\t2\t1\tpanjang\ttinggi\t30\n"
        "1\t2\t2\tpendek\tsedang\t50\n"
        "1\t2\t3\t\t\t\n"
    )
    print("=== Deterministic parser (Table 2 shape) ===")
    print("looks_like_raw_alternatives:", looks_like_raw_alternatives(raw_alternatives_text))
    design = parse_raw_alternatives(raw_alternatives_text)
    print(json.dumps(design, indent=2))

    # Fixture 2: OA-Design Table 4 "Questionnaire" cards (LLM fallback path),
    # exact real text captured from exercise.xlsx's OADesign_20260703_202812
    # sheet, BLOK 1 only (for a quick test; full text has 3 blocks).
    questionnaire_text = (
        "BLOK 1\n"
        "Question 1\n"
        "\tAlternative 1\tAlternative 2\tNone of these\n"
        "visi\tmenengah\tpanjang\t\n"
        "politk\ttinggi\tsedang\t\n"
        "bid\t50\t20\t\n"
        "\n"
        "Question 2\n"
        "\tAlternative 1\tAlternative 2\tNone of these\n"
        "visi\tpanjang\tpendek\t\n"
        "politk\ttinggi\tsedang\t\n"
        "bid\t30\t20\t\n"
    )
    print("\n=== looks_like_raw_alternatives on Questionnaire text (should be False) ===")
    print(looks_like_raw_alternatives(questionnaire_text))

    print("\n=== LLM-based parser (Table 4 shape) -- requires ANTHROPIC_API_KEY ===")
    try:
        from llm import get_api_key, make_client, MODEL_MAP
        api_key = get_api_key()
        client = make_client(api_key)
        design2 = parse_design(questionnaire_text, client=client, model=MODEL_MAP["haiku"])
        print(json.dumps(design2, indent=2))
    except ValueError as e:
        print(f"Skipped (no API key available in this environment): {e}")
