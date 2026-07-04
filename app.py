"""Streamlit app: synthetic persona survey responses for CVM (SB-DC/DB-DC) and
DCE, grounded in real SUSENAS demographic data.

Run with: streamlit run app.py
"""

import base64
import threading
import time
from pathlib import Path

import streamlit as st
import pandas as pd

from susenas_streamlined import load_persona_base as _load_persona_base
from sampling import sample_personas
from llm import get_api_key, make_client, MODEL_MAP
from design_parser import parse_design, looks_like_raw_alternatives
from cvm import generate_sbdc_responses, generate_dbdc_responses
from dce import generate_dce_responses

st.set_page_config(page_title="Synthetic Persona Survey Responses", layout="wide", page_icon="🗺️")

# ── Asset helpers ─────────────────────────────────────────────────────────────

def _img_b64(rel_path: str) -> str:
    p = Path(__file__).parent / rel_path
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return ""

_LOGO    = _img_b64("assets/logo.png")
_DCESIM  = _img_b64("assets/toolbox_icons/dcesim.png")
_CLOGIT  = _img_b64("assets/toolbox_icons/clogit.png")

# SVG icon data-URIs for the three tools that use built-in Office imageMso.
# Style matches the custom PNGs: rounded square, white symbol on color fill.
def _svg_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

_SVG_SBDC = _svg_uri(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="6" fill="#4A3D8F"/>'
    '<text x="16" y="22" font-family="Georgia,serif" font-style="italic" font-size="17" '
    'font-weight="bold" fill="white" text-anchor="middle">&#x192;x</text>'
    '</svg>'
)
_SVG_DBDC = _svg_uri(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="6" fill="#1F7A5E"/>'
    '<rect x="7" y="9" width="7" height="5" rx="1" fill="white"/>'
    '<rect x="18" y="9" width="7" height="5" rx="1" fill="white"/>'
    '<rect x="7" y="18" width="7" height="5" rx="1" fill="white"/>'
    '<rect x="18" y="18" width="7" height="5" rx="1" fill="white"/>'
    '</svg>'
)
_SVG_OADESIGN = _svg_uri(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="6" fill="#C07B1A"/>'
    '<line x1="8" y1="12" x2="24" y2="12" stroke="white" stroke-width="1.5"/>'
    '<line x1="8" y1="18" x2="24" y2="18" stroke="white" stroke-width="1.5"/>'
    '<line x1="8" y1="24" x2="24" y2="24" stroke="white" stroke-width="1.5"/>'
    '<line x1="14" y1="8" x2="14" y2="25" stroke="white" stroke-width="1.5"/>'
    '<line x1="20" y1="8" x2="20" y2="25" stroke="white" stroke-width="1.5"/>'
    '</svg>'
)

# ── Global styles ─────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    :root {
      --spa-surface-alt: #E4E9F1;
      --spa-ink: #1E2A3A;
      --spa-ink-muted: #5C6B7A;
      --spa-primary: #33547E;
      --spa-primary-deep: #223A56;
      --spa-line: #C9D3DE;
    }

    .spa-hero {
      background: var(--spa-surface-alt);
      border: 1px solid var(--spa-line);
      border-radius: 12px;
      padding: 22px 26px 22px;
      margin-bottom: 20px;
    }
    .spa-hero-inner {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 24px;
    }
    .spa-hero-body { flex: 1; min-width: 0; }
    .spa-hero-logo { flex: none; display: flex; align-items: center; padding-top: 4px; }
    .spa-hero .eyebrow {
      font-family: Consolas, "SF Mono", monospace;
      font-size: 12px;
      letter-spacing: .14em;
      text-transform: uppercase;
      color: var(--spa-primary);
      margin-bottom: 6px;
    }
    .spa-hero h1 {
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      font-size: 2.05rem;
      font-weight: 700;
      color: var(--spa-ink);
      margin: 0 0 8px;
    }
    .spa-hero p {
      color: var(--spa-ink-muted);
      font-size: 15px;
      max-width: 76ch;
      margin: 0 0 16px;
      line-height: 1.5;
    }
    @media (max-width: 600px) {
      .spa-hero { padding: 14px 14px 16px; }
      .spa-hero-inner {
        flex-direction: column-reverse;
        gap: 14px;
      }
      .spa-hero h1 { font-size: 1.45rem; }
      .spa-hero .eyebrow { font-size: 10px; }
      .spa-hero p { font-size: 14px; }
      .spa-hero-logo img { height: 52px !important; width: 52px !important; }
      .spa-tools { gap: 7px; }
      .spa-tool span { font-size: 11px; }
    }
    .spa-tools {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      border-top: 1px solid var(--spa-line);
      padding-top: 14px;
    }
    .spa-tools .label {
      font-family: Consolas, "SF Mono", monospace;
      font-size: 11px;
      letter-spacing: .1em;
      text-transform: uppercase;
      color: var(--spa-ink-muted);
      margin-right: 4px;
    }
    .spa-tool {
      display: flex;
      align-items: center;
      gap: 6px;
      background: white;
      border: 1px solid var(--spa-line);
      border-radius: 8px;
      padding: 5px 10px 5px 6px;
    }
    .spa-tool img {
      width: 24px;
      height: 24px;
      border-radius: 4px;
      flex: none;
    }
    .spa-tool span {
      font-size: 12px;
      color: var(--spa-ink);
      font-weight: 600;
      white-space: nowrap;
    }

    .spa-step {
      display: flex;
      align-items: center;
      gap: 12px;
      margin: 30px 0 2px;
    }
    .spa-step .badge {
      flex: none;
      width: 32px;
      height: 32px;
      border-radius: 8px;
      background: var(--spa-primary);
      color: #fff;
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      font-weight: 700;
      font-size: 15px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .spa-step h2 {
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      font-size: 1.3rem;
      font-weight: 700;
      color: var(--spa-ink);
      margin: 0;
    }

    .spa-results-header {
      display: flex;
      align-items: center;
      gap: 12px;
      margin: 30px 0 2px;
    }
    .spa-results-header .check {
      flex: none;
      width: 32px;
      height: 32px;
      border-radius: 8px;
      background: #19486A;
      color: #fff;
      font-size: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .spa-results-header h2 {
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      font-size: 1.3rem;
      font-weight: 700;
      color: var(--spa-ink);
      margin: 0;
    }

    [data-testid="stMetric"] {
      background: var(--spa-surface-alt);
      border: 1px solid var(--spa-line);
      border-radius: 10px;
      padding: 12px 14px 8px;
    }
    [data-testid="stMetricValue"] {
      font-variant-numeric: tabular-nums;
      color: var(--spa-primary-deep);
    }
    .stButton > button, .stDownloadButton > button { border-radius: 8px; border: 1px solid var(--spa-line); }
    [data-testid="stDataFrame"] { border-radius: 8px; border: 1px solid var(--spa-line); overflow: hidden; }
    [data-testid="stExpander"] { border: 1px solid var(--spa-line); border-radius: 8px; }
    div[data-baseweb="textarea"], div[data-baseweb="input"], div[data-baseweb="select"] { border-radius: 8px; }
    div[data-testid="stAlert"] { border: 1px solid var(--spa-line); border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def step_header(number, title):
    st.markdown(
        f'<div class="spa-step">'
        f'<div class="badge">{number}</div>'
        f'<h2>{title}</h2></div>',
        unsafe_allow_html=True,
    )


def results_header():
    st.markdown(
        '<div class="spa-results-header">'
        '<div class="check">&#10003;</div>'
        '<h2>Results</h2></div>',
        unsafe_allow_html=True,
    )


@st.cache_data
def load_persona_base():
    return _load_persona_base()


SEED = 42

try:
    api_key = get_api_key(st.secrets)
except ValueError:
    api_key = None

# ── Hero ──────────────────────────────────────────────────────────────────────

def _tool_badge(img_src: str, label: str) -> str:
    return (
        f'<div class="spa-tool">'
        f'<img src="{img_src}" alt="{label}">'
        f'<span>{label}</span>'
        f'</div>'
    )

_logo_html = (
    f'<div style="display:flex;align-items:center;gap:12px;">'
    f'<img src="data:image/png;base64,{_LOGO}" '
    f'alt="SDGs Center Universitas Padjadjaran" '
    f'style="height:72px;width:72px;object-fit:contain;flex:none;">'
    f'<div style="font-family:\'Iowan Old Style\',\'Palatino Linotype\',Georgia,serif;'
    f'line-height:1.3;text-align:left;">'
    f'<div style="font-weight:700;font-size:17px;color:#1E2A3A;">SDGs Center</div>'
    f'<div style="font-size:15px;color:#5C6B7A;">Universitas</div>'
    f'<div style="font-size:15px;color:#5C6B7A;">Padjadjaran</div>'
    f'</div>'
    f'</div>'
    if _LOGO else ""
)

_tools_html = (
    '<div class="spa-tools">'
    '<span class="label">Excel Addins &nbsp;&middot;&nbsp; Valuation Toolbox</span>'
    + _tool_badge(_SVG_SBDC, "SB-DC")
    + _tool_badge(_SVG_DBDC, "DB-DC")
    + _tool_badge(_SVG_OADESIGN, "OA-Design")
    + (_tool_badge(f"data:image/png;base64,{_DCESIM}", "DCE-DataSim") if _DCESIM else "")
    + (_tool_badge(f"data:image/png;base64,{_CLOGIT}", "CLogit") if _CLOGIT else "")
    + '</div>'
)

st.markdown(
    f"""
    <div class="spa-hero">
      <div class="spa-hero-inner">
        <div class="spa-hero-body">
          <div class="eyebrow">SUSENAS &times; Claude &middot; synthpersona</div>
          <h1>Synthetic Persona Survey Responses</h1>
          <p>Simulate how Indonesian households would respond to your CVM or DCE survey &mdash;
          grounded in real SUSENAS microdata and powered by AI. Draw demographically
          representative personas from any province or district, describe your scenario,
          and receive a ready-to-analyse dataset of synthetic responses shaped to paste
          directly into CVMToolbox.</p>
          {_tools_html}
        </div>
        <div class="spa-hero-logo">
          {_logo_html}
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Step 1: method selection ──────────────────────────────────────────────────
step_header(1, "Survey method")
method = st.radio(
    "Method", options=["SB-DC (single-bounded)", "DB-DC (double-bounded)", "DCE"],
    horizontal=True,
)

# ── Step 2: context / narration ───────────────────────────────────────────────
step_header(2, "Context / narration")
if method.startswith("SB-DC") or method.startswith("DB-DC"):
    st.caption(
        "Describe the scenario in the same language your respondents would read it in. "
        "Include a literal `{BID}` placeholder where the bid amount should be substituted."
    )
    default_context = (
        "Pemerintah daerah sedang mempertimbangkan sebuah kebijakan/proyek baru. "
        "Apakah Anda bersedia membayar Rp {BID} per bulan untuk mendukung hal ini?"
    )
else:
    st.caption("Describe the scenario/context respondents should keep in mind while answering the choice questions.")
    default_context = "Pemerintah daerah sedang mempertimbangkan beberapa alternatif kebijakan/produk berikut."

context = st.text_area("Context/narration", value=default_context, height=120)

if (method.startswith("SB-DC") or method.startswith("DB-DC")) and "{BID}" not in context:
    st.warning("Context must include a literal `{BID}` placeholder for the bid amount to be substituted in.")

# ── Step 3: design input (branches by method) ─────────────────────────────────
step_header(3, "Design input")

bid_levels = None
up_multiplier = down_multiplier = None
design = None

if method.startswith("SB-DC"):
    bid_text = st.text_input("Bid levels (comma-separated, Rp)", value="10000, 20000, 35000, 60000, 100000")
    bid_levels = [float(x.strip()) for x in bid_text.split(",") if x.strip()]
    st.caption(f"{len(bid_levels)} bid level(s), rotated evenly across respondents.")

elif method.startswith("DB-DC"):
    bid_text = st.text_input("Initial (Bid1) levels (comma-separated, Rp)", value="10000, 20000, 35000, 60000, 100000")
    bid_levels = [float(x.strip()) for x in bid_text.split(",") if x.strip()]
    up_multiplier = st.number_input("Follow-up multiplier if 'Ya'  (Bid2 = Bid1 × this)", value=2.0, min_value=1.01)
    down_multiplier = st.number_input("Follow-up multiplier if 'Tidak'  (Bid2 = Bid1 × this)", value=0.5, max_value=0.99)
    st.caption(f"{len(bid_levels)} initial bid level(s), rotated evenly across respondents.")

else:  # DCE
    st.caption(
        "Paste OA-Design's Table 2 'Raw Alternatives' (recommended) or "
        "Table 4 'Questionnaire' cards. The parsed design will be shown below for confirmation."
    )
    design_text = st.text_area("Pasted design", height=200)
    if design_text.strip():
        is_raw_alts = looks_like_raw_alternatives(design_text)
        st.caption("Detected shape: " + ("Raw Alternatives (fast parse)" if is_raw_alts else "Questionnaire cards (AI-assisted parse)"))
        if st.button("Parse design"):
            try:
                if is_raw_alts:
                    design = parse_design(design_text)
                else:
                    client = make_client(api_key)
                    design = parse_design(design_text, client=client, model=MODEL_MAP["haiku"])
                st.session_state["parsed_design"] = design
            except (ValueError, RuntimeError) as e:
                st.error(f"Could not parse design: {e}")

    design = st.session_state.get("parsed_design")
    if design is not None:
        st.success(f"Parsed {len(design['attributes'])} attribute(s), {len(design['blocks'])} block(s). "
                   "Please confirm this looks right before generating responses.")
        with st.expander("Show parsed design (confirm before generating)", expanded=True):
            for block in design["blocks"]:
                st.write(f"**BLOK {block['block']}**")
                for q in block["questions"]:
                    rows = []
                    for alt in q["alternatives"]:
                        label = "None of these" if alt["is_optout"] else f"Alternative {alt['alt']}"
                        rows.append({"": label, **alt["levels"]})
                    st.write(f"Question {q['qes']}")
                    st.table(pd.DataFrame(rows).set_index(""))

# ── Step 4: persona sampling ───────────────────────────────────────────────────
step_header(4, "Persona sampling")

persona_base = load_persona_base()
all_provinces = sorted(persona_base["province"].unique())

n_respondents = st.number_input("Number of respondents", min_value=3, value=100, step=1)
stratify = st.checkbox("Stratify by urban/rural × expenditure quintile (recommended)", value=True)

st.markdown(
    '<div style="font-family:\'Iowan Old Style\',\'Palatino Linotype\',Georgia,serif;'
    'font-size:1.05rem;font-weight:700;color:#1E2A3A;margin:18px 0 8px;">'
    "Geographic scope</div>",
    unsafe_allow_html=True,
)
scope_type = st.radio(
    "Scope",
    options=["National", "Province", "District"],
    horizontal=True,
)

provinces = None
district_codes = None

if scope_type == "Province":
    provinces = st.multiselect("Province(s)", options=all_provinces)
elif scope_type == "District":
    district_options = (
        persona_base[["district_code", "district", "province"]]
        .drop_duplicates()
        .sort_values(["province", "district"])
    )
    district_labels = [
        f"{row.district} ({row.province})" for row in district_options.itertuples()
    ]
    label_to_code = dict(zip(district_labels, district_options["district_code"]))
    selected_labels = st.multiselect(
        "District(s)/city(-ies) — e.g. 'Kota Bandung' vs 'Kabupaten Bandung' are distinct",
        options=district_labels,
    )
    district_codes = [label_to_code[label] for label in selected_labels]

# ── Step 5: generate ───────────────────────────────────────────────────────────
step_header(5, "Generate")

model_choice = st.radio(
    "Model", options=["haiku", "sonnet"], index=0, horizontal=True,
    help="Haiku: fast default. Sonnet: higher quality responses.",
)
model = MODEL_MAP[model_choice]

if method.startswith("SB-DC") or method.startswith("DB-DC"):
    ready = bool(bid_levels) and "{BID}" in context
else:
    ready = design is not None

if scope_type == "Province" and not provinces:
    st.warning("Pick at least one province, or switch scope to National.")
    ready = False
elif scope_type == "District" and not district_codes:
    st.warning("Pick at least one district, or switch scope to National.")
    ready = False

gen_active = st.session_state.get("gen_thread") is not None

if st.button("Generate synthetic responses", disabled=not ready or gen_active, type="primary"):
    personas = sample_personas(
        persona_base, n=n_respondents, seed=SEED,
        provinces=provinces or None, district_codes=district_codes or None, stratify=stratify,
    )
    if len(personas) < n_respondents:
        st.warning(f"Only {len(personas)} personas available for this scope (requested {n_respondents}).")

    client = make_client(api_key)
    cancel_event = threading.Event()
    progress_state = {"done": 0, "total": len(personas)}
    result_holder = {}

    def progress_cb(done, total):
        progress_state["done"] = done
        progress_state["total"] = total

    def worker():
        try:
            if method.startswith("SB-DC"):
                res = generate_sbdc_responses(
                    personas, bid_levels, context, client, model, SEED,
                    progress_callback=progress_cb, cancel_event=cancel_event,
                )
            elif method.startswith("DB-DC"):
                res = generate_dbdc_responses(
                    personas, bid_levels, up_multiplier, down_multiplier, context,
                    client, model, SEED, progress_callback=progress_cb, cancel_event=cancel_event,
                )
            else:
                res = generate_dce_responses(
                    personas, design, context, client, model,
                    progress_callback=progress_cb, cancel_event=cancel_event,
                )
            result_holder["results"] = res
        except Exception as e:  # noqa: BLE001
            result_holder["exception"] = str(e)

    thread = threading.Thread(target=worker, daemon=True)
    st.session_state["gen_thread"] = thread
    st.session_state["gen_cancel_event"] = cancel_event
    st.session_state["gen_progress_state"] = progress_state
    st.session_state["gen_result_holder"] = result_holder
    st.session_state["gen_method"] = method
    thread.start()
    st.rerun()

if gen_active:
    thread = st.session_state["gen_thread"]
    cancel_event = st.session_state["gen_cancel_event"]
    progress_state = st.session_state["gen_progress_state"]

    if thread.is_alive():
        done, total = progress_state["done"], progress_state["total"] or 1
        st.progress(done / total, text=f"Generating... {done}/{total}")
        if st.button("Cancel"):
            cancel_event.set()
            st.info("Cancelling — finishing the respondent currently in progress, then stopping.")
        time.sleep(0.5)
        st.rerun()
    else:
        result_holder = st.session_state["gen_result_holder"]
        was_cancelled = cancel_event.is_set()
        st.session_state["gen_thread"] = None
        st.session_state["gen_cancel_event"] = None

        if "exception" in result_holder:
            st.session_state["gen_last_error"] = result_holder["exception"]
        elif "results" in result_holder:
            st.session_state["results"] = result_holder["results"]
            st.session_state["results_method"] = st.session_state["gen_method"]
            st.session_state["gen_last_error"] = None
            if was_cancelled:
                st.session_state["gen_last_cancelled_partial"] = (
                    len(result_holder["results"]), progress_state["total"]
                )
        else:
            st.session_state["gen_last_error"] = "Generation ended unexpectedly with no result."
        st.rerun()

if st.session_state.get("gen_last_error"):
    st.error(f"Generation failed: {st.session_state['gen_last_error']}")
    st.session_state["gen_last_error"] = None

if st.session_state.get("gen_last_cancelled_partial"):
    done, total = st.session_state["gen_last_cancelled_partial"]
    st.warning(f"Cancelled — {done} of {total} respondent(s) completed before stopping.")
    st.session_state["gen_last_cancelled_partial"] = None

# ── Results ────────────────────────────────────────────────────────────────────
if "results" in st.session_state:
    results = st.session_state["results"]
    results_header()

    n_errors = results["error"].notna().sum()
    c1, c2 = st.columns(2)
    c1.metric("Respondents generated", len(results))
    c2.metric("Errors", int(n_errors))

    if st.session_state["results_method"].startswith("SB-DC"):
        toolbox_ready = results[["Y", "Bid"] + [c for c in results.columns if c not in
                                                  ("Y", "Bid", "respondent_id", "wtp_response_label",
                                                   "wtp_confidence", "reasoning", "input_tokens",
                                                   "output_tokens", "error")]]
    elif st.session_state["results_method"].startswith("DB-DC"):
        toolbox_ready = results[["Y1", "Y2", "Bid1", "Bid2"] + [c for c in results.columns if c not in
                                  ("Y1", "Y2", "Bid1", "Bid2", "respondent_id", "wtp_confidence_1",
                                   "wtp_confidence_2", "reasoning_1", "reasoning_2", "input_tokens",
                                   "output_tokens", "error")]]
    else:
        q_cols = [c for c in results.columns if c.startswith("q") and c[1:].isdigit()]
        toolbox_ready = results[["ID", "BLOCK"] + q_cols]

    st.subheader("Toolbox-ready output")
    st.caption("Select and paste this directly into CVMToolbox's Respondent Data / Response / Bid range(s).")
    st.dataframe(toolbox_ready, use_container_width=True)
    st.download_button(
        "Download toolbox-ready CSV", toolbox_ready.to_csv(index=False),
        file_name="synthetic_responses.csv", mime="text/csv",
    )

    with st.expander("Show full results (incl. reasoning, confidence, errors)"):
        st.dataframe(results, use_container_width=True)
        st.download_button(
            "Download full CSV", results.to_csv(index=False),
            file_name="synthetic_responses_full.csv", mime="text/csv",
        )
