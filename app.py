"""Streamlit app: synthetic persona survey responses for CVM (SB-DC/DB-DC) and
DCE, grounded in real SUSENAS demographic data. See the project plan for the
full design rationale.

Run with: streamlit run app.py
"""

import threading
import time

import streamlit as st
import pandas as pd

from susenas_streamlined import load_persona_base as _load_persona_base
from sampling import sample_personas
from llm import get_api_key, make_client, MODEL_MAP
from design_parser import parse_design, looks_like_raw_alternatives
from cvm import generate_sbdc_responses, generate_dbdc_responses
from dce import generate_dce_responses

st.set_page_config(page_title="Synthetic Persona Survey Responses", layout="wide", page_icon="🗺️")

st.markdown(
    """
    <style>
    :root {
      --spa-surface-alt: #E4E9F1;
      --spa-ink: #1E2A3A;
      --spa-ink-muted: #5C6B7A;
      --spa-primary: #33547E;
      --spa-primary-deep: #223A56;
      --spa-gold: #C08A2E;
      --spa-gold-soft: #F6ECD9;
      --spa-line: #C9D3DE;
    }

    .spa-hero {
      background: var(--spa-surface-alt);
      border: 1px solid var(--spa-line);
      border-radius: 12px;
      padding: 22px 26px 26px;
      margin-bottom: 4px;
    }
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
      margin: 0;
      line-height: 1.5;
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
      border-radius: 50%;
      background: var(--spa-gold-soft);
      color: var(--spa-primary-deep);
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      font-weight: 700;
      font-size: 15px;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--spa-gold);
    }
    .spa-step h2 {
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

    .stButton > button, .stDownloadButton > button {
      border-radius: 8px;
      border: 1px solid var(--spa-line);
    }
    [data-testid="stDataFrame"] {
      border-radius: 8px;
      border: 1px solid var(--spa-line);
      overflow: hidden;
    }
    [data-testid="stExpander"] {
      border: 1px solid var(--spa-line);
      border-radius: 8px;
    }
    div[data-baseweb="textarea"], div[data-baseweb="input"], div[data-baseweb="select"] {
      border-radius: 8px;
    }
    div[data-baseweb="textarea"] textarea, div[data-baseweb="input"] input {
      border: 1px solid var(--spa-line) !important;
    }
    div[data-testid="stAlert"] {
      border: 1px solid var(--spa-line);
      border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def step_header(number, title):
    st.markdown(
        f'<div class="spa-step"><div class="badge">{number}</div><h2>{title}</h2></div>',
        unsafe_allow_html=True,
    )


@st.cache_data
def load_persona_base():
    """Cached wrapper -- the underlying parquet read/decode only needs to run
    once per server process, not on every rerun (Streamlit reruns the whole
    script on every interaction)."""
    return _load_persona_base()

# ── API key: auto-detect from secrets/environment, else ask inline ──────────
# No sidebar and no seed control -- this app doesn't need replication control
# for now, so sampling/generation use a fixed internal seed instead of a
# user-facing input.
SEED = 42

api_key = None
try:
    api_key = get_api_key(st.secrets)
except ValueError:
    api_key = None

st.markdown(
    """
    <div class="spa-hero">
      <div class="eyebrow">SUSENAS &times; Claude &middot; synthpersona</div>
      <h1>Synthetic Persona Survey Responses</h1>
      <p>Generate behaviorally plausible synthetic responses to CVM or DCE surveys, using
      real SUSENAS-based Indonesian personas and Claude. Paste a design from CVMToolbox
      (Excel add-in), or enter CVM bid levels directly &mdash; the output is shaped to
      paste straight back into the toolbox.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not api_key:
    api_key = st.text_input(
        "Anthropic API key", type="password",
        help="Never stored or logged. Set ANTHROPIC_API_KEY in the environment or "
             ".streamlit/secrets.toml to skip this field.",
    )

# ── Step 1: method selection ──────────────────────────────────────────────────
step_header(1, "Survey method")
method = st.radio(
    "Method", options=["SB-DC (single-bounded CVM)", "DB-DC (double-bounded CVM)", "DCE"],
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
    col1, col2 = st.columns(2)
    up_multiplier = col1.number_input("Follow-up multiplier if 'Ya' (Bid2 = Bid1 x this)", value=2.0, min_value=1.01)
    down_multiplier = col2.number_input("Follow-up multiplier if 'Tidak' (Bid2 = Bid1 x this)", value=0.5, max_value=0.99)
    st.caption(f"{len(bid_levels)} initial bid level(s), rotated evenly across respondents.")

else:  # DCE
    st.caption(
        "Paste OA-Design's Table 2 'Raw Alternatives' (recommended -- parsed deterministically, no API "
        "call needed) or Table 4 'Questionnaire' cards (parsed via Claude, shown below for confirmation)."
    )
    design_text = st.text_area("Pasted design", height=200)
    if design_text.strip():
        is_raw_alts = looks_like_raw_alternatives(design_text)
        st.caption("Detected shape: " + ("Raw Alternatives (deterministic parse)" if is_raw_alts else "Questionnaire cards (Claude-assisted parse)"))
        if st.button("Parse design"):
            try:
                if is_raw_alts:
                    design = parse_design(design_text)
                else:
                    if not api_key:
                        st.error("An API key is required to parse Questionnaire-card text.")
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

col1, col2 = st.columns(2)
n_respondents = col1.number_input("Number of respondents", min_value=3, value=100, step=1)
stratify = col2.checkbox("Stratify by urban/rural x expenditure quintile (recommended)", value=True)

st.markdown(
    '<div style="font-family:\'Iowan Old Style\',\'Palatino Linotype\',Georgia,serif;'
    'font-size:1.05rem;font-weight:700;color:var(--spa-ink);margin:18px 0 8px;">'
    "Geographic scope</div>",
    unsafe_allow_html=True,
)
scope_type = st.radio(
    "Scope",
    options=["National (all Indonesia)", "Province", "District"],
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
        "District(s)/city(-ies) -- e.g. 'Kota Bandung' vs 'Kabupaten Bandung' are distinct",
        options=district_labels,
    )
    district_codes = [label_to_code[label] for label in selected_labels]

# ── Step 5: generate ───────────────────────────────────────────────────────────
step_header(5, "Generate")

model_choice = st.radio(
    "Model", options=["haiku", "sonnet"], index=0, horizontal=True,
    help="Haiku: fast/cheap default. Sonnet: higher quality, more expensive.",
)
model = MODEL_MAP[model_choice]

ready = bool(api_key)
if method.startswith("SB-DC") or method.startswith("DB-DC"):
    ready = ready and "{BID}" in context and bid_levels
else:
    ready = ready and design is not None

if scope_type == "Province" and not provinces:
    st.warning("Province scope is selected but no province is chosen -- pick at least one, or switch to National.")
    ready = False
elif scope_type == "District" and not district_codes:
    st.warning("District scope is selected but no district is chosen -- pick at least one, or switch to National.")
    ready = False

# Generation runs in a background thread with a cancellation flag, because
# Streamlit executes one script run synchronously per interaction -- a plain
# `for` loop here would block the whole app and a "Cancel" click wouldn't be
# processed until the loop finished on its own. Instead: start a thread, then
# do short poll-and-rerun cycles so the script keeps returning control to
# Streamlit (and can therefore notice a Cancel click) between polls.
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
        # A background thread's unhandled exception is silently swallowed by
        # Python (printed to stderr, never raised in the main thread) -- without
        # this try/except, a genuine API/network failure here would leave
        # result_holder empty and crash the polling code below with a
        # confusing KeyError instead of a real error message.
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
        except Exception as e:  # noqa: BLE001 - genuinely want to surface any failure to the UI thread
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
            st.info("Cancelling -- finishing the respondent currently in progress, then stopping.")
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
            # Thread ended without setting either key -- shouldn't happen given
            # the worker's own try/except, but fail loud rather than KeyError.
            st.session_state["gen_last_error"] = "Generation thread ended unexpectedly with no result and no captured exception."
        st.rerun()

if st.session_state.get("gen_last_error"):
    st.error(f"Generation failed: {st.session_state['gen_last_error']}")
    st.session_state["gen_last_error"] = None

if st.session_state.get("gen_last_cancelled_partial"):
    done, total = st.session_state["gen_last_cancelled_partial"]
    st.warning(f"Cancelled -- {done} of {total} respondent(s) completed before stopping.")
    st.session_state["gen_last_cancelled_partial"] = None

# ── Results ────────────────────────────────────────────────────────────────────
if "results" in st.session_state:
    results = st.session_state["results"]
    step_header("&#10003;", "Results")

    n_errors = results["error"].notna().sum()
    total_input_tokens = results["input_tokens"].sum()
    total_output_tokens = results["output_tokens"].sum()
    approx_cost = (
        total_input_tokens / 1_000_000 * 1.0 + total_output_tokens / 1_000_000 * 5.0
        if model == MODEL_MAP["haiku"]
        else total_input_tokens / 1_000_000 * 3.0 + total_output_tokens / 1_000_000 * 15.0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents", len(results))
    c2.metric("Errors", int(n_errors))
    c3.metric("Tokens (in/out)", f"{total_input_tokens:,}/{total_output_tokens:,}")
    c4.metric("Approx. cost (USD)", f"${approx_cost:.3f}")

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
