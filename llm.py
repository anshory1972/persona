"""Anthropic API wrapper: client setup, robust JSON extraction with retry,
and token/cost tracking.

Fixes a real bug found in dceR/run_cvm_bandung.py's call_claude(): that
function returned immediately on the *first* json.JSONDecodeError (dumping the
whole response into a fallback field and giving up), rather than retrying --
which is exactly what corrupted a real production run when a response's
markdown-fence-stripping edge case failed to parse. Here, a JSON parse failure
is retried like any other transient failure, and only marked as a hard error
after max_retries is exhausted.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field

import anthropic

MODEL_MAP = {
    "haiku": "claude-haiku-4-5-20251001",   # fast/cheap default
    "sonnet": "claude-sonnet-5",             # quality option
    # NOTE: the old dceR script used "claude-sonnet-4-6" for this slot, which
    # is not a real current model id -- claude-sonnet-5 is the current Sonnet.
}

# Rough, illustrative per-million-token pricing (USD) -- for the app's running
# cost estimate only. Verify current pricing before relying on this for
# billing decisions; it is not fetched live.
APPROX_PRICE_PER_MTOK = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
}


def get_api_key(streamlit_secrets=None) -> str:
    """Resolves the Anthropic API key: st.secrets first (if a Streamlit
    secrets mapping is passed in), then the ANTHROPIC_API_KEY env var. Raises
    if neither is set -- callers should catch this and show a manual masked
    input field as the last resort, never logging the key either way.

    NOTE: st.secrets is not a plain dict -- if no secrets.toml exists at all
    (anywhere Streamlit looks), even .get() raises StreamlitSecretNotFoundError
    instead of behaving like a normal missing-key lookup. Caught broadly here
    so a missing secrets file doesn't crash the whole app; it should just fall
    through to the env var / manual-entry path."""
    if streamlit_secrets is not None:
        try:
            key = streamlit_secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            key = None
        if key:
            return key
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    raise ValueError(
        "No Anthropic API key found in st.secrets or the ANTHROPIC_API_KEY "
        "environment variable. Enter one manually."
    )


def make_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)


def extract_json(raw_text: str):
    """Robustly extracts a JSON object/array from a Claude text response.
    Tries, in order: (1) direct json.loads, (2) strip a leading/trailing
    markdown code fence, (3) regex-search for the first {...} or [...] block
    (DOTALL, so it spans newlines) and parse that. Raises json.JSONDecodeError
    if all of these fail -- callers should retry the whole API call, not just
    re-attempt extraction, since a malformed response may not be fixable by
    parsing tricks alone."""
    text = raw_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if text.startswith("```"):
        stripped = text.split("\n", 1)[1] if "\n" in text else text
        stripped = stripped.rsplit("```", 1)[0].strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not extract valid JSON from response", text, 0)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens

    def approx_cost_usd(self, model: str) -> float:
        prices = APPROX_PRICE_PER_MTOK.get(model)
        if prices is None:
            return float("nan")
        return (
            self.input_tokens / 1_000_000 * prices["input"]
            + self.output_tokens / 1_000_000 * prices["output"]
        )


@dataclass
class ClaudeCallResult:
    parsed: dict | list | None
    raw_text: str
    usage: TokenUsage
    error: str | None = None


def call_claude(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    messages: list[dict],
    max_tokens: int = 500,
    max_retries: int = 3,
) -> ClaudeCallResult:
    """Generic single- or multi-turn call. `messages` is the full Anthropic
    messages list (so callers can pass prior turns for a conversation, e.g.
    DB-DC's two-call sequence). Retries on rate limits, transient errors, AND
    JSON-parse failures (unlike the dceR bug this replaces) -- only returns
    error != None after max_retries attempts are exhausted."""
    last_raw = ""
    last_error = ""

    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            raw = msg.content[0].text
            usage = TokenUsage(msg.usage.input_tokens, msg.usage.output_tokens)
            last_raw = raw

            try:
                parsed = extract_json(raw)
            except json.JSONDecodeError:
                last_error = "json_parse_error"
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return ClaudeCallResult(None, raw, usage, error="json_parse_error")

            return ClaudeCallResult(parsed, raw, usage, error=None)

        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            time.sleep(wait)
            last_error = "rate_limit"
        except Exception as e:  # noqa: BLE001 - genuinely want to catch/retry any transient API error
            last_error = str(e)
            if attempt == max_retries - 1:
                return ClaudeCallResult(None, last_raw, TokenUsage(), error=last_error)
            time.sleep(2)

    return ClaudeCallResult(None, last_raw, TokenUsage(), error=last_error or "unknown_error")


if __name__ == "__main__":
    # Unit-test the JSON extraction against deliberately malformed inputs --
    # regression test for the dceR bug (don't give up after one bad response).
    cases = [
        ('{"a": 1, "b": "clean"}', {"a": 1, "b": "clean"}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure, here is the answer:\n```json\n{"a": 2, "b": "with preamble"}\n```\nHope that helps!',
         {"a": 2, "b": "with preamble"}),
        ('Some rambling text {"a": 3} and trailing junk', {"a": 3}),
        ('[{"qes": 1, "choice": 2}, {"qes": 2, "choice": 1}]',
         [{"qes": 1, "choice": 2}, {"qes": 2, "choice": 1}]),
    ]
    print("=== extract_json regression tests ===")
    for i, (raw, expected) in enumerate(cases, 1):
        try:
            result = extract_json(raw)
            status = "PASS" if result == expected else f"FAIL (got {result!r})"
        except json.JSONDecodeError:
            status = "FAIL (raised JSONDecodeError)"
        print(f"  case {i}: {status}")

    print("\n=== case that should genuinely fail (no JSON at all) ===")
    try:
        extract_json("This response has no JSON in it whatsoever.")
        print("  FAIL: should have raised")
    except json.JSONDecodeError:
        print("  PASS: correctly raised JSONDecodeError")
