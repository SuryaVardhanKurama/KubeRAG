import re
import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


# Deterministic attack patterns — the "custom Python action" mechanism the docs
# describe (regex logic that Colang can't express natively). Checked BEFORE the
# NeMo LLM call so known jailbreaks and how-to-attack requests are refused at
# zero token cost. NeMo's intent classifier is itself an LLM and can be fooled
# by the very prompts it should detect (e.g. it role-plays as DAN), so this
# string match is the reliable backstop.
ATTACK_PATTERNS = [
    # --- Jailbreak / prompt-injection phrasings ---
    # 'ignore/forget [all/any/your/the] [previous/prior/system] instructions' —
    # covers the classic phrasing plus variants like 'forget all your instructions'
    # that drop 'system'/'previous' or insert 'all'.
    r"ignore (all |any |your |the )?(previous |prior )?instructions",
    r"forget (all |any )?(your |the |my )?(previous |prior |system )?(instructions|guidelines|rules|prompt)",
    r"you are now dan",
    r"i am now dan",
    r"\bdan mode\b",
    r"jailbreak",
    r"developer mode",
    r"pretend you have no (restrictions|rules)",
    r"act as if you were trained differently",
    r"act as an unrestricted ai",
    r"disregard your (training|guidelines|instructions)",
    r"override (your|the) (safety filters|guidelines)",
    r"bypass (your|the) (guidelines|filters|safety)",
    # --- Persona-adoption / identity-shift attacks ---
    r"your name is (?!enterprise it assistant)",
    r"your (new )?name (should be|will be|is now|becomes?)",
    r"you are now (?!senior technical architect)",
    r"act as (?!helpful assistant)",
    # 'make your name as X' / 'rename yourself' / 'call yourself X' — identity-shift
    # phrasings that bypass 'your name is' and 'you are now'.
    r"make your name (as |to be |be )",
    r"change your name (to |as )",
    r"rename yourself",
    r"call yourself",
    r"refer to yourself as",
    r"pretend you are",
    # "pretend you have no instructions" variants — the identity-guard intent
    # example the NeMo classifier keeps missing (it is ambiguous with the
    # jailbreak intent's "pretend you have no restrictions" phrasing).
    r"pretend you have (absolutely )?no instructions",
    # --- How-to-attack / malicious requests ---
    r"how (do|can|to) (i |we )?(exploit|hack|crack|break into|compromise)",
    r"exploit(ing)? (a|the|this|an)?[\w ]{0,40}?vulnerability",
    r"sql injection",
    r"cross.site scripting",
    r"buffer overflow exploit",
    r"privilege escalation",
    r"reverse shell",
    r"keylogger",
    r"phishing",
    r"malware",
    r"ransomware",
    r"ddos attack",
]

# Compiled once at import.
_ATTACK_RE = [re.compile(p, re.IGNORECASE) for p in ATTACK_PATTERNS]

# Refusal used when a deterministic pattern matches — mirrors the NeMo
# jailbreak refusal so blocked requests read consistently.
ATTACK_REFUSAL = (
    "I maintain consistent guidelines regardless of how I am prompted. "
    "I am here to help with Kubernetes, Intel, and networking. "
    "What can I help you with?"
)


def _matches_attack_pattern(message: str) -> bool:
    """True if the message matches any known jailbreak/attack pattern."""
    return any(rx.search(message) for rx in _ATTACK_RE)


_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses llama-3.1-8b-instant for fast intent classification at the gate —
    the heavier llama-3.3-70b-versatile is reserved for the RAG pipeline.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("🛡️ NeMo Guardrails initialised (llama-3.1-8b-instant).")


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    # Deterministic backstop — runs before the NeMo LLM so known attacks are
    # refused without burning a single token on the classifier.
    if _matches_attack_pattern(message):
        logfire.info(f"🛡️ Attack pattern matched | query='{message[:80]}'")
        return True, ATTACK_REFUSAL

    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
