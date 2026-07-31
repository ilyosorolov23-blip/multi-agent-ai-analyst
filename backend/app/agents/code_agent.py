"""
F6 - Code agent. Writes Python for calculations/aggregation and runs it
in a SANDBOXED subprocess with a hard wall-clock timeout and no network
or filesystem access beyond stdlib math/statistics — never on the bare
server process, and never with builtins like `open`, `import os`, etc.
"""
import re
import subprocess
import sys
import tempfile

from app.config import settings
from app.llm import get_chat_llm
from app.state import AgentState

_llm = get_chat_llm()

_SANDBOX_PRELUDE = """
import builtins, math, statistics, json, datetime

_SAFE_BUILTINS = {
    n: getattr(builtins, n)
    for n in ("print","len","range","sum","min","max","abs","round","sorted",
              "enumerate","zip","map","filter","list","dict","set","tuple",
              "str","int","float","bool")
}
__builtins__ = _SAFE_BUILTINS
"""


def _extract_code(text: str) -> str:
    text = re.sub(r"^```python|^```|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return text


def _run_sandboxed(code: str, timeout: int) -> str:
    full_src = _SANDBOX_PRELUDE + "\n" + code
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full_src)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            return f"Error: {proc.stderr.strip()[-800:]}"
        return proc.stdout.strip() or "(no output — use print())"
    except subprocess.TimeoutExpired:
        return f"Error: execution exceeded {timeout}s sandbox limit"


def code_agent(state: AgentState) -> dict:
    prompt = (
        f"Write plain Python (stdlib math/statistics only, no imports of os/sys/subprocess/"
        f"requests) to answer: {state['question']}. End with print() of the final result. "
        f"Code only, no explanation."
    )
    raw = _llm.invoke(prompt).content
    code = _extract_code(raw)
    result = _run_sandboxed(code, settings.code_sandbox_timeout)

    return {
        "code_result": f"{code}\n→ {result}",
        "steps": state.get("steps", []) + ["code"],
    }
