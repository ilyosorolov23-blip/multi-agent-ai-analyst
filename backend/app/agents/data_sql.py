"""
F5 - Data agent (text-to-SQL). Writes a SQL query from the question,
runs it against a *read-only* view of the SQLite database, and returns
the result. Hard guard: only SELECT statements are ever executed.
"""
import re

from langchain_community.utilities import SQLDatabase

from app.config import settings
from app.llm import get_chat_llm
from app.state import AgentState

_llm = get_chat_llm()
_db = None


def _database() -> SQLDatabase:
    global _db
    if _db is None:
        _db = SQLDatabase.from_uri(f"sqlite:///{settings.sqlite_db_path}")
    return _db


_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|pragma)\b", re.I)


def _extract_sql(text: str) -> str:
    text = text.strip()
    # strip markdown fences if the model added them
    text = re.sub(r"^```sql|^```|```$", "", text, flags=re.MULTILINE).strip()
    return text


def data_agent(state: AgentState) -> dict:
    db = _database()
    prompt = (
        f"Schema:\n{db.get_table_info()}\n\n"
        f"Write ONE SQLite SELECT query (no explanation, SQL only) that answers: "
        f"{state['question']}"
    )
    raw = _llm.invoke(prompt).content
    sql = _extract_sql(raw)

    if not sql.lower().startswith("select") or _FORBIDDEN.search(sql):
        result = f"REJECTED (not a read-only SELECT): {sql}"
    else:
        try:
            result = db.run(sql)
        except Exception as e:  # noqa: BLE001
            result = f"SQL error: {e}"

    return {
        "sql_result": f"{sql}\n→ {result}",
        "steps": state.get("steps", []) + ["data(sql)"],
    }
