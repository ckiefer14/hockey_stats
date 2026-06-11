# scripts/generate_schema_doc.py
import sys
import os
from datetime import datetime
sys.path.insert(0, ".")

from src.data.db import get_connection


def generate_schema_doc(out_path: str = "docs/schema.md") -> None:
    conn = get_connection()

    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]

    lines = []
    lines.append("# Database Schema")
    lines.append("")
    lines.append(f"_Auto-generated from the live database on {datetime.now():%Y-%m-%d %H:%M}._")
    lines.append("_Regenerate with `python scripts/generate_schema_doc.py`._")
    lines.append("")
    lines.append(f"**{len(tables)} tables.**")
    lines.append("")

    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        lines.append(f"## `{table}`")
        lines.append("")
        lines.append(f"{count:,} rows.")
        lines.append("")
        lines.append("| Column | Type | Primary Key | Not Null |")
        lines.append("|---|---|---|---|")

        for cid, name, col_type, notnull, default, pk in conn.execute(
            f"PRAGMA table_info({table})"
        ).fetchall():
            pk_mark = "✓" if pk else ""
            nn_mark = "✓" if notnull else ""
            lines.append(f"| `{name}` | {col_type} | {pk_mark} | {nn_mark} |")

        lines.append("")

        fks = conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        if fks:
            lines.append("**Foreign keys:**")
            lines.append("")
            for fk in fks:
                lines.append(f"- `{fk[3]}` → `{fk[2]}.{fk[4]}`")
            lines.append("")

    conn.close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Schema doc written to {out_path}")


if __name__ == "__main__":
    generate_schema_doc()