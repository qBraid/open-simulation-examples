"""Write the verdict table and stamp from results/benchmark.json into README.md."""
import json
import re
import sys

b = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/benchmark.json"))
readme = sys.argv[2] if len(sys.argv) > 2 else "README.md"
rows = ["| Check | Result | Verdict | Notes |", "|---|---|---|---|"]
for r in b["rows"]:
    rows.append(f"| {r['label']} | {r['value']} | {('**' + r['verdict'] + '**') if r.get('verdict') else '—'} | {r.get('note', '')} |")
table = "\n".join(rows)
stamp = "## Verification stamp\n```\n" + b["stamp"] + "\n```"
s = open(readme).read()
s = re.sub(r"<!--BENCH-->.*?<!--/BENCH-->|<!--BENCH-->", "<!--BENCH-->\n" + table + "\n<!--/BENCH-->", s, flags=re.S)
s = re.sub(r"<!--STAMP-->.*?<!--/STAMP-->|<!--STAMP-->", "<!--STAMP-->\n" + stamp + "\n<!--/STAMP-->", s, flags=re.S)
open(readme, "w").write(s)
print("README updated:", len(b["rows"]), "rows")
