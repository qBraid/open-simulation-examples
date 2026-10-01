"""Inline results/viewer_data.json into viewer_template.html -> viewer.html (single self-contained page)."""
from pathlib import Path

root = Path(__file__).parent
data = (root / "results" / "viewer_data.json").read_text()
assert "</script" not in data
html = (root / "viewer_template.html").read_text().replace("/*__DATA__*/", data)
(root / "viewer.html").write_text(html)
print(f"viewer.html {len(html) / 1e6:.2f} MB")
