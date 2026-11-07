#!/usr/bin/env python3
"""Check the report page (bench/results/report/index.html) against the data it binds to.

Run after `python3 bench/make_report_data.py`:

    python3 bench/check_report.py [--report DIR]

Checks:
  - every `data-source` id on the page is a JSON file the generator wrote (DIR/<id>.json) or a
    `dash.calc` of the page's own script;
  - every `data-field` names a column of that source's rows (an element without its own
    `data-source` takes the nearest enclosing one, as a table's `<th>` does);
  - every `data-where` (`col=value&col=value`, percent-encoded) and `data-row` selects one row, and a
    single value's field is not empty there (a note instead inside a `data-when` element, which the
    script shows only when its data is there);
  - every `data-row-key` names columns whose values tell the rows apart;
  - every `data-draw` has a `dash.draw` in the script, and the other way round;
  - the script parses (`node --check`) and its top level runs with stand-ins for the dashboard's
    `dash` and `d3`, which also yields the rows of every `dash.calc` for the checks above.

Without node only the written sources are checked, and the calcs' rows are reported as unchecked.
Exit status 1 if anything fails.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class Page(HTMLParser):
    """Elements with data-* attributes (with the data-source they inherit) and the scripts."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, dict]] = []
        self.marks: list[dict] = []
        self.scripts: list[str] = []
        self._script: list[str] | None = None

    def _open(self, tag: str, attrs: list, push: bool) -> None:
        a = {k: (v if v is not None else "") for k, v in attrs}
        if any(k.startswith("data-") for k in a):
            inherited = next((s["data-source"] for _, s in reversed(self.stack) if "data-source" in s), None)
            # inside an element the script shows only when its data is there (data-when)
            conditional = "data-when" in a or any("data-when" in s for _, s in self.stack)
            self.marks.append({"line": self.getpos()[0], "tag": tag, "attrs": a, "inherited": inherited, "conditional": conditional})
        if tag == "script":
            self._script = []
        if push and tag not in VOID:
            self.stack.append((tag, a))

    def handle_starttag(self, tag, attrs):
        self._open(tag, attrs, True)

    def handle_startendtag(self, tag, attrs):
        self._open(tag, attrs, False)

    def handle_endtag(self, tag):
        if tag == "script" and self._script is not None:
            self.scripts.append("".join(self._script))
            self._script = None
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        if self._script is not None:
            self._script.append(data)


# Stand-ins for the dashboard runtime: enough for the page's top level and its dash.calc functions.
HARNESS = r"""
const fs = require('fs');
const [, , scriptPath, dataDir, outPath] = process.argv;
const written = new Set(fs.readdirSync(dataDir).filter((f) => f.endsWith('.json')).map((f) => f.slice(0, -5)));
const calcs = new Map(), draws = new Set(), cache = new Map();
const rowsOf = (id, seen = new Set()) => {
  if (cache.has(id)) return cache.get(id);
  let out;
  if (written.has(id)) out = { ok: true, rows: JSON.parse(fs.readFileSync(`${dataDir}/${id}.json`, 'utf8')) };
  else if (calcs.has(id)) {
    if (seen.has(id)) return { ok: false, error: `calc cycle through ${id}` };
    seen.add(id);
    const c = calcs.get(id);
    const ins = (c.inputs || []).map((i) => rowsOf(i, seen));
    const bad = ins.find((r) => !r.ok);
    if (bad) out = { ok: false, error: `input: ${bad.error}` };
    else {
      try { out = { ok: true, rows: c.fn(...ins.map((r) => r.rows)) }; } catch (e) { out = { ok: false, error: String(e && e.stack || e) }; }
    }
  } else out = { ok: false, error: `no dataset or calc named ${id}` };
  cache.set(id, out);
  return out;
};
globalThis.dash = {
  colors: ['#000', '#111', '#222', '#333', '#444', '#555', '#666', '#777'],
  calc: (id, spec) => { calcs.set(id, spec); },
  draw: (name) => { draws.add(name); },
  loader: () => {},
  onData: () => {},
  data: (id) => { const r = rowsOf(id); return r.ok ? { status: 'ok', data: r.rows, refreshing: false, meta: [] } : { status: 'error', data: [], message: r.error }; },
  params: () => ({}), setParams: () => {}, resetParams: () => {}, setLink: () => {}, refresh: () => {},
};
const num = (a, f) => (f ? a.map(f) : a).filter((v) => v != null && !Number.isNaN(+v)).map(Number);
const d3base = {
  max: (a, f) => { const v = num(a, f); return v.length ? Math.max(...v) : undefined; },
  min: (a, f) => { const v = num(a, f); return v.length ? Math.min(...v) : undefined; },
  sum: (a, f) => num(a, f).reduce((s, v) => s + v, 0),
  mean: (a, f) => { const v = num(a, f); return v.length ? v.reduce((s, x) => s + x, 0) / v.length : undefined; },
  extent: (a, f) => { const v = num(a, f); return v.length ? [Math.min(...v), Math.max(...v)] : [undefined, undefined]; },
  range: (a, b, s = 1) => { if (b === undefined) { b = a; a = 0; } const o = []; for (let x = a; s > 0 ? x < b : x > b; x += s) o.push(x); return o; },
  format: () => (v) => String(v),
  group: (a, f) => { const m = new Map(); for (const x of a) { const k = f(x); if (!m.has(k)) m.set(k, []); m.get(k).push(x); } return m; },
};
globalThis.d3 = new Proxy(d3base, { get: (t, k) => (k in t ? t[k] : () => { throw new Error(`d3.${String(k)} is not in the check's stand-in`); }) });
globalThis.window = globalThis;
globalThis.document = { querySelector: () => null, querySelectorAll: () => [], getElementById: () => null, createElement: () => ({}) };
globalThis.location = { hash: '' };
globalThis.requestAnimationFrame = () => 0;
const out = { top_level_error: null, draws: [], calcs: {} };
try { require('vm').runInThisContext(fs.readFileSync(scriptPath, 'utf8'), { filename: 'index.html <script>' }); }
catch (e) { out.top_level_error = String(e && e.stack || e); }
out.draws = [...draws];
for (const id of calcs.keys()) {
  const r = rowsOf(id);
  const inputs = calcs.get(id).inputs || [];
  out.calcs[id] = r.ok ? { rows: Array.isArray(r.rows) ? r.rows : null, value_type: typeof r.rows, inputs } : { error: r.error, inputs };
}
fs.writeFileSync(outPath, JSON.stringify(out));
"""


def js_str(v) -> str:
    """A cell value as the page compares it (JavaScript's String())."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def parse_where(text: str) -> list[tuple[str, str]]:
    out = []
    for part in text.split("&"):
        k, _, v = part.partition("=")
        out.append((urllib.parse.unquote(k), urllib.parse.unquote(v)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--report", type=Path, default=ROOT / "bench/results/report")
    args = ap.parse_args()
    page_path = args.report / "index.html"
    page = Page()
    page.feed(page_path.read_text())
    script = "\n".join(page.scripts)

    written: dict[str, list] = {}
    for f in sorted(args.report.glob("*.json")):
        written[f.stem] = json.loads(f.read_text())

    fails: list[str] = []
    notes: list[str] = []

    # the script: parse, then run its top level and its calcs with stand-ins
    calc_rows: dict[str, list | None] = {}
    calc_ids: set[str] = set()
    calc_inputs: set[str] = set()
    registered: set[str] | None = None
    node = shutil.which("node")
    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "page.js"
        sp.write_text(script)
        if node:
            r = subprocess.run([node, "--check", str(sp)], capture_output=True, text=True)
            if r.returncode:
                fails.append(f"script does not parse (node --check):\n{r.stderr.strip()}")
            else:
                hp, op = Path(tmp) / "harness.js", Path(tmp) / "out.json"
                hp.write_text(HARNESS)
                r = subprocess.run([node, str(hp), str(sp), str(args.report), str(op)], capture_output=True, text=True)
                if r.returncode or not op.exists():
                    fails.append(f"script harness failed:\n{(r.stderr or r.stdout).strip()}")
                else:
                    res = json.loads(op.read_text())
                    if res["top_level_error"]:
                        fails.append(f"script's top level throws: {res['top_level_error'].splitlines()[0]}")
                    registered = set(res["draws"])
                    for cid, c in res["calcs"].items():
                        calc_ids.add(cid)
                        calc_inputs.update(c.get("inputs", []))
                        if "error" in c:
                            fails.append(f"calc {cid} fails: {c['error'].splitlines()[0]}")
                            calc_rows[cid] = None
                        elif c["rows"] is None:
                            fails.append(f"calc {cid} returns a {c['value_type']}, not rows")
                            calc_rows[cid] = None
                        else:
                            calc_rows[cid] = c["rows"]
        else:
            notes.append("node not found: the script was not parsed and the calcs' rows were not checked")
            import re
            calc_ids = set(re.findall(r"dash\.calc\(\s*['\"]([A-Za-z0-9_-]+)['\"]", script))
            calc_rows = {c: None for c in calc_ids}

    clash = sorted(calc_ids & set(written))
    if clash:
        fails.append(f"calc ids that are also written datasets: {', '.join(clash)}")

    def rows_of(sid: str):
        if sid in written:
            return written[sid]
        return calc_rows.get(sid)

    counts = {"marks": 0, "sources": set(), "fields": 0, "wheres": 0, "keys": 0}
    drawn: set[str] = set()
    for m in page.marks:
        a, where = m["attrs"], f"line {m['line']} <{m['tag']}>"
        if "data-draw" in a:
            drawn.add(a["data-draw"])
        own = a.get("data-source")
        if not any(k in a for k in ("data-source", "data-field", "data-where", "data-row", "data-row-key")):
            continue
        counts["marks"] += 1
        ids = (own or m["inherited"] or "").split()
        if own is not None:
            for sid in ids:
                counts["sources"].add(sid)
                if sid not in written and sid not in calc_ids:
                    fails.append(f"{where}: data-source {sid!r} is neither a written dataset nor a dash.calc")
        if not ids:
            fails.append(f"{where}: {', '.join(k for k in a if k.startswith('data-'))} without a data-source here or above")
            continue
        if not any(k in a for k in ("data-field", "data-where", "data-row", "data-row-key")):
            continue
        if len(ids) > 1:
            fails.append(f"{where}: data-field/where/row-key under several sources ({' '.join(ids)})")
            continue
        sid = ids[0]
        rows = rows_of(sid)
        if rows is None:
            if sid in calc_ids:
                notes.append(f"{where}: rows of calc {sid} unknown, field/where not checked")
            continue
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            fails.append(f"{where}: source {sid} is not a list of row objects")
            continue
        cols = set().union(*(r.keys() for r in rows)) if rows else set()
        sel = rows
        if "data-where" in a:
            counts["wheres"] += 1
            conds = parse_where(a["data-where"])
            for k, _ in conds:
                if k not in cols:
                    fails.append(f"{where}: data-where column {k!r} not in {sid}")
            sel = [r for r in rows if all(js_str(r.get(k)) == v for k, v in conds)]
            if not sel:
                msg = f"{where}: data-where {a['data-where']!r} selects no row of {sid} ({len(rows)} rows)"
                if m["conditional"]:
                    notes.append(msg + " (shown only when its data is there)")
                else:
                    fails.append(msg)
            elif len(sel) > 1:
                fails.append(f"{where}: data-where {a['data-where']!r} selects {len(sel)} rows of {sid}, not one")
        if "data-row" in a:
            i = int(a["data-row"])
            if not 0 <= i < len(rows):
                fails.append(f"{where}: data-row {i} outside {sid} ({len(rows)} rows)")
            else:
                sel = [rows[i]]
        if "data-field" in a:
            counts["fields"] += 1
            f = a["data-field"]
            if not rows:
                fails.append(f"{where}: data-field {f!r}: {sid} has no rows")
            elif f not in cols:
                fails.append(f"{where}: data-field {f!r} is not a column of {sid}")
            elif sel and all(r.get(f) is None for r in sel) and ("data-where" in a or "data-row" in a):
                msg = f"{where}: data-field {f!r} is empty in the row {a.get('data-where') or a.get('data-row')} of {sid}"
                if m["conditional"]:
                    notes.append(msg + " (shown only when its data is there)")
                else:
                    fails.append(msg)
        if "data-row-key" in a:
            counts["keys"] += 1
            keys = a["data-row-key"].split(",")
            missing = [k for k in keys if k not in cols]
            if missing:
                fails.append(f"{where}: data-row-key {', '.join(missing)} not a column of {sid}")
            else:
                vals = [tuple(js_str(r.get(k)) for k in keys) for r in rows]
                dup = len(vals) - len(set(vals))
                if dup:
                    fails.append(f"{where}: data-row-key {a['data-row-key']!r} repeats in {dup} rows of {sid}")
            if not rows:
                fails.append(f"{where}: table source {sid} has no rows")

    if registered is not None:
        for name in sorted(drawn - registered):
            fails.append(f"data-draw {name!r} has no dash.draw in the script")
        for name in sorted(registered - drawn):
            fails.append(f"dash.draw {name!r} draws no element of the page")

    unused = sorted(set(written) - counts["sources"] - calc_inputs)
    unused_calcs = sorted(calc_ids - counts["sources"] - calc_inputs)
    print(f"page: {page_path.relative_to(ROOT) if page_path.is_relative_to(ROOT) else page_path}")
    print(f"  {counts['marks']} marked elements, {len(counts['sources'])} sources ({len(written)} written datasets, "
          f"{len(calc_ids)} calcs), {counts['fields']} fields, {counts['wheres']} where-selections, {counts['keys']} row keys")
    if registered is not None:
        print(f"  script: parses; top level runs; {len(registered)} drawings, {len(drawn)} data-draw names on the page")
    if unused:
        print(f"  written datasets that no element marks and no calc reads: {', '.join(unused)}")
    if unused_calcs:
        print(f"  calcs that no element marks and no calc reads: {', '.join(unused_calcs)}")
    for n in notes:
        print(f"  note: {n}")
    if fails:
        print(f"FAIL: {len(fails)} problem(s)")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
