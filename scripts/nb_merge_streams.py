"""Git clean filter for .ipynb: merge back-to-back print outputs of a cell into one.

Jupyter sends printed text in batches, and the editor stores each batch as its own output.
A cell that prints, computes, then prints again lands as one output or two depending on
timing, so an unchanged notebook shows "outputs changed" after a rerun. Git runs this on
every notebook it stages or compares, so only the merged form is ever stored.

Wired up by .gitattributes (`*.ipynb filter=nbstreams`) plus one git config line per clone:

    git config filter.nbstreams.clean "python3 scripts/nb_merge_streams.py"

Without that line git skips the filter (no error). Reads the notebook on stdin, writes it to
stdout; a notebook with nothing to merge, or that isn't JSON, goes through byte for byte.
"""

import json
import sys


def merge(outputs: list[dict]) -> list[dict]:
    # adjacent stream outputs of the same stream (stdout, stderr) become one
    merged: list[dict] = []
    for out in outputs:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and out.get("output_type") == "stream"
            and prev.get("output_type") == "stream"
            and out.get("name") == prev.get("name")
        ):
            text = "".join(prev["text"]) + "".join(out["text"])
            prev["text"] = text.splitlines(keepends=True)
        else:
            merged.append(out)
    return merged


def main() -> None:
    raw = sys.stdin.buffer.read()
    try:
        nb = json.loads(raw)
    except ValueError:
        sys.stdout.buffer.write(raw)
        return
    changed = False
    for cell in nb.get("cells", []):
        outputs = cell.get("outputs")
        if outputs:
            new = merge(outputs)
            if len(new) != len(outputs):
                cell["outputs"] = new
                changed = True
    if not changed:
        sys.stdout.buffer.write(raw)
        return
    # the editor's own layout: 1-space indent, unicode as is, trailing newline
    sys.stdout.buffer.write((json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode())


if __name__ == "__main__":
    main()
