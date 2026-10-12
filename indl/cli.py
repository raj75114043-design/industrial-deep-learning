"""Command line.

    python -m indl.cli dissolution --mgst_pct 1.4 --lube_blend_min 9
    python -m indl.cli fleet
"""
from __future__ import annotations

import argparse
import json
import logging

from indl import service as S


def main() -> None:
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser(prog="indl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dissolution")
    for k, v in S.DEFAULT_BATCH.items():
        d.add_argument(f"--{k}", type=type(v), default=v)
    sub.add_parser("fleet")
    a = ap.parse_args()
    if a.cmd == "dissolution":
        print(json.dumps(S.predict_dissolution({k: getattr(a, k) for k in S.DEFAULT_BATCH}), indent=2))
    else:
        print(S.fleet_status().head(20).to_string(index=False))


if __name__ == "__main__":
    main()
