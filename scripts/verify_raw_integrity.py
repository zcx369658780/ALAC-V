"""校验原始归档不可变性：重算 SHA-256 与清单比对。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alacv.config import load_config  # noqa: E402
from alacv.ingest import verify_raw  # noqa: E402
from alacv.store import VisualStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    store = VisualStore(cfg.path("paths.index_db"), cfg.path("paths.features_dir"))
    res = verify_raw(cfg, store)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
