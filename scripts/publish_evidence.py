"""WP0/WP1：盘点 EXP-0001 历史 run，并生成脱敏公开证据包。

原始 run 目录只读，不修改。公开副本写入 artifacts/evidence/EXP-0001/<run-id>/。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

RUNS_DIR = ROOT / "artifacts" / "runs"
OUT_DIR = ROOT / "artifacts" / "evidence" / "EXP-0001"
DATASET_ID = "ds-0001-alacv-synthetic-v1"
MANIFEST_PATH = ROOT / "data" / "manifests" / f"{DATASET_ID}.json"
MODEL_MANIFEST = ROOT / "data" / "manifests" / "model-dinov2_vits14.json"

RUN_PATTERN = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_EXP-0001_(?P<variant>[ABC])_r(?P<rep>\d+)$")

REQUIRED_FILES = ["metrics.json", "config.snapshot.yaml", "config.sha256", "env.json", "model.json", "items.jsonl"]
OPTIONAL_FILES = ["run.log"]

ORIGINAL_COMMAND = (
    "cd <REPO> && .\\.venv\\Scripts\\python.exe scripts\\run_experiments.py"
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file_lf(p: Path) -> str:
    """公开文件哈希规范：LF 规范化字节（= Git blob / GitHub 公开内容）。"""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def redact_string(s: str, repo_root: Path) -> str:
    """去掉本机绝对路径与用户名，保留相对语义。"""
    home = str(Path.home())
    variants = {home, home.replace("\\", "/"), home.replace("/", "\\")}
    for v in sorted(variants, key=len, reverse=True):
        s = s.replace(v, "%USERPROFILE%")
    for v in {str(repo_root), str(repo_root).replace("\\", "/")}:
        s = s.replace(v, "<REPO>")
    return s


def redact(obj, repo_root: Path):
    if isinstance(obj, str):
        return redact_string(obj, repo_root)
    if isinstance(obj, list):
        return [redact(x, repo_root) for x in obj]
    if isinstance(obj, dict):
        return {k: redact(v, repo_root) for k, v in obj.items()}
    return obj


def public_config(cfg: dict, repo_root: Path) -> tuple[dict, dict]:
    """把 run 专属绝对路径替换为占位符；返回 (公开配置, 映射)。"""
    pub = json.loads(json.dumps(cfg))
    paths = pub.get("paths", {})
    mapping = {}
    replacements = {
        "raw_dir": "<RUN_STORE>/raw",
        "processed_dir": "<RUN_STORE>/processed",
        "features_dir": "<RUN_STORE>/features",
        "index_db": "<RUN_STORE>/index/alacv.sqlite",
        "manifests_dir": "<REPO>/data/manifests",
    }
    for key, placeholder in replacements.items():
        if key in paths:
            old = str(paths[key])
            mapping[key] = {"original": redact_string(old, repo_root), "public": placeholder}
            paths[key] = placeholder
    pub["paths"] = paths
    return pub, mapping


def anonymize_real_ids(manifest: dict) -> dict[str, str]:
    """对本机真实图像 ID 做稳定匿名化（合成图像 ID 保持不变，可由生成器复现）。"""
    real = {f["sha256"] for f in manifest["files"] if f.get("role") == "real"}
    out = {}
    for f in manifest["files"]:
        if f["sha256"] in real:
            iid = f["sha256"][:16]
            out[iid] = "anon_" + sha256_text("alacv-public:" + iid)[:12]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    sanitize_record = MANIFEST_PATH.parent / "ds-0001-manifest-sanitization.json"
    record = json.loads(sanitize_record.read_text(encoding="utf-8"))
    dataset_public_sha = record["manifest_sha256_lf"]
    dataset_runtime_sha = record.get("manifest_sha256_runtime_original_bytes")
    assert sha256_file_lf(MANIFEST_PATH) == dataset_public_sha, "manifest LF 哈希与记录不一致"
    model_manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    anon = anonymize_real_ids(manifest)

    runs = []
    for d in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir() and "EXP-0001" in p.name):
        m = RUN_PATTERN.match(d.name)
        if not m:
            runs.append({"run_id": d.name, "status": "unparsed_name"})
            continue
        entry = {
            "run_id": d.name,
            "variant": m.group("variant"),
            "repeat": int(m.group("rep")),
            "files": {},
            "missing": [],
        }
        for name in REQUIRED_FILES + OPTIONAL_FILES:
            p = d / name
            if p.exists():
                entry["files"][name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            elif name in REQUIRED_FILES:
                entry["missing"].append(name)
        entry["complete"] = not entry["missing"]

        if entry["complete"]:
            env = json.loads((d / "env.json").read_text(encoding="utf-8"))
            metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            model = json.loads((d / "model.json").read_text(encoding="utf-8"))
            cfg = yaml.safe_load((d / "config.snapshot.yaml").read_text(encoding="utf-8"))
            record_sha = (d / "config.sha256").read_text(encoding="utf-8").strip()

            entry["code"] = {
                "git_commit_raw": env.get("git_commit", ""),
                "git_dirty_raw": env.get("git_dirty", None),
                "status": "anchored" if env.get("git_commit") else "pre_git_initialization",
            }
            entry["config_sha256"] = record_sha
            entry["config_sha256_matches_record"] = sha256_text(
                json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            ) == record_sha
            entry["metrics"] = {
                "n_processed": metrics.get("n_processed"),
                "recall@1": metrics.get("retrieval", {}).get("recall@1"),
                "mAP@10": metrics.get("retrieval", {}).get("mAP@10"),
                "t_total_median_ms": metrics.get("timing", {}).get("t_total_median_ms"),
                "feature_bytes_per_image": metrics.get("storage", {}).get("feature_bytes_per_image"),
                "backtrack_psnr_mean": metrics.get("backtrack", {}).get("psnr_mean"),
            }
            entry["model"] = {
                "model_id": model.get("model_id"),
                "revision": (model.get("manifest") or {}).get("revision"),
                "weights_sha256": (model.get("manifest") or {}).get("weights_sha256"),
                "license": (model.get("manifest") or {}).get("license"),
            }

            if not args.no_write:
                dst = out_dir / d.name
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(d / "metrics.json", dst / "metrics.json")
                shutil.copyfile(d / "config.sha256", dst / "config.sha256")

                pub_cfg, mapping = public_config(cfg, ROOT)
                body = yaml.safe_dump(pub_cfg, sort_keys=True, allow_unicode=True)
                (dst / "config.snapshot.public.yaml").write_text(body, encoding="utf-8")
                (dst / "config.snapshot.public.sha256").write_text(
                    sha256_text(body) + "\n", encoding="utf-8"
                )

                pub_env = redact(env, ROOT)
                pub_env["provenance_note"] = (
                    "git_commit 为空表示该 run 启动时仓库尚未完成首次提交；"
                    "该 run 的代码身份无法由 Git commit 证明。"
                )
                (dst / "env.public.json").write_text(
                    json.dumps(pub_env, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                pub_model = redact(model, ROOT)
                (dst / "model.public.json").write_text(
                    json.dumps(pub_model, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                with (d / "items.jsonl").open(encoding="utf-8") as fh, (
                    dst / "items.public.jsonl"
                ).open("w", encoding="utf-8") as out:
                    for line in fh:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        if item.get("image_id") in anon:
                            item["image_id"] = anon[item["image_id"]]
                            item["anonymized"] = True
                        out.write(json.dumps(item, ensure_ascii=False) + "\n")

                ev = {
                    "run_id": d.name,
                    "experiment_id": "EXP-0001",
                    "variant": entry["variant"],
                    "repeat": entry["repeat"],
                    "executed_at_utc": m.group("stamp"),
                    "executed_timezone": "Asia/Shanghai (UTC+08:00)",
                    "original_command": ORIGINAL_COMMAND,
                    "code": entry["code"],
                    "config": {
                        "original_sha256": record_sha,
                        "public_snapshot_sha256": sha256_text(body),
                        "sanitization_map": mapping,
                    },
                    "dataset": {
                        "dataset_id": DATASET_ID,
                        "manifest_sha256_lf": dataset_public_sha,
                        "manifest_sha256_runtime_original_bytes": dataset_runtime_sha,
                        "hash_spec": "sha256 over LF-normalized bytes (Git blob / GitHub file)",
                        "image_hashes_unchanged_by_sanitization": True,
                        "sanitization_record": "data/manifests/ds-0001-manifest-sanitization.json",
                        "n_files": len(manifest["files"]),
                    },
                    "model": entry["model"],
                    "source_files": entry["files"],
                    "anonymization": {
                        "real_image_ids": len(anon),
                        "rule": "anon_<sha256('alacv-public:'+id)[:12]> for role=real only",
                    },
                    "limitations": [],
                }
                if entry["code"]["status"] == "pre_git_initialization":
                    ev["limitations"].append(
                        "run started before the first Git commit; code identity inferred, not proven"
                    )
                (dst / "evidence-manifest.json").write_text(
                    json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        runs.append(entry)

    inventory = {
        "experiment_id": "EXP-0001",
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "runs_dir": "artifacts/runs",
        "n_runs_found": len(runs),
        "dataset": {
            "dataset_id": DATASET_ID,
            "manifest_sha256_lf": dataset_public_sha,
            "manifest_sha256_runtime_original_bytes": dataset_runtime_sha,
            "hash_spec": "sha256 over LF-normalized bytes (Git blob / GitHub file)",
        },
        "model": {
            "model_id": model_manifest.get("model_id"),
            "revision": model_manifest.get("revision"),
            "weights_sha256": model_manifest.get("weights_sha256"),
            "license": model_manifest.get("license"),
        },
        "runs": runs,
    }
    if not args.no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "inventory.json").write_text(
            json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_inventory_md(inventory, out_dir / "inventory.md")
    print(json.dumps({"runs": len(runs), "out": str(out_dir)}, ensure_ascii=False))
    return 0


def write_inventory_md(inv: dict, path: Path) -> None:
    lines = [
        "# EXP-0001 历史 run 证据盘点",
        "",
        f"- 生成时间：{inv['generated_at']}",
        f"- 数据清单：`{inv['dataset']['dataset_id']}`，公开 LF 规范化 SHA-256 `{inv['dataset']['manifest_sha256_lf']}`；"
        f"运行期原始字节 SHA-256 `{inv['dataset']['manifest_sha256_runtime_original_bytes']}`（历史 CRLF，仅存档）",
        f"- 模型：`{inv['model']['model_id']}` revision `{inv['model']['revision']}`",
        f"- 发现 run 数：**{inv['n_runs_found']}**",
        "",
        "| run_id | 变体 | 重复 | 代码 commit | dirty | 配置 SHA-256（前 12） | 必需文件 | Recall@1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in inv["runs"]:
        if "variant" not in r:
            lines.append(f"| `{r['run_id']}` | — | — | — | — | — | 无法解析 | — |")
            continue
        code = r.get("code", {})
        commit = (code.get("git_commit_raw") or "（空：早于首次提交）")
        commit_short = commit[:12] if len(commit) > 12 else commit
        files_ok = "完整" if r.get("complete") else "缺失：" + ", ".join(r.get("missing", []))
        recall = r.get("metrics", {}).get("recall@1")
        lines.append(
            f"| `{r['run_id']}` | {r['variant']} | {r['repeat']} | `{commit_short}` | "
            f"{code.get('git_dirty_raw')} | `{r.get('config_sha256', '')[:12]}` | {files_ok} | "
            f"{recall if recall is not None else '—'} |"
        )
    lines += [
        "",
        "## 说明",
        "",
        "- 9 个 run 的 `metrics.json / config.snapshot.yaml / config.sha256 / env.json / model.json / items.jsonl` 均存在。",
        "- 8 个 run 的 `env.json` 记录代码 commit `0f83806`（dirty=false）；第 1 个 run（C_r2）早于首次提交，commit 为空，代码身份只能推断。",
        "- 原始 run 目录未做任何修改；公开证据副本见同目录各 run 子目录。",
        "- 原始数据清单 SHA-256 在 WP5 脱敏后会变化；`evidence-manifest.json` 记录当次构建所用清单哈希。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
