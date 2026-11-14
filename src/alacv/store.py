"""视觉知识库：SQLite 元信息 + SafeTensors 特征。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS models (
  model_id      TEXT PRIMARY KEY,
  display_name  TEXT NOT NULL,
  arch          TEXT NOT NULL,
  params        INTEGER,
  source        TEXT NOT NULL,
  revision      TEXT NOT NULL,
  weights_sha256 TEXT NOT NULL,
  license       TEXT NOT NULL,
  license_file  TEXT,
  input_size    INTEGER NOT NULL,
  patch_size    INTEGER NOT NULL,
  cls_dim       INTEGER NOT NULL,
  norm_mean    TEXT NOT NULL,
  norm_std     TEXT NOT NULL,
  created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
  image_id      TEXT PRIMARY KEY,
  sha256        TEXT NOT NULL UNIQUE,
  rel_path      TEXT NOT NULL,
  source_name   TEXT,
  ext           TEXT NOT NULL,
  mime          TEXT NOT NULL,
  width         INTEGER NOT NULL,
  height        INTEGER NOT NULL,
  channels      INTEGER NOT NULL,
  pil_mode      TEXT NOT NULL,
  color_space   TEXT NOT NULL,
  icc_sha256    TEXT,
  file_size     INTEGER NOT NULL,
  exif_json     TEXT,
  dataset_id    TEXT,
  status        TEXT NOT NULL DEFAULT 'ok',
  reject_reason TEXT,
  ingested_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (
  asset_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  image_id      TEXT NOT NULL REFERENCES images(image_id),
  kind          TEXT NOT NULL,
  level         INTEGER,
  label         TEXT,
  rel_path      TEXT NOT NULL,
  width         INTEGER NOT NULL,
  height        INTEGER NOT NULL,
  format        TEXT NOT NULL,
  bytes         INTEGER NOT NULL,
  sha256        TEXT NOT NULL,
  config_sha256 TEXT NOT NULL,
  created_at    TEXT NOT NULL,
  UNIQUE (image_id, kind, label, config_sha256)
);

CREATE TABLE IF NOT EXISTS features (
  feature_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  image_id      TEXT NOT NULL REFERENCES images(image_id),
  variant       TEXT,
  source        TEXT NOT NULL,
  level         INTEGER,
  model_id      TEXT NOT NULL REFERENCES models(model_id),
  precision     TEXT NOT NULL,
  n_patches     INTEGER NOT NULL,
  patch_grid    INTEGER NOT NULL,
  cls_dim       INTEGER NOT NULL,
  rel_path      TEXT NOT NULL,
  bytes         INTEGER NOT NULL,
  sha256        TEXT NOT NULL,
  config_sha256 TEXT NOT NULL,
  created_at    TEXT NOT NULL,
  UNIQUE (image_id, source, config_sha256)
);

CREATE TABLE IF NOT EXISTS rois (
  roi_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  image_id      TEXT NOT NULL REFERENCES images(image_id),
  roi_index     INTEGER NOT NULL,
  x INTEGER NOT NULL, y INTEGER NOT NULL,
  w INTEGER NOT NULL, h INTEGER NOT NULL,
  score         REAL NOT NULL,
  method        TEXT NOT NULL,
  crop_asset_id INTEGER REFERENCES assets(asset_id),
  created_at    TEXT NOT NULL,
  UNIQUE (image_id, roi_index, method)
);

CREATE TABLE IF NOT EXISTS runs (
  run_id        TEXT PRIMARY KEY,
  exp_id        TEXT,
  variant       TEXT,
  config_sha256 TEXT NOT NULL,
  git_commit    TEXT,
  git_dirty     INTEGER,
  host          TEXT,
  started_at    TEXT NOT NULL,
  finished_at   TEXT,
  status        TEXT NOT NULL,
  metrics_json  TEXT,
  notes         TEXT
);

CREATE TABLE IF NOT EXISTS run_items (
  run_id      TEXT NOT NULL REFERENCES runs(run_id),
  image_id    TEXT NOT NULL,
  variant     TEXT NOT NULL,
  t_ingest_ms REAL, t_pyramid_ms REAL, t_fovea_ms REAL, t_encode_ms REAL,
  t_total_ms  REAL,
  peak_vram_mb REAL,
  rss_delta_mb REAL,
  feature_bytes INTEGER,
  PRIMARY KEY (run_id, image_id)
);

CREATE INDEX IF NOT EXISTS idx_features_lookup ON features(image_id, source);
CREATE INDEX IF NOT EXISTS idx_assets_lookup   ON assets(image_id, kind);
CREATE INDEX IF NOT EXISTS idx_rois_image      ON rois(image_id);
"""


@dataclass
class ImageRecord:
    image_id: str
    sha256: str
    rel_path: str
    width: int
    height: int


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_source(source: str) -> str:
    return source.replace(":", "-").replace("/", "_")


class VisualStore:
    def __init__(self, db_path: str | Path, features_dir: str | Path):
        self.db_path = Path(db_path)
        self.features_dir = Path(features_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ models
    def upsert_model(self, rec: dict) -> str:
        self.conn.execute(
            """INSERT INTO models (model_id, display_name, arch, params, source, revision,
                weights_sha256, license, license_file, input_size, patch_size, cls_dim,
                norm_mean, norm_std, created_at)
               VALUES (:model_id,:display_name,:arch,:params,:source,:revision,
                :weights_sha256,:license,:license_file,:input_size,:patch_size,:cls_dim,
                :norm_mean,:norm_std,:created_at)
               ON CONFLICT(model_id) DO UPDATE SET
                weights_sha256=excluded.weights_sha256,
                revision=excluded.revision,
                params=excluded.params""",
            rec,
        )
        self.conn.commit()
        return rec["model_id"]

    # ------------------------------------------------------------------ images
    def add_image(self, rec: dict) -> tuple[str, bool]:
        cur = self.conn.execute("SELECT image_id FROM images WHERE sha256=?", (rec["sha256"],))
        row = cur.fetchone()
        if row:
            return row["image_id"], False
        self.conn.execute(
            """INSERT INTO images (image_id, sha256, rel_path, source_name, ext, mime,
                 width, height, channels, pil_mode, color_space, icc_sha256, file_size,
                 exif_json, dataset_id, status, reject_reason, ingested_at)
               VALUES (:image_id,:sha256,:rel_path,:source_name,:ext,:mime,
                 :width,:height,:channels,:pil_mode,:color_space,:icc_sha256,:file_size,
                 :exif_json,:dataset_id,:status,:reject_reason,:ingested_at)""",
            rec,
        )
        self.conn.commit()
        return rec["image_id"], True

    def get_image(self, image_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM images WHERE image_id=?", (image_id,)).fetchone()

    def image_by_sha(self, sha: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM images WHERE sha256=?", (sha,)).fetchone()

    def list_images(self, dataset_id: str | None = None) -> list[sqlite3.Row]:
        if dataset_id:
            return self.conn.execute(
                "SELECT * FROM images WHERE status='ok' AND dataset_id=? ORDER BY image_id", (dataset_id,)
            ).fetchall()
        return self.conn.execute("SELECT * FROM images WHERE status='ok' ORDER BY image_id").fetchall()

    # ------------------------------------------------------------------ assets
    def add_asset(self, rec: dict) -> int:
        self.conn.execute(
            """INSERT INTO assets (image_id, kind, level, label, rel_path, width, height,
                 format, bytes, sha256, config_sha256, created_at)
               VALUES (:image_id,:kind,:level,:label,:rel_path,:width,:height,
                 :format,:bytes,:sha256,:config_sha256,:created_at)
               ON CONFLICT(image_id, kind, label, config_sha256) DO UPDATE SET
                 bytes=excluded.bytes, sha256=excluded.sha256, rel_path=excluded.rel_path""",
            rec,
        )
        row = self.conn.execute(
            "SELECT asset_id FROM assets WHERE image_id=? AND kind=? AND label=? AND config_sha256=?",
            (rec["image_id"], rec["kind"], rec["label"], rec["config_sha256"]),
        ).fetchone()
        self.conn.commit()
        if row is None:
            raise RuntimeError("asset 写入后无法回读 asset_id")
        return int(row["asset_id"])

    def assets(self, image_id: str, kind: str | None = None) -> list[sqlite3.Row]:
        if kind:
            return self.conn.execute(
                "SELECT * FROM assets WHERE image_id=? AND kind=? ORDER BY level, label", (image_id, kind)
            ).fetchall()
        return self.conn.execute("SELECT * FROM assets WHERE image_id=? ORDER BY kind, level, label", (image_id,)).fetchall()

    # ------------------------------------------------------------------ features
    def feature_path(self, image_id: str, source: str) -> Path:
        return self.features_dir / image_id / f"{safe_source(source)}.safetensors"

    def has_feature(self, image_id: str, source: str, config_sha256: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM features WHERE image_id=? AND source=? AND config_sha256=?",
            (image_id, source, config_sha256),
        ).fetchone()
        return row is not None

    def add_feature(self, rec: dict) -> bool:
        if self.has_feature(rec["image_id"], rec["source"], rec["config_sha256"]):
            return False
        self.conn.execute(
            """INSERT INTO features (image_id, variant, source, level, model_id, precision,
                 n_patches, patch_grid, cls_dim, rel_path, bytes, sha256, config_sha256, created_at)
               VALUES (:image_id,:variant,:source,:level,:model_id,:precision,
                 :n_patches,:patch_grid,:cls_dim,:rel_path,:bytes,:sha256,:config_sha256,:created_at)""",
            rec,
        )
        self.conn.commit()
        return True

    def write_features(self, image_id: str, source: str, cls: np.ndarray, patch: np.ndarray, meta: dict) -> dict:
        """把 CLS + Patch 写为 SafeTensors，返回入库所需的记录片段。"""
        from safetensors.numpy import save_file

        path = self.feature_path(image_id, source)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_bytes = np.frombuffer(json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8"), dtype=np.uint8)
        arrays = {
            "cls": np.ascontiguousarray(cls, dtype=np.float16),
            "patch": np.ascontiguousarray(patch, dtype=np.float16),
            "grid_h": np.array([meta.get("grid", [0, 0])[0]], dtype=np.int32),
            "grid_w": np.array([meta.get("grid", [0, 0])[1]], dtype=np.int32),
            "meta_json": meta_bytes,
        }
        tmp = path.with_suffix(".safetensors.tmp")
        save_file(arrays, str(tmp))
        tmp.replace(path)
        return {
            "rel_path": str(path.relative_to(self.features_dir.parent)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "n_patches": int(patch.shape[0]),
            "patch_grid": int(meta.get("grid", [0, 0])[0]),
            "cls_dim": int(cls.shape[0]),
        }

    def load_feature(self, image_id: str, source: str) -> dict:
        from safetensors.numpy import load_file

        path = self.feature_path(image_id, source)
        if not path.exists():
            raise FileNotFoundError(f"特征文件不存在: {path}")
        data = load_file(str(path))
        meta = {}
        if "meta_json" in data:
            meta = json.loads(bytes(data["meta_json"]).decode("utf-8"))
        return {
            "cls": data["cls"].astype(np.float32),
            "patch": data["patch"].astype(np.float32),
            "grid": (int(data["grid_h"][0]), int(data["grid_w"][0])),
            "meta": meta,
            "path": path,
        }

    def load_cls(self, image_id: str, source: str) -> np.ndarray:
        return self.load_feature(image_id, source)["cls"]

    # ------------------------------------------------------------------ rois
    def add_roi(self, rec: dict) -> int:
        cur = self.conn.execute(
            """INSERT INTO rois (image_id, roi_index, x, y, w, h, score, method, crop_asset_id, created_at)
               VALUES (:image_id,:roi_index,:x,:y,:w,:h,:score,:method,:crop_asset_id,:created_at)
               ON CONFLICT(image_id, roi_index, method) DO UPDATE SET
                 x=excluded.x, y=excluded.y, w=excluded.w, h=excluded.h,
                 score=excluded.score, crop_asset_id=excluded.crop_asset_id""",
            rec,
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def rois_of(self, image_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM rois WHERE image_id=? ORDER BY roi_index", (image_id,)
        ).fetchall()

    # ------------------------------------------------------------------ runs
    def start_run(self, rec: dict) -> None:
        self.conn.execute(
            """INSERT INTO runs (run_id, exp_id, variant, config_sha256, git_commit, git_dirty,
                 host, started_at, status, notes)
               VALUES (:run_id,:exp_id,:variant,:config_sha256,:git_commit,:git_dirty,
                 :host,:started_at,'running',:notes)""",
            rec,
        )
        self.conn.commit()

    def finish_run(self, run_id: str, status: str, metrics_json: str | None, finished_at: str) -> None:
        self.conn.execute(
            "UPDATE runs SET status=?, metrics_json=?, finished_at=? WHERE run_id=?",
            (status, metrics_json, finished_at, run_id),
        )
        self.conn.commit()

    def add_run_item(self, rec: dict) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO run_items (run_id, image_id, variant, t_ingest_ms,
                 t_pyramid_ms, t_fovea_ms, t_encode_ms, t_total_ms, peak_vram_mb,
                 rss_delta_mb, feature_bytes)
               VALUES (:run_id,:image_id,:variant,:t_ingest_ms,:t_pyramid_ms,:t_fovea_ms,
                 :t_encode_ms,:t_total_ms,:peak_vram_mb,:rss_delta_mb,:feature_bytes)""",
            rec,
        )
        self.conn.commit()

    # ------------------------------------------------------------------ descriptor
    def descriptor(self, image_id: str, sources: list[str]) -> np.ndarray:
        """等权 L2 归一化拼接描述子。"""
        vecs = []
        for src in sources:
            v = self.load_cls(image_id, src).astype(np.float64)
            n = np.linalg.norm(v)
            if n <= 0:
                raise ValueError(f"{image_id}/{src} 的 CLS 向量范数为 0")
            vecs.append(v / n)
        out = np.concatenate(vecs) if len(vecs) > 1 else vecs[0]
        return (out / np.linalg.norm(out)).astype(np.float32)

    def stats(self) -> dict:
        q = {
            "images": "SELECT COUNT(*) FROM images WHERE status='ok'",
            "rejected": "SELECT COUNT(*) FROM images WHERE status!='ok'",
            "features": "SELECT COUNT(*) FROM features",
            "feature_bytes": "SELECT COALESCE(SUM(bytes),0) FROM features",
            "assets": "SELECT COUNT(*) FROM assets",
            "asset_bytes": "SELECT COALESCE(SUM(bytes),0) FROM assets",
            "rois": "SELECT COUNT(*) FROM rois",
            "runs": "SELECT COUNT(*) FROM runs",
        }
        out = {k: self.conn.execute(s).fetchone()[0] for k, s in q.items()}
        out["db_bytes"] = self.db_path.stat().st_size if self.db_path.exists() else 0
        return out
