#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型管理器：通过国内镜像站 hf-mirror.com 搜索并下载语音识别模型。

- 搜索：GET {mirror}/api/models?search=...&sort=downloads&limit=...
- 元信息：GET {mirror}/api/models/{repo_id}  （返回 siblings 文件列表）
- 下载：GET {mirror}/{repo_id}/resolve/main/{filename}（流式，带进度回调）
所有下载模型存放于项目 models/ 目录下，以 repo_id 命名（/ 替换为 __）。
"""
import os
import re
from dataclasses import dataclass, field

import requests

from . import settings
from .config import HF_MIRROR

TIMEOUT = 15


@dataclass
class ModelInfo:
    """搜索结果条目。"""
    repo_id: str
    downloads: int = 0
    likes: int = 0
    pipeline_tag: str = ""
    files: list = field(default_factory=list)

    @property
    def local_dir(self) -> str:
        """对应的本地存放目录（跟随用户设置，而非固定常量）。"""
        safe = self.repo_id.replace("/", "__")
        return os.path.join(settings.get_models_dir(), safe)

    @property
    def is_downloaded(self) -> bool:
        """以 model.bin 是否存在作为“已下载完成”的判据。"""
        return os.path.isfile(os.path.join(self.local_dir, "model.bin"))


def _get(url: str, **kwargs) -> requests.Response:
    resp = requests.get(url, timeout=TIMEOUT, **kwargs)
    resp.raise_for_status()
    return resp


def filter_by_engine(repo_id: str, engine: str) -> bool:
    """按引擎架构过滤模型仓库：
    - cpu/cuda：需要 CTranslate2 格式（faster-whisper / ct2 / ctranslate2），
      排除 mlx 与 openai/whisper 原始权重（原始格式 faster-whisper 无法加载）
    - mlx：需要 MLX 格式（仓库名含 mlx，通常为 mlx-community）
    - None：不过滤
    """
    rid = repo_id.lower()
    if engine in ("cpu", "cuda"):
        if "mlx" in rid or rid.startswith("openai/whisper"):
            return False
        return ("faster-whisper" in rid or "ct2" in rid or "ctranslate2" in rid)
    if engine == "mlx":
        return "mlx" in rid
    return True


def search_models(query: str, limit: int = 30, engine: str = None) -> list:
    """在镜像站搜索模型，返回按下载量排序的 ModelInfo 列表。

    engine 不为 None 时严格过滤为该架构可用的模型格式：
    - mlx：限定 author=mlx-community 检索（MLX 格式模型社区仓库）
    - cpu/cuda：搜索后按 CTranslate2 格式过滤
    """
    url = f"{HF_MIRROR}/api/models"
    params = {"search": query, "limit": limit, "sort": "downloads", "direction": -1}
    if engine == "mlx" and "community" not in query.lower():
        params["author"] = "mlx-community"
    data = _get(url, params=params).json()
    results = []
    for item in data:
        repo_id = item.get("id") or item.get("modelId") or ""
        if not repo_id:
            continue
        info = ModelInfo(
            repo_id=repo_id,
            downloads=item.get("downloads", 0) or 0,
            likes=item.get("likes", 0) or 0,
            pipeline_tag=item.get("pipeline_tag", "") or "",
        )
        results.append(info)

    if engine:
        # 严格过滤：只保留当前架构可用的模型格式
        results = [m for m in results if filter_by_engine(m.repo_id, engine)]
    return results


def get_model_files(repo_id: str) -> list:
    """获取模型仓库的文件列表（过滤掉 .git 等无关文件）。"""
    data = _get(f"{HF_MIRROR}/api/models/{repo_id}").json()
    siblings = data.get("siblings", []) or []
    files = [s["rfilename"] for s in siblings
             if s.get("rfilename") and not s["rfilename"].startswith(".")]
    return files


def download_model(repo_id: str, progress_cb=None, cancel_check=None) -> str:
    """下载整个模型到 models/<repo> 目录。

    progress_cb(filename, downloaded_bytes, total_bytes)  用于进度上报。
    cancel_check() 返回 True 时中止下载并抛出 CancelledError。
    优先跳过已存在且大小一致的文件，支持断点续传（Range 请求）。
    """
    files = get_model_files(repo_id)
    # 只下载转写必需的文件，跳过 README/LICENSE 等无关大文件风险
    skip_pattern = re.compile(r"^(README|LICENSE|NOTICE|CONTRIBUTING)", re.I)
    files = [f for f in files if not skip_pattern.match(os.path.basename(f))]

    dest_dir = ModelInfo(repo_id).local_dir
    os.makedirs(dest_dir, exist_ok=True)

    session = requests.Session()
    for filename in files:
        if cancel_check and cancel_check():
            raise CancelledError()
        dest_path = os.path.join(dest_dir, filename)
        os.makedirs(os.path.dirname(dest_path) or dest_dir, exist_ok=True)
        url = f"{HF_MIRROR}/{repo_id}/resolve/main/{filename}"

        # 已存在且远端大小一致 → 跳过（断点续跑）
        remote_size = _remote_size(session, url)
        if os.path.isfile(dest_path) and remote_size and os.path.getsize(dest_path) == remote_size:
            if progress_cb:
                progress_cb(filename, remote_size, remote_size)
            continue

        headers = {}
        mode = "wb"
        resume_from = 0
        if os.path.isfile(dest_path) and remote_size and os.path.getsize(dest_path) < remote_size:
            resume_from = os.path.getsize(dest_path)
            headers["Range"] = f"bytes={resume_from}-"
            mode = "ab"

        with session.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as resp:
            resp.raise_for_status()
            total = remote_size or int(resp.headers.get("content-length", 0) or 0) + resume_from
            done = resume_from
            with open(dest_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if cancel_check and cancel_check():
                        raise CancelledError()
                    if chunk:
                        f.write(chunk)
                        done += len(chunk)
                        if progress_cb:
                            progress_cb(filename, done, total)
    return dest_dir


def _remote_size(session: requests.Session, url: str):
    """通过 HEAD 请求获取远端文件大小（失败返回 None，不阻塞下载）。"""
    try:
        r = session.head(url, timeout=TIMEOUT, allow_redirects=True)
        size = r.headers.get("content-length")
        return int(size) if size else None
    except Exception:
        return None


class CancelledError(Exception):
    """用户取消下载。"""
