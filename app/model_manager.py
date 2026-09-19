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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests

from . import settings
from .config import HF_MIRROR

TIMEOUT = 15
GET_RETRIES = 3          # 元信息/搜索请求失败重试次数（网络抖动自愈）

# 单个文件下载失败后的重试次数。大模型权重动辄数 GB，一次瞬时断连就整包失败
# 对用户代价太高；续传机制使重试成本极低（只补缺失部分），故默认重试 3 次。
FILE_RETRIES = 3

# 并行下载并发数：模型仓库通常含多个文件（权重 + config/vocab 等小文件），
# 适度并发可显著缩短总耗时；过高可能触发镜像站限流，默认 4。
MAX_DOWNLOAD_WORKERS = 4


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
        """是否已下载完成：兼容两种引擎的模型格式。

        - CTranslate2（faster-whisper / CUDA）：含 model.bin
        - MLX（mlx-whisper）：含 *.safetensors 或 weights.npz
        """
        d = self.local_dir
        if os.path.isfile(os.path.join(d, "model.bin")):
            return True
        if os.path.isdir(d):
            for name in os.listdir(d):
                if name.endswith(".safetensors") or name == "weights.npz":
                    return True
        return False


def _get(url: str, retries: int = GET_RETRIES, **kwargs) -> requests.Response:
    """带重试与指数退避的 GET（应对镜像站瞬时抖动）。"""
    last_exc = None
    for attempt in range(max(1, retries)):
        try:
            resp = requests.get(url, timeout=TIMEOUT, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(0.6 * (2 ** attempt))   # 0.6s → 1.2s 退避
    raise last_exc


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


def _required_files(repo_id: str) -> list:
    """返回需要下载的文件列表（跳过 README/LICENSE 等无关文件）。"""
    skip_pattern = re.compile(r"^(README|LICENSE|NOTICE|CONTRIBUTING)", re.I)
    files = get_model_files(repo_id)
    return [f for f in files if not skip_pattern.match(os.path.basename(f))]


def _content_range_total(header) -> int:
    """解析 Content-Range 头中的总长度。

    兼容两种写法：
      - 206 响应：`bytes 100-999/1000` → 1000
      - 416 响应：`bytes */1000`       → 1000
    解析失败返回 None。
    """
    if not header or "/" not in header:
        return None
    tail = header.rsplit("/", 1)[1].strip()
    return int(tail) if tail.isdigit() else None


def _make_reporter(progress_cb):
    """把进度回调包装为线程安全版本（并行下载时多线程会并发调用）。"""
    if progress_cb is None:
        return lambda *_: None
    lock = threading.Lock()

    def _report(filename, done, total):
        with lock:
            progress_cb(filename, done, total)

    return _report


def _download_file(repo_id: str, filename: str, dest_dir: str,
                   report, cancel_check, retries: int = FILE_RETRIES) -> None:
    """下载单个文件；瞬时网络错误自动重试（续传），其余交给 _download_file_once。"""
    for attempt in range(max(1, int(retries))):
        try:
            return _download_file_once(repo_id, filename, dest_dir, report, cancel_check)
        except CancelledError:
            raise
        except Exception as e:
            if attempt >= retries - 1 or not _is_retryable(e):
                raise
            time.sleep(0.6 * (2 ** attempt))   # 0.6s → 1.2s 退避


def _is_retryable(e: Exception) -> bool:
    """判断下载异常是否值得重试。

    重试：连接类错误、5xx/429、磁盘/校验类 IOError（续传即可修复）。
    不重试：4xx 客户端错误（如 404 仓库/文件不存在），重试只会浪费时间。
    """
    if isinstance(e, requests.HTTPError):
        code = getattr(getattr(e, "response", None), "status_code", None)
        return code is None or code >= 500 or code == 429
    return isinstance(e, (requests.RequestException, IOError))


def _download_file_once(repo_id: str, filename: str, dest_dir: str,
                        report, cancel_check) -> None:
    """下载单个文件一次，支持断点续传与完成后大小校验。

    - 本地已有部分内容 → 带 Range 请求续传（服务器返回 206）
    - 服务器忽略 Range 返回 200 → 截断重下（避免进度超 100%）
    - 返回 416（断点位置越界）→ 本地已完整则跳过，异常则删除重下
    - 下载完成后校验文件大小与远端一致，不符则报错
    """
    dest_path = os.path.join(dest_dir, filename)
    parent = os.path.dirname(dest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    url = f"{HF_MIRROR}/{repo_id}/resolve/main/{filename}"
    local_size = os.path.getsize(dest_path) if os.path.isfile(dest_path) else 0
    headers = {"Range": f"bytes={local_size}-"} if local_size > 0 else {}

    session = requests.Session()
    try:
        with session.get(url, headers=headers, stream=True, timeout=60,
                         allow_redirects=True) as resp:
            if resp.status_code == 416:
                # 断点位置越界：本地文件已达（或超过）远端大小
                total = _content_range_total(resp.headers.get("content-range"))
                if total is None or local_size == total:
                    report(filename, local_size, local_size)   # 视为已完成
                    return
                # 本地文件异常偏大 → 删除后全量重下一次
                os.remove(dest_path)
                return _download_file_once(repo_id, filename, dest_dir, report, cancel_check)

            resp.raise_for_status()
            if resp.status_code == 206:
                mode, resume_from = "ab", local_size
                total = _content_range_total(resp.headers.get("content-range"))
                if total is None:   # 无 Content-Range → 用剩余长度推算
                    total = resume_from + int(resp.headers.get("content-length", 0) or 0)
            else:
                # 服务器未按 Range 返回（200）→ 从头写入，避免重复计数
                mode, resume_from = "wb", 0
                total = int(resp.headers.get("content-length", 0) or 0)

            if mode == "ab" and total and resume_from >= total:
                report(filename, total, total)
                return

            done = resume_from
            with open(dest_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if cancel_check and cancel_check():
                        raise CancelledError()
                    if chunk:
                        f.write(chunk)
                        done += len(chunk)
                        report(filename, done, total)

        # 完成校验：已知总大小时必须完全一致，否则视为下载失败
        if total and os.path.getsize(dest_path) != total:
            raise IOError(
                f"下载校验失败：{filename} 期望 {total} 字节，"
                f"实际 {os.path.getsize(dest_path)} 字节（可重试续传）")
    finally:
        session.close()


def download_model(repo_id: str, progress_cb=None, cancel_check=None,
                   max_workers: int = MAX_DOWNLOAD_WORKERS) -> str:
    """并行下载整个模型到 models/<repo> 目录。

    progress_cb(filename, downloaded_bytes, total_bytes)  用于进度上报（线程安全）。
    cancel_check() 返回 True 时中止下载并抛出 CancelledError。
    多文件并行下载；每个文件支持断点续传（Range）、失败重试与完成后大小校验。
    """
    files = _required_files(repo_id)
    dest_dir = ModelInfo(repo_id).local_dir
    os.makedirs(dest_dir, exist_ok=True)
    if not files:
        return dest_dir

    report = _make_reporter(progress_cb)
    errors = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        futures = [ex.submit(_download_file, repo_id, fn, dest_dir, report, cancel_check)
                   for fn in files]
        for fut in as_completed(futures):
            try:
                fut.result()
            except BaseException as e:   # noqa: BLE001 - 收集后统一处理
                errors.append(e)

    # 取消优先于其它错误抛出；其余取首个错误
    for e in errors:
        if isinstance(e, CancelledError):
            raise e
    if errors:
        raise errors[0]
    return dest_dir


class CancelledError(Exception):
    """用户取消下载。"""
