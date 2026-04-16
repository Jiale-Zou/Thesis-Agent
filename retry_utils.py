"""
Retry utilities for this project (max 3 attempts by default).

Design goals:
- Work without vendor-specific exception imports (OpenAI/Anthropic/etc. differ by version).
- Support retrying LLM calls, JSON parsing, and generic callables.
- Exponential backoff with small jitter; bounded.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T") # 范形变量：代表任意返回值类型


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3 # 最大重试3次
    base_delay_s: float = 0.8 # 基础等待0.8秒
    max_delay_s: float = 6.0 # 最大等待6秒
    backoff: float = 2.0 # 指数倍数：每次×2
    jitter_s: float = 0.25 # 随机抖动±0.25秒


def _exc_name(e: BaseException) -> str:
    '''拿到异常的类名，例如 TimeoutError'''
    return e.__class__.__name__


def is_retriable_exception(e: BaseException) -> bool:
    """
    Heuristic retry classifier.

    We avoid importing provider-specific exception classes (they vary by package version).
    Instead we match common exception names / messages.
    """
    name = _exc_name(e)
    msg = str(e).lower()

    # 网络错误
    transient_names = {
        "TimeoutError",
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectionError",
        "RemoteDisconnected",
        "ProtocolError",
        "SSLError",
    }
    if name in transient_names:
        return True

    # AI服务商限流/过载
    providerish = (
        "RateLimit",
        "RateLimitError",
        "APITimeout",
        "APITimeoutError",
        "APIConnectionError",
        "ServiceUnavailable",
        "ServiceUnavailableError",
        "InternalServerError",
        "BadGateway",
        "GatewayTimeout",
        "Unavailable",
    )
    if any(tok in name for tok in providerish):
        return True

    # HTTP状态码
    if any(s in msg for s in ("429", "rate limit", "too many requests")):
        return True
    if any(s in msg for s in ("503", "service unavailable", "temporarily unavailable")):
        return True
    if any(s in msg for s in ("502", "bad gateway")):
        return True
    if any(s in msg for s in ("504", "gateway timeout")):
        return True

    # 超时关键词
    if "timeout" in msg or "timed out" in msg:
        return True

    return False


def _sleep_for_attempt(policy: RetryPolicy, attempt: int) -> None:
    # a计算重试等待时间
    delay = min(policy.max_delay_s, policy.base_delay_s * (policy.backoff ** (attempt - 1)))
    delay = max(0.0, delay + random.uniform(0.0, policy.jitter_s))
    time.sleep(delay)


def retry_call(
    fn: Callable[[], T], # 要执行的函数
    *,
    policy: RetryPolicy = RetryPolicy(),
    retriable: Callable[[BaseException], bool] = is_retriable_exception,
) -> T:
    last: Optional[BaseException] = None

    # 循环尝试最多 max_attempts 次
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn() # 执行函数 → 成功就返回
        except BaseException as e:
            last = e
            # 如果达到最大次数 或 错误不允许重试 → 抛出
            if attempt >= policy.max_attempts or not retriable(e):
                raise
            # 否则等待 → 重试
            _sleep_for_attempt(policy, attempt)
    # unreachable, 最终失败，抛出最后一次异常
    if last is not None:
        raise last
    raise RuntimeError("retry_call failed without exception")


def retry_json_parse(
    make_text: Callable[[], str],
    *,
    policy: RetryPolicy = RetryPolicy(),
    cleaner: Optional[Callable[[str], str]] = None, # 支持传入 cleaner 函数清理格式
) -> Any:
    """
    Retry flow useful for LLM JSON responses:
    - call make_text()
    - optionally clean the text
    - json.loads()
    Retries on JSON decode errors only (not on arbitrary exceptions).
    """
    last_err: Optional[BaseException] = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            raw = make_text() # 获取文本
            cleaned = cleaner(raw) if cleaner else raw
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_err = e # 解析JSON
            if attempt >= policy.max_attempts:
                raise
            _sleep_for_attempt(policy, attempt)
    if last_err is not None:
        raise last_err
    raise RuntimeError("retry_json_parse failed without exception")


def clean_json_response(text: str) -> str:
    """
    终极清理大模型返回的JSON字符串，解决所有常见格式错误：
    1. 移除 ```json ``` 代码块
    2. 移除 Markdown、自然语言解释、多余注释
    3. 修复单引号、中文引号错误
    4. 修复截断、缺失括号
    5. 移除非法换行、空白
    6. 提取被文字包裹的JSON
    """
    if not text:
        return ""

    cleaned = text.strip()

    # 1. 移除代码块 ```json ... ```
    cleaned = re.sub(r'```(?:json|JSON|javascript|js)?\n?', '', cleaned)
    cleaned = re.sub(r'```$', '', cleaned)

    # 2. 移除行内 `
    cleaned = re.sub(r'^`|`$', '', cleaned)

    #正确找到第一个 { 或 [，避免前面有文字
    start_curl = cleaned.find('{')
    start_brack = cleaned.find('[')
    candidates = []
    if start_curl >= 0:
        candidates.append(start_curl)
    if start_brack >= 0:
        candidates.append(start_brack)

    if candidates:
        start_idx = min(candidates)
        cleaned = cleaned[start_idx:]

    # 4. 截断到最后一个 } 或 ]
    end_brace = cleaned.rfind('}')
    end_bracket = cleaned.rfind(']')
    end_idx = -1
    if end_brace > end_idx:
        end_idx = end_brace
    if end_bracket > end_idx:
        end_idx = end_bracket

    if end_idx != -1:
        cleaned = cleaned[:end_idx + 1]

    cleaned = cleaned.strip()

    # 5. 单引号 → 双引号（只修键值对）
    def replace_single_quotes(s):
        s = re.sub(r"(?<!\\)'([^']*)'(?=\s*:)", r'"\1"', s)
        s = re.sub(r":\s*(?<!\\)'([^']*)'(?=\s*[,}\]])", r': "\1"', s)
        return s
    cleaned = replace_single_quotes(cleaned)

    # 6. 中文引号 → 英文
    cleaned = cleaned.replace('“', '"').replace('”', '"')

    # 7. 清理换行、制表符
    cleaned = re.sub(r'[\n\r\t]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # 8. 强制补全括号
    if cleaned.startswith('{') and not cleaned.endswith('}'):
        cleaned += '}'
    if cleaned.startswith('[') and not cleaned.endswith(']'):
        cleaned += ']'

    return cleaned