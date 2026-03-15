# app_main.py — 智能语音助手主服务
# 职责：FastAPI 服务器 + WebSocket 音频通信 + 语音对话全链路
#
# 教学说明（整体架构）：
# ┌──────────────┐       WebSocket /ws_audio        ┌──────────────┐
# │   ESP32      │ ──── 上行：麦克风 PCM 音频 ────→ │              │
# │ INMP441 麦克风│                                   │  Python 后端  │
# │ MAX98357 喇叭│ ←── 下行：HTTP /stream.wav ────── │  (本文件)     │
# └──────────────┘                                   └──────┬───────┘
#                                                          │
#                                          ┌───────────────┼───────────────┐
#                                          ↓               ↓               ↓
#                                     阿里云 ASR      通义千问 LLM    CosyVoice TTS
#                                     (语音识别)       (对话生成)      (语音合成)

import os
import sys
import time
import json
import asyncio
import wave
from collections import deque
from typing import Any, Dict, Optional, List
from pathlib import Path

# ---- Windows 控制台 UTF-8 ----
# 教学说明：Windows 默认用 GBK 编码，中文日志会乱码，强制 UTF-8
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
import uvicorn
import numpy as np

# ---- Windows 事件循环策略 ----
# 教学说明：Python 3.12+ 的 ProactorEventLoop 已修复兼容性问题
# WindowsSelectorEventLoopPolicy 在 Python 3.16 被移除，不再需要手动切换

# ---- 加载环境变量 ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- 读取配置 ----
def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()

API_KEY = _env("DASHSCOPE_API_KEY")
if not API_KEY:
    raise RuntimeError("未设置 DASHSCOPE_API_KEY，请编辑 .env 文件")

# 音频参数
SAMPLE_RATE = 16000             # 采样率
CHUNK_MS = 20                   # 每帧时长（毫秒）
BYTES_CHUNK = SAMPLE_RATE * CHUNK_MS // 1000 * 2  # 每帧字节数（16bit）
SILENCE_20MS = bytes(BYTES_CHUNK)  # 20ms 静音帧

# API 配置
BASE_URL = _env("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
WS_URL = _env("DASHSCOPE_WS_BASE_URL", "wss://dashscope.aliyuncs.com/api-ws/v1/inference")
ROUTER_MODEL = _env("ROUTER_MODEL", "qwen-turbo")
TEXT_MODEL = _env("TEXT_MODEL", "qwen-turbo")
TTS_MODEL = _env("TTS_MODEL", "cosyvoice-v3-plus")
DEFAULT_VOICE = _env("DEFAULT_VOICE", "longanhuan")

# ---- 引入语音系统模块 ----
from voice_adapter import WebSocketASREngine
from voice_core.cosyvoice import CosyVoiceEngine, _normalize_for_tts, _strip_emojis_for_tts
from voice_core.dispatcher import IntentDispatcher
from voice_core.workflows import Workflows
from voice_core.state import ChatState, DIALECT_VOICES

from audio_stream import (
    register_stream_route,
    broadcast_pcm16_realtime,
    hard_reset_audio,
    BYTES_PER_20MS_16K,
    is_playing_now,
    wait_for_stream_client,
    send_silence_prebuffer,
    set_current_ai_task,
    STREAM_SR,
)

# ===== 全局状态 =====
chat_state: Optional[ChatState] = None
asr_engine: Optional[WebSocketASREngine] = None
tts_engine: Optional[CosyVoiceEngine] = None
dispatcher: Optional[IntentDispatcher] = None
workflows: Optional[Workflows] = None


def init_voice_system():
    """
    初始化语音系统所有组件。

    教学说明（初始化顺序很重要）：
    1. ChatState: 对话状态（不依赖任何外部服务）
    2. ASR Engine: 语音识别引擎（连接阿里云 ASR）
    3. TTS Engine: 语音合成引擎（连接阿里云 CosyVoice）
    4. Dispatcher: 意图路由器（连接阿里云 LLM）
    5. Workflows: 工作流执行器（依赖上面所有组件）
    """
    global chat_state, asr_engine, tts_engine, dispatcher, workflows

    chat_state = ChatState(default_voice=DEFAULT_VOICE, current_voice=DEFAULT_VOICE, max_turns=8)
    asr_engine = WebSocketASREngine(api_key=API_KEY, sample_rate=SAMPLE_RATE, ws_url=WS_URL)
    tts_engine = CosyVoiceEngine(api_key=API_KEY, tts_model=TTS_MODEL, default_voice=DEFAULT_VOICE, sample_rate=SAMPLE_RATE)
    dispatcher = IntentDispatcher(api_key=API_KEY, base_url=BASE_URL, router_model=ROUTER_MODEL, text_model=TEXT_MODEL)
    workflows = Workflows(api_key=API_KEY, base_url=BASE_URL, cosy=tts_engine, dispatcher=dispatcher)

    print("[VOICE] 语音系统初始化完成", flush=True)


# ===== ASR 管理 =====
_current_recognition = None
_rec_lock = asyncio.Lock()


async def set_current_recognition(r):
    global _current_recognition
    async with _rec_lock:
        _current_recognition = r


async def stop_current_recognition():
    """
    停止当前 ASR 识别（线程池执行 + await 等待完成）。

    教学说明（对比 RobotDuck 的两种方案）：
    - 方案 A（RobotDuck）：直接调用 r.stop()
      → 同步阻塞事件循环 0.5-1.5 秒（期间音频处理/UI全停）
      → 但保证 ASR WebSocket 关完后才开 TTS WebSocket
    - 方案 B（我们之前）：run_in_executor 不 await（fire-and-forget）
      → 不阻塞事件循环（好）
      → 但 ASR WebSocket 可能还没关完就去开 TTS → 5s 超时（坏）
    - 方案 C（现在）：run_in_executor + await
      → 不阻塞事件循环（事件循环照常运行）
      → 等 ASR 真正关完才继续（保证不冲突）
      → 兼顾了 A 的可靠性和 B 的异步性 ✅
    """
    global _current_recognition
    async with _rec_lock:
        r = _current_recognition
        _current_recognition = None
    if r:
        # 在线程池中停止 ASR，不阻塞事件循环，但 await 等它关完
        # 教学说明（踩坑修复）：
        # 原来用 fire-and-forget（不 await），ASR 的 WebSocket 还没关完
        # 后面 TTS 就去开新 WebSocket → DashScope 并发冲突 → 5s 超时
        # RobotDuck 用同步 r.stop()（阻塞事件循环 1-2 秒，暴力但有效）
        # 我们用 await run_in_executor：既等 ASR 关完，又不阻塞事件循环
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: _safe_stop_recognition(r))


def _safe_stop_recognition(r):
    """在后台线程安全地停止 ASR 识别"""
    try:
        r.stop()
    except Exception:
        pass


# ===== FastAPI 应用 =====
# 教学说明（lifespan 上下文管理器）：
# FastAPI 废弃了 @app.on_event("startup")，改用 lifespan
# yield 之前 = 启动代码，yield 之后 = 关闭代码，写在一起更清晰
@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期：启动时初始化，关闭时清理"""
    init_voice_system()
    print("=" * 60)
    print("  智能语音助手已启动")
    print(f"  Web 面板: http://localhost:8081")
    print(f"  ESP32 音频: ws://你的电脑IP:8081/ws_audio")
    print(f"  音频流:     http://你的电脑IP:8081/stream.wav")
    print("=" * 60)
    yield
    # 关闭时的清理逻辑（目前不需要）

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 状态容器
ui_clients: Dict[int, WebSocket] = {}  # Web UI 客户端
current_partial: str = ""               # 当前 ASR 中间结果
recent_finals: List[str] = []           # 最近的 ASR 最终结果
RECENT_MAX = 50
esp32_audio_ws: Optional[WebSocket] = None  # ESP32 音频 WebSocket 连接

# 音频滚动缓冲区（即时克隆用）
# 教学说明：ESP32 上传音频时，同时往这个缓冲区存一份
# 用户说"克隆我的声音"时，直接用缓冲区里的语音克隆，不需要额外录音
# 只保留最近 15 秒的音频（16kHz 16bit mono = 32000 bytes/sec → 480KB）
_clone_rolling_buf: deque[bytes] = deque()  # 音频帧双端队列（popleft 是 O(1)，比 list.pop(0) 的 O(n) 快）
_clone_rolling_total: int = 0          # 当前缓冲区总字节数
_CLONE_BUF_MAX = 15 * SAMPLE_RATE * 2  # 最多保留 15 秒

# 中断锁
interrupt_lock = asyncio.Lock()


# ===== UI 广播工具函数 =====

async def ui_broadcast_raw(msg: str):
    """向所有 Web UI 客户端广播消息"""
    dead = []
    for k, ws in list(ui_clients.items()):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(k)
    for k in dead:
        ui_clients.pop(k, None)


async def ui_broadcast_partial(text: str):
    """广播 ASR 中间结果"""
    global current_partial
    current_partial = text
    await ui_broadcast_raw("PARTIAL:" + text)


async def ui_broadcast_final(text: str):
    """广播 ASR 最终结果"""
    global current_partial, recent_finals
    current_partial = ""
    recent_finals.append(text)
    if len(recent_finals) > RECENT_MAX:
        recent_finals = recent_finals[-RECENT_MAX:]
    await ui_broadcast_raw("FINAL:" + text)
    print(f"[ASR/AI FINAL] {text}", flush=True)


# ===== 系统重置 =====

async def full_system_reset(reason: str = ""):
    """
    全系统重置：回到刚启动的状态。

    教学说明：
    - 用户说"停下"/"别说了" → 触发全系统重置
    - 或者出错需要恢复 → 也调用这个
    """
    await hard_reset_audio(reason or "full_system_reset")
    await stop_current_recognition()

    global current_partial, recent_finals
    current_partial = ""
    recent_finals = []

    # 通知 ESP32 重置
    try:
        if esp32_audio_ws and (esp32_audio_ws.client_state == WebSocketState.CONNECTED):
            await esp32_audio_ws.send_text("RESET")
    except Exception:
        pass

    print("[SYSTEM] 全系统重置完成", flush=True)


# ===== 非流式 TTS 播放 =====

async def _speak_text_to_broadcast(text: str, voice: str, instruction: Optional[str],
                                    model: Optional[str] = None):
    """
    非流式 TTS：文本 → WAV 文件 → PCM → 广播给 ESP32。
    用于短句回复（如"好的，已切换方言"）。
    model: 可选模型覆盖（方言音色需要 cosyvoice-v3-flash）
    """
    if not text or not text.strip():
        return
    if not tts_engine:
        print("[AI] 错误：TTS 引擎未初始化", flush=True)
        return

    import uuid
    runtime_dir = Path("runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    wav_path = runtime_dir / f"tts_{uuid.uuid4().hex}.wav"

    try:
        # 清洗 emoji 和 markdown（防止 Windows GBK 编码崩溃）
        # 教学说明：流式 TTS 的 flush_text() 已有清洗，但非流式路径遗漏了
        # 所有走 _speak_text_to_broadcast() 的调用（方言确认、克隆提示、兜底回复等）
        # 都在这里统一清洗，不需要每个调用点单独处理
        text = _normalize_for_tts(_strip_emojis_for_tts(text))
        if not text or not text.strip():
            return

        # 文本 → WAV 文件
        tts_engine.tts_to_wav(text=text, voice=voice, instruction=instruction,
                              out_path=str(wav_path), model=model)

        # WAV → PCM bytes
        with wave.open(str(wav_path), "rb") as wf:
            data = wf.readframes(wf.getnframes())

        # 广播给 ESP32
        await broadcast_pcm16_realtime(data)

    finally:
        try:
            if wav_path.exists():
                wav_path.unlink()
        except Exception:
            pass


# ===== 流式 TTS 播放 =====

async def _async_gen(sync_gen):
    """
    把同步生成器包装成异步生成器，避免阻塞事件循环。

    教学说明：
    - LLM 的 chat_answer_stream() 返回同步生成器（用 yield）
    - 每次取下一个 token 要等网络 IO（50-100ms）
    - 如果直接 for delta in sync_gen: 会冻住事件循环
    - 冻住期间 process_audio() 无法发送音频 → ESP32 缓冲区空了 → 卡顿
    - 用 run_in_executor 把阻塞等待放到后台线程 → 事件循环持续运转
    """
    loop = asyncio.get_running_loop()
    _sentinel = object()  # 哨兵值，用于检测生成器耗尽
    while True:
        # 在线程池中等待下一个 token（不阻塞事件循环）
        item = await loop.run_in_executor(None, next, sync_gen, _sentinel)
        if item is _sentinel:
            break
        yield item


async def _speak_stream_to_broadcast(text_stream, voice: str, instruction: Optional[str],
                                      txt_buf: List[str], model: Optional[str] = None):
    """
    流式 TTS：LLM 逐 token → 分块送 TTS → PCM 音频 → 广播。
    model: 可选模型覆盖（方言音色需要 cosyvoice-v3-flash）

    教学说明（流式 TTS 的核心设计）：
    1. LLM 逐个 token 返回文本（如 "你"、"好"、"，"、"今天"...）
    2. 我们把 token 攒到一定长度（或遇到标点），再发给 TTS
       - 太碎：TTS 生成断续，有爆音
       - 太长：首句延迟大
       - 所以有 MIN_CHARS / MAX_CHARS / MAX_WAIT 三个策略
    3. TTS 的 callback.on_data() 返回 PCM 音频块
    4. 异步任务把音频块广播给 ESP32
    """
    from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer, ResultCallback
    import queue
    import threading

    # 音频数据队列：TTS 回调（后台线程）→ 队列 → 异步广播
    audio_queue: queue.Queue[Optional[bytes]] = queue.Queue()
    tts_done = threading.Event()
    tts_error = [None]

    # 首音频标记（用于计时）
    _first_audio_time = [None]

    class StreamCallback(ResultCallback):
        """TTS 流式回调：接收 PCM 音频数据"""
        def on_data(self, data: bytes):
            if data:
                if _first_audio_time[0] is None:
                    _first_audio_time[0] = time.time()
                audio_queue.put(data)

        def on_complete(self):
            audio_queue.put(None)  # 发送结束信号
            tts_done.set()

        def on_error(self, msg):
            tts_error[0] = str(msg)
            audio_queue.put(None)
            tts_done.set()

    callback = StreamCallback()

    # 创建流式 TTS 合成器（放线程池，构造函数可能建立 WebSocket 连接阻塞 1-3s）
    # model 参数允许覆盖默认模型（方言音色需要 cosyvoice-v3-flash）
    _tts_model = model or tts_engine.tts_model
    _t_tts_init = time.time()
    def _create_tts():
        return SpeechSynthesizer(
            model=_tts_model,
            voice=voice,
            format=AudioFormat.PCM_16000HZ_MONO_16BIT,
            instruction=instruction,
            callback=callback,
        )
    _init_loop = asyncio.get_running_loop()
    tts = await _init_loop.run_in_executor(None, _create_tts)
    _tts_init_cost = time.time() - _t_tts_init
    if _tts_init_cost > 0.1:
        print(f"[TIMING] ⚠ TTS 初始化耗时 {_tts_init_cost:.3f}s（已在线程池，不阻塞事件循环）", flush=True)

    # 文本分块策略（自适应：首句小块快响应，后续大块防断供）
    # 教学说明：TTS 每次 streaming_call() 都有 200-500ms 的处理延迟
    # 分块太碎（如 12 字一段）→ 频繁调用 TTS → 间隙多 → ESP32 缓冲区播空 → 卡断
    # 分块太大（如 90 字一段）→ 首句等太久 → 感觉迟钝
    # 所以：第一句用小块（快开口），后续用大块（不断供）
    # 教学说明（对比 RobotDuck 的优化参数）：
    # RobotDuck cosyvoice.py 用 MIN=28/MAX=90/WAIT=0.9（通用保守值）
    # RobotDuck app_main.py 用 MIN=12/MAX=60/WAIT=0.5（实际优化值）
    # 我们之前误用了 cosyvoice.py 的保守值，导致后续分块太大、延迟高
    FIRST_MIN_CHARS = 6  # 首句：攒 6 个字就发（快速开口）
    LATER_MIN_CHARS = 12 # 后续：攒 12 个字再发（对齐 RobotDuck app_main 优化值）
    MAX_CHARS = 60       # 最多 60 字必须发（从 80 降低，更频繁发送减少间隙）
    MAX_WAIT = 0.5       # 最长等 0.5 秒（从 0.8 降低，兜底更快）
    PUNCT = set("。！？!?；;\n，,：:")  # 添加逗号冒号（中文最常见的断句点！）

    buf = ""
    last_send = time.time()
    sent_count = 0       # 已发送的 TTS 块计数

    async def flush_text():
        """把缓冲区的文本发给 TTS（异步，防止阻塞事件循环）"""
        nonlocal buf, last_send, sent_count, _first_tts_logged, tts
        if not buf:
            return
        # 清洗 emoji 和 markdown
        to_send = _normalize_for_tts(_strip_emojis_for_tts(buf))
        if to_send:
            if not _first_tts_logged:
                print(f"[TIMING] 首次 TTS 送文 \"{to_send[:20]}\" (延迟 {time.time()-_t_stream_start:.3f}s)", flush=True)
                _first_tts_logged = True
            _t_call = time.time()
            # 关键优化：streaming_call 首次调用会建立 WebSocket 连接，可能阻塞 3-5 秒
            # 放到线程池执行，避免冻结事件循环（冻结会导致音频广播停滞）
            #
            # 重试机制（教学说明）：
            # DashScope SDK 里 WebSocket 连接超时写死 5 秒
            # 网络波动时 3-5 秒很常见 → 经常卡在边界 → 超时失败
            # 解决：失败后重建 SpeechSynthesizer 再试，最多 3 次
            _loop = asyncio.get_running_loop()
            _max_retries = 3
            for _attempt in range(_max_retries):
                try:
                    await _loop.run_in_executor(None, tts.streaming_call, to_send)
                    break  # 成功，跳出重试循环
                except (TimeoutError, Exception) as _e:
                    if _attempt < _max_retries - 1:
                        print(f"[TTS] streaming_call 第{_attempt+1}次失败: {_e}，重建连接重试...", flush=True)
                        await asyncio.sleep(0.3)  # 缩短等待（ASR 已在准备阶段关完）
                        # 重建 SpeechSynthesizer（新的 WebSocket 连接）
                        tts = await _loop.run_in_executor(None, _create_tts)
                    else:
                        raise  # 最后一次仍失败，抛出异常
            _call_cost = time.time() - _t_call
            if _call_cost > 0.1:
                print(f"[TIMING] ⚠ streaming_call 耗时 {_call_cost:.3f}s（已在线程池，不阻塞事件循环）", flush=True)
            sent_count += 1
        buf = ""
        last_send = time.time()

    # 异步任务：从队列取音频数据并广播
    _first_audio_broadcast = [False]

    async def process_audio():
        while True:
            try:
                try:
                    audio_data = audio_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                if audio_data is None:
                    break

                if audio_data:
                    if not _first_audio_broadcast[0]:
                        _first_audio_broadcast[0] = True
                        print(f"[TIMING] 首音频广播 (总延迟 {time.time()-_t_stream_start:.3f}s)", flush=True)
                    await broadcast_pcm16_realtime(audio_data)
            except Exception as e:
                print(f"[AI] 音频处理错误: {e}", flush=True)
                break

    audio_task = asyncio.create_task(process_audio())

    # 计时：测量首 token、首 TTS、首音频延迟
    _t_stream_start = time.time()
    _first_token_logged = False
    _first_tts_logged = False

    try:
        # 处理 LLM 文本流（用异步迭代，避免阻塞事件循环导致音频卡顿）
        async for delta in _async_gen(text_stream):
            # 记录首 token 时间
            if not _first_token_logged:
                print(f"[TIMING] LLM 首token延迟 {time.time()-_t_stream_start:.3f}s", flush=True)
                _first_token_logged = True

            txt_buf.append(delta)
            full_text = "".join(txt_buf)

            # 更新 UI 显示
            try:
                await ui_broadcast_partial("[AI] " + full_text)
            except Exception:
                pass

            buf += delta
            now = time.time()

            # 分块发送策略（自适应：首句快，后续稳）
            min_chars = FIRST_MIN_CHARS if sent_count == 0 else LATER_MIN_CHARS

            if len(buf) >= MAX_CHARS:
                await flush_text()
                continue

            if len(buf) >= min_chars and any(ch in PUNCT for ch in delta):
                await flush_text()
                continue

            if len(buf) >= min_chars and (now - last_send) >= MAX_WAIT:
                await flush_text()
                continue

            if (now - last_send) >= 6.0 and len(buf) > 0:
                await flush_text()

            await asyncio.sleep(0)  # 让出事件循环

        # 发送剩余文本
        if buf.strip():
            await flush_text()

        # 通知 TTS 流式完成 + 等待完成（都可能阻塞，放线程池）
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, tts.streaming_complete)
        await loop.run_in_executor(None, lambda: tts_done.wait(timeout=30.0))

        # 等待音频队列清空
        while not audio_queue.empty():
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.2)

        if tts_error[0]:
            raise RuntimeError(tts_error[0])

    finally:
        audio_task.cancel()
        try:
            await audio_task
        except asyncio.CancelledError:
            pass


# ===== AI 回复主流程 =====

async def start_ai_with_text(user_text: str):
    """
    核心函数：收到用户语音 → 路由 → LLM 回复 → TTS 播放。

    教学说明（完整流程）：
    1. 硬重置音频（打断之前的播放，但保留连接）
    2. 停止 ASR（不要把 AI 的声音又识别了）
    3. 通知 ESP32 停止上传音频
    4. 意图路由：判断用户想干什么
    5. 根据意图执行：聊天/克隆/方言/重置
    6. TTS 播放回复
    7. 播放完成后重启 ASR（等待下一轮对话）
    """
    if not chat_state or not dispatcher or not tts_engine or not workflows:
        print("[AI] 错误：语音系统未初始化", flush=True)
        return

    async def _runner():
        # 教学说明（Python 作用域踩坑）：
        # 克隆成功后要执行 _clone_rolling_total = 0（赋值）
        # Python 规则：函数内有赋值 → 整个函数内该变量被视为局部变量
        # 不加 global → 前面读取时报 UnboundLocalError（局部变量还没赋值就被读了）
        global _clone_rolling_total
        txt_buf: List[str] = []
        _was_cancelled = False  # 是否被新输入打断（影响 finally 清理行为）
        # 计时器：精确测量每一步耗时，找出延迟瓶颈
        _t_start = time.time()

        try:
            # 0. 关键词预检：普通聊天直接跳过 LLM 路由，省 1-2 秒
            # 教学说明：只有特殊意图（克隆/方言/重置/角色）才需要调 LLM 路由
            # 普通对话占 90% 以上，直接走 default 快得多
            from voice_core.dispatcher import RouteDecision
            _special_keywords = {
                "clone": ["克隆", "模仿", "学我的声音", "用我的声音", "复制声音",
                          "学我说话", "模仿我", "学我的", "克隆声音"],
                "dialect": ["粤语", "白话", "广东话", "东北话", "四川话", "川话",
                           "上海话", "沪语", "闽南话", "台语", "河南话", "山东话",
                           "天津话", "云南话", "贵州话", "湖北话", "江西话",
                           "陕西话", "山西话", "甘肃话", "宁夏话", "用方言"],
                "reset": ["恢复默认", "默认模式", "默认音色", "别模仿了", "换回来",
                          "不要克隆", "用原来的声音", "用回你的声音", "不用我的声音",
                          "取消克隆", "换回原来"],
                "role_scene": ["脱口秀", "rap", "押韵", "唱", "客服", "解说",
                              "电台", "诗歌", "科普", "推广", "你是"],
            }
            # 教学说明（关键词预检逻辑）：
            # 1. 先用关键词匹配判断意图类型
            # 2. "克隆"关键词 → 直接走 clone，不调 LLM（快且准）
            #    原因：LLM 路由有时不选 workflow_clone → 用户说"克隆"没反应
            # 3. 其他特殊意图（方言/重置/角色）→ 仍走 LLM 路由（需要解析参数）
            # 4. 普通聊天 → 直接走 default（省 1-2 秒）
            matched_intent = None
            text_lower = user_text.strip().lower()
            for intent, keywords in _special_keywords.items():
                if any(kw in text_lower for kw in keywords):
                    matched_intent = intent
                    break

            if matched_intent == "clone":
                # 克隆意图：直接设定，不调 LLM 路由（避免 LLM 误判为聊天）
                decision = RouteDecision(intent="clone", emotion="neutral", query=user_text)
                print(f"[AI] 关键词直接命中 clone，跳过 LLM 路由", flush=True)
            elif matched_intent:
                # 其他特殊意图：调 LLM 路由精确判断（需要解析方言名/角色名等参数）
                decision = dispatcher.route(user_text, chat_state)
            else:
                # 普通聊天：直接走 default，跳过路由（省 1-2 秒）
                decision = RouteDecision(intent="default", emotion="neutral", query=user_text)
            emotion = decision.emotion
            print(f"[AI] 意图: {decision.intent}, 情绪: {emotion} (路由耗时 {time.time()-_t_start:.3f}s)", flush=True)

            # 2. 根据意图选择处理方式
            if decision.intent in ("default", "dialect", "role_scene"):
                # === 流式处理：文本流 + TTS 流 ===

                # 更新方言设置
                if decision.intent == "dialect":
                    if decision.dialect:
                        chat_state.dialect = decision.dialect
                    if not (decision.query or "").strip():
                        # 根据是否有方言专用音色，给不同的确认语
                        _has_dialect_voice = (chat_state.dialect in DIALECT_VOICES)
                        if _has_dialect_voice or chat_state.is_cloned_voice:
                            ack = f"好的，接下来我会用{chat_state.dialect or '方言'}和你聊。"
                        else:
                            # 没有专用方言音色也没克隆，只能文字模拟
                            ack = (f"好的，接下来我会用{chat_state.dialect or '方言'}的口吻和你聊。"
                                   "克隆你的声音后可以获得更真实的方言发音。")
                        # 获取方言对应的音色和模型
                        voice, tts_model = chat_state.get_tts_voice_and_model(tts_engine.tts_model)
                        instruction = chat_state.build_tts_instruction(emotion)
                        await _speak_text_to_broadcast(ack, voice, instruction, model=tts_model)
                        txt_buf.append(ack)
                        return

                # 更新角色/场景设置
                if decision.intent == "role_scene":
                    if decision.role:
                        chat_state.role = decision.role
                    if decision.scene:
                        chat_state.scene = decision.scene
                    if decision.style_hint:
                        chat_state.style_hint = decision.style_hint
                    if not (decision.query or "").strip():
                        ack = "好的，已进入角色/场景模式。你想让我怎么表演？"
                        instruction = chat_state.build_tts_instruction(emotion)
                        await _speak_text_to_broadcast(ack, chat_state.current_voice, instruction)
                        txt_buf.append(ack)
                        return

                # 流式生成文本并 TTS
                query = (decision.query or "").strip() or user_text
                # 获取当前应使用的音色和模型（方言模式会切换到方言音色+v3-flash）
                voice, tts_model = chat_state.get_tts_voice_and_model(tts_engine.tts_model)
                instruction = chat_state.build_tts_instruction(emotion)
                _t_llm = time.time()
                text_stream = dispatcher.chat_answer_stream(query, chat_state, emotion)
                print(f"[TIMING] LLM 流创建耗时 {time.time()-_t_llm:.3f}s (总 {time.time()-_t_start:.3f}s)", flush=True)
                await _speak_stream_to_broadcast(text_stream, voice, instruction, txt_buf, model=tts_model)

                # 兜底：如果 LLM 返回空（乱码输入时常见），给个默认回复
                # 教学说明：ASR 有时会把噪音识别成 "再关一。" 之类的乱码
                # LLM 不知道怎么回答 → 返回空 → 用户看到"空响应"
                if not "".join(txt_buf).strip():
                    fallback = "抱歉，我没听清楚，你能再说一次吗？"
                    await _speak_text_to_broadcast(fallback, voice, instruction, model=tts_model)
                    txt_buf.append(fallback)

            elif decision.intent == "clone":
                # === 即时音色克隆：直接用滚动缓冲区里的语音 ===
                # 教学说明（即时克隆流程）：
                # 用户聊天时，ESP32 上传的音频同时存入滚动缓冲区
                # 用户说"克隆" → 直接用缓冲区里的语音克隆 → 零等待
                instruction = chat_state.build_tts_instruction("neutral")

                # 教学说明：stop_current_recognition() 现在已改为 await
                # ASR WebSocket 在 start_ai_with_text() 的准备阶段就已关完
                # 不再需要手动等待（之前要 sleep 1.0 秒是因为 fire-and-forget）

                # 检查缓冲区状态（详细日志，方便排查）
                buf_frames = len(_clone_rolling_buf)
                buf_bytes = _clone_rolling_total
                buf_sec = buf_bytes / (SAMPLE_RATE * 2)
                print(f"[CLONE] 缓冲区状态: {buf_frames} 帧, {buf_bytes} 字节, {buf_sec:.1f} 秒", flush=True)

                # CosyVoice 至少需要 5 秒有效语音
                _min_clone_bytes = 5 * SAMPLE_RATE * 2  # 5 秒 = 160000 字节
                if buf_bytes < _min_clone_bytes:
                    # 缓冲区太短，用友好的方式引导用户继续聊天
                    # 教学说明：不说"录音不够"，而说"需要多听几句"
                    # 让用户感觉系统是在"学习"而不是"要求录音"
                    reply_text = f"我需要多听你几句话才能学会你的声音。现在只收集到{buf_sec:.1f}秒，至少需要5秒。你先随便跟我聊会儿天，聊完再说克隆就行。"
                    print(f"[CLONE] 缓冲区不足: {buf_sec:.1f}s < 5.0s", flush=True)
                    await _speak_text_to_broadcast(reply_text, chat_state.current_voice, instruction)
                    txt_buf.append(reply_text)
                else:
                    # 1) 播放提示（带重试，防止 WebSocket 偶发超时）
                    # 教学说明：TTS 内部用 WebSocket 连 DashScope，偶尔会超时
                    # 提示语只是礼貌用语，失败了不影响克隆流程，跳过即可
                    prompt_text = "收到，正在克隆你的声音，请稍等。"
                    try:
                        await _speak_text_to_broadcast(prompt_text, chat_state.current_voice, instruction)
                    except Exception as e:
                        print(f"[CLONE] 提示语 TTS 失败（跳过，不影响克隆）: {e}", flush=True)
                    txt_buf.append(prompt_text)

                    # 2) 从滚动缓冲区取出音频（最多 10 秒，太长质量反而下降）
                    pcm_data = b''.join(_clone_rolling_buf)
                    _max_clone_bytes = 10 * SAMPLE_RATE * 2  # 10 秒
                    if len(pcm_data) > _max_clone_bytes:
                        pcm_data = pcm_data[-_max_clone_bytes:]  # 取最近 10 秒
                    print(f"[CLONE] 使用缓冲区音频: {len(pcm_data)} 字节 = {len(pcm_data)/32000:.1f}秒", flush=True)

                    # 3) 后台克隆：立即回复用户，克隆在后台进行
                    # 教学说明（后台克隆方案）：
                    # CosyVoice 不支持 URL 即时克隆（传 URL 当 voice → 418 错误）
                    # 必须走注册流程（enroll），但注册要 1-2 分钟
                    # 解决：把注册丢到后台任务，用户不用等，继续聊天
                    # 后台任务完成后自动切换音色，下一次对话就用克隆声音
                    _loop = asyncio.get_running_loop()
                    _clone_pcm = pcm_data  # 保存引用，后台任务用

                    async def _bg_clone():
                        """后台克隆任务：注册音色 + 切换"""
                        try:
                            print(f"[CLONE] 后台克隆开始...", flush=True)
                            new_voice = await _loop.run_in_executor(
                                None, tts_engine.enroll_voice_from_pcm, _clone_pcm, "myvoice"
                            )
                            chat_state.set_cloned_voice(new_voice)
                            _clone_rolling_buf.clear()
                            _clone_rolling_total = 0
                            print(f"[CLONE] 后台克隆成功！音色ID: {new_voice}", flush=True)
                            # 用默认音色通知用户（克隆音色可能需要热身）
                            notify = "克隆完成了！接下来我会用你的声音说话。"
                            try:
                                instruction_n = chat_state.build_tts_instruction("neutral")
                                await _speak_text_to_broadcast(notify, chat_state.current_voice, instruction_n)
                            except Exception:
                                pass
                        except Exception as e:
                            print(f"[CLONE] 后台克隆失败: {e}", flush=True)

                    # 启动后台任务（不 await，不阻塞当前流程）
                    asyncio.create_task(_bg_clone())

                    # 立即回复用户（用默认音色），不用等克隆完成
                    reply_text = "好的，我在后台学习你的声音，你可以继续跟我聊天。学会了我会告诉你的。"
                    await _speak_text_to_broadcast(reply_text, chat_state.default_voice, instruction)
                    txt_buf.append(reply_text)

            else:
                # === 非流式处理（reset 等）===
                reply_text = workflows.run(decision, user_text, chat_state)
                instruction = chat_state.build_tts_instruction(emotion)
                await _speak_text_to_broadcast(reply_text, chat_state.current_voice, instruction)
                txt_buf.append(reply_text)

        except asyncio.CancelledError:
            # 被新的 ASR 结果打断 → 标记为取消，跳过重型清理
            _was_cancelled = True
            # 关键修复：清理孤儿用户消息
            # chat_answer_stream() 先 add_user() 再流式生成，最后才 add_assistant()
            # 被打断时 add_assistant() 没执行 → messages 末尾留下孤儿 user 消息
            # 多次打断后出现连续 user 消息 → LLM API 报错 → "长对话老是出错"
            if chat_state:
                chat_state.pop_last_user_if_orphan()
            raise
        except Exception as e:
            print(f"[AI] 错误: {e}", flush=True)
            import traceback
            traceback.print_exc()
            try:
                await ui_broadcast_final(f"[AI] 发生错误：{e}")
            except Exception:
                pass
        finally:
            # 教学重点：取消 vs 自然完成的 finally 行为完全不同！
            #
            # 被取消（用户说了新的话）：
            #   - 不发 None（保留 HTTP 流连接给新任务用）
            #   - 不 sleep（新任务在等我们退出，每多等 1ms 都是延迟）
            #   - 不发 START（新任务会自己管 ASR）
            #
            # 自然完成（AI 说完了）：
            #   - 发 None 关闭 HTTP 流（让 ESP32 重连以获取下一轮音频）
            #   - sleep 0.5s 等喇叭播完
            #   - 发 START 重启 ASR（准备听下一句）
            #
            # 这个区分是延迟优化的关键！
            # 原来不区分，取消时也跑全套清理 → cancel_current_ai 的 await task
            # 要等 0.5s+ 才返回 → 准备阶段从 <0.1s 变成 2-27s！

            if not _was_cancelled:
                # === 自然完成：全套清理 ===
                # 教学说明（连接复用优化）：
                # 旧方案：发 None → 关闭 HTTP 流 → ESP32 重连 → 下一轮多等 200-500ms
                # 新方案：发静音帧 + None 哨兵（stream 生成器收到 None 不退出，只是继续等）
                # 这样 /stream.wav 连接保持活跃 → 下一轮直接复用 → 省去重连延迟
                from audio_stream import stream_clients
                for sc in list(stream_clients):
                    if not sc.abort_event.is_set():
                        try:
                            sc.q.put_nowait(b"\x00" * BYTES_PER_20MS_16K)
                        except Exception:
                            pass

                # 广播最终结果到 UI
                final_text = ("".join(txt_buf)).strip() or "（空响应）"
                try:
                    await ui_broadcast_final("[AI] " + final_text)
                except Exception:
                    pass
                try:
                    await ui_broadcast_partial("")
                except Exception:
                    pass

                # 等待喇叭播完 + 回声消散后再重启 ASR
                await asyncio.sleep(0.5)
                print("[AI] 播放完成，准备重启 ASR", flush=True)

                try:
                    if esp32_audio_ws and (esp32_audio_ws.client_state == WebSocketState.CONNECTED):
                        await esp32_audio_ws.send_text("START")
                        print("[AI] 已通知 ESP32 重启 ASR", flush=True)
                except Exception as e:
                    print(f"[AI] 通知 ESP32 重启 ASR 失败：{e}", flush=True)
            else:
                # === 被取消：极速退出 ===
                print("[AI] 被新输入打断，跳过清理", flush=True)

    # --- 启动前准备 ---
    _t_setup = time.time()
    print(f"[TIMING] === 新一轮对话开始 ===", flush=True)

    # 硬重置（保留连接，避免 ESP32 重新拉流）
    await hard_reset_audio("start_ai_with_text", keep_connections=True)
    await stop_current_recognition()

    # 清空 UI 状态
    global current_partial
    current_partial = ""
    await ui_broadcast_partial("")

    # 通知 ESP32 停止上传音频（防止 AI 声音被 ASR 识别）
    try:
        if esp32_audio_ws and (esp32_audio_ws.client_state == WebSocketState.CONNECTED):
            await esp32_audio_ws.send_text("STOP")
    except Exception:
        pass

    # 等待音频客户端连接 + 预缓冲
    # 教学说明：每次 AI 播完后会关闭 HTTP 流连接（发 None）
    # ESP32 检测到断开后约 200ms 重连（已优化），所以等待时间很短
    from audio_stream import stream_clients
    if len(stream_clients) > 0:
        # 已有连接，直接发预缓冲
        await send_silence_prebuffer(duration_ms=60)
        print(f"[TIMING] 准备阶段 {time.time()-_t_setup:.3f}s (流已连接)", flush=True)
    elif esp32_audio_ws and (esp32_audio_ws.client_state == WebSocketState.CONNECTED):
        # ESP32 WebSocket 在线但 /stream.wav 还没重连，等一下
        _t_wait = time.time()
        if await wait_for_stream_client(timeout=1.5):
            await send_silence_prebuffer(duration_ms=60)
            print(f"[TIMING] 准备阶段 {time.time()-_t_setup:.3f}s (等流重连 {time.time()-_t_wait:.3f}s)", flush=True)
        else:
            print(f"[TIMING] 准备阶段 {time.time()-_t_setup:.3f}s (等流超时!)", flush=True)
    else:
        print(f"[TIMING] 准备阶段 {time.time()-_t_setup:.3f}s (无ESP32)", flush=True)
    # 没有 ESP32 连接时直接跳过，不等待

    # 创建异步任务，并通过 setter 函数注册到 audio_stream 模块
    # 教学说明：audio_stream 需要知道当前 AI 任务，以便 cancel_current_ai() 能取消它
    loop = asyncio.get_running_loop()
    task = loop.create_task(_runner())
    set_current_ai_task(task)


# ===== HTTP 路由 =====

@app.get("/", response_class=HTMLResponse)
def root():
    """主页：返回 Web 控制面板"""
    with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/health", response_class=PlainTextResponse)
def health():
    return "OK"


# 注册 /stream.wav 音频流路由
register_stream_route(app)


# ===== WebSocket：Web UI =====

@app.websocket("/ws_ui")
async def ws_ui(ws: WebSocket):
    """Web UI WebSocket：推送 ASR/AI 状态"""
    await ws.accept()
    ui_clients[id(ws)] = ws
    try:
        # 发送初始状态
        init = {"partial": current_partial, "finals": recent_finals[-10:]}
        await ws.send_text("INIT:" + json.dumps(init, ensure_ascii=False))
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=60)
                # 目前 UI 不需要发命令，保留扩展接口
            except asyncio.TimeoutError:
                continue
    except WebSocketDisconnect:
        pass
    finally:
        ui_clients.pop(id(ws), None)


# ===== WebSocket：ESP32 音频 =====

@app.websocket("/ws_audio")
async def ws_audio(ws: WebSocket):
    """
    ESP32 音频 WebSocket：接收麦克风数据 + 发送控制命令。

    教学说明（通信协议）：
    - ESP32 → Python（文本）：START / STOP / PROMPT:xxx
    - ESP32 → Python（二进制）：PCM 音频帧（20ms 一帧）
    - Python → ESP32（文本）：OK:STARTED / OK:STOPPED / RESET / START / STOP
    """
    global esp32_audio_ws
    esp32_audio_ws = ws
    await ws.accept()
    print("\n[AUDIO] ESP32 已连接", flush=True)

    recognition = None
    streaming = False
    last_ts = time.monotonic()
    keepalive_task: Optional[asyncio.Task] = None

    async def stop_rec(send_notice: Optional[str] = None):
        """停止 ASR 识别"""
        nonlocal recognition, streaming, keepalive_task
        if keepalive_task and not keepalive_task.done():
            keepalive_task.cancel()
            try:
                await keepalive_task
            except (asyncio.CancelledError, Exception):
                pass
        keepalive_task = None
        if recognition:
            try:
                recognition.stop()
            except Exception:
                pass
            recognition = None
        await set_current_recognition(None)
        streaming = False
        if send_notice:
            try:
                await ws.send_text(send_notice)
            except Exception:
                pass

    async def on_sdk_error(_msg: str):
        await stop_rec(send_notice="RESTART")

    async def keepalive_loop():
        """
        保活循环：ASR 引擎如果长时间没收到音频会断开。
        定期发送静音帧保持连接。
        """
        nonlocal last_ts, recognition, streaming
        try:
            while streaming and recognition is not None:
                idle = time.monotonic() - last_ts
                if idle > 0.35:
                    try:
                        for _ in range(30):
                            recognition.send_audio_frame(SILENCE_20MS)
                        last_ts = time.monotonic()
                    except Exception:
                        await on_sdk_error("keepalive send failed")
                        return
                await asyncio.sleep(0.10)
        except asyncio.CancelledError:
            return

    try:
        while True:
            if ws.client_state != WebSocketState.CONNECTED:
                break
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                if "Cannot call \"receive\"" in str(e):
                    break
                raise

            # ===== 文本消息处理 =====
            if "text" in msg and msg["text"] is not None:
                raw = (msg["text"] or "").strip()
                cmd = raw.upper()

                if cmd == "START":
                    # ESP32 请求开始 ASR
                    print("[AUDIO] 收到 START", flush=True)
                    await stop_rec()

                    if not asr_engine:
                        await ws.send_text("ERR:ASR_NOT_INIT")
                        continue

                    # 清空 UI 状态
                    await ui_broadcast_partial("")
                    global current_partial
                    current_partial = ""

                    await asyncio.sleep(0.3)

                    # 冷却期：防止识别到 AI 残留音频
                    # 教学说明：ASR 重启后前几秒内的识别结果全部丢弃
                    # 因为这段时间内麦克风可能还在拾取喇叭的回声
                    # 3 秒足够回声消散（RobotDuck 验证过，5 秒太保守）
                    session_start_time = time.time()
                    cooldown_seconds = 3.0
                    last_final_text = ""

                    main_loop = asyncio.get_running_loop()

                    # ASR 回调（在后台线程中执行）
                    def on_partial_sync(text: str):
                        async def _on_partial(text: str):
                            if time.time() - session_start_time < cooldown_seconds:
                                return
                            if is_playing_now():
                                return
                            global current_partial
                            current_partial = text
                            await ui_broadcast_partial(text)

                        try:
                            asyncio.run_coroutine_threadsafe(_on_partial(text), main_loop)
                        except Exception as e:
                            print(f"[ASR] on_partial 错误: {e}", flush=True)

                    def on_final_sync(text: str):
                        async def _on_final(text: str):
                            nonlocal last_final_text
                            if time.time() - session_start_time < cooldown_seconds:
                                print(f"[ASR] 冷却期内，忽略: {text}", flush=True)
                                return
                            if is_playing_now():
                                return
                            if text == last_final_text:
                                return
                            last_final_text = text
                            if not text or not text.strip():
                                return

                            # 热词检查（打断）
                            text_lower = text.strip().lower()
                            hotwords = {"停下", "别说了", "停止"}
                            if any(hw in text_lower for hw in hotwords):
                                print(f"[ASR] 热词触发重置: {text}", flush=True)
                                await full_system_reset("Hotword interrupt")
                                return

                            await ui_broadcast_final(text)
                            print(f"[ASR FINAL] {text}", flush=True)

                            async with interrupt_lock:
                                await start_ai_with_text(text)

                        try:
                            asyncio.run_coroutine_threadsafe(_on_final(text), main_loop)
                        except Exception as e:
                            print(f"[ASR] on_final 错误: {e}", flush=True)

                    # 启动 ASR
                    recognition = asr_engine.start(
                        on_partial=on_partial_sync,
                        on_final=on_final_sync,
                    )
                    await set_current_recognition(recognition)
                    streaming = True
                    last_ts = time.monotonic()
                    keepalive_task = asyncio.create_task(keepalive_loop())
                    await ui_broadcast_partial("（已开始接收音频…）")
                    await ws.send_text("OK:STARTED")

                elif cmd == "STOP":
                    if recognition:
                        for _ in range(15):
                            try:
                                recognition.send_audio_frame(SILENCE_20MS)
                            except Exception:
                                break
                    await stop_rec(send_notice="OK:STOPPED")

                elif raw.startswith("PROMPT:"):
                    # ESP32 直接发文本触发 AI
                    text = raw[len("PROMPT:"):].strip()
                    if text:
                        async with interrupt_lock:
                            await start_ai_with_text(text)
                        await ws.send_text("OK:PROMPT_ACCEPTED")
                    else:
                        await ws.send_text("ERR:EMPTY_PROMPT")

            # ===== 二进制消息（音频数据）=====
            elif "bytes" in msg and msg["bytes"] is not None:
                audio_bytes = msg["bytes"]

                # AI 正在播放时，丢弃上行音频（避免录到喇叭回声）
                if is_playing_now():
                    continue

                # 正常模式：送入 ASR
                if streaming and recognition:
                    try:
                        asr_engine.send_audio_frame(audio_bytes)
                        last_ts = time.monotonic()
                    except Exception:
                        await on_sdk_error("send_audio_frame failed")

                # 同时存入滚动缓冲区（即时克隆用）
                # 教学说明：不论 ASR 是否在运行，只要不是 AI 播放期间
                # 就把用户的语音存下来。这样说"克隆我的声音"时直接有素材用
                _clone_rolling_buf.append(audio_bytes)
                _clone_rolling_total_delta = len(audio_bytes)
                global _clone_rolling_total
                _clone_rolling_total += _clone_rolling_total_delta
                # 超出上限时，从头部删除最老的帧
                while _clone_rolling_total > _CLONE_BUF_MAX and _clone_rolling_buf:
                    removed = _clone_rolling_buf.popleft()  # O(1)，不再移动整个数组
                    _clone_rolling_total -= len(removed)

    except Exception as e:
        print(f"\n[WS ERROR] {e}")
    finally:
        await stop_rec()
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        if esp32_audio_ws is ws:
            esp32_audio_ws = None
        print("[WS] ESP32 连接已断开")


# ===== Web 聊天接口（手动输入文字触发）=====

@app.get("/api/chat")
async def api_chat(request: Request):
    """
    Web 聊天接口：通过 URL 参数发送文字消息。
    用于测试，不需要 ESP32 也能对话。
    用法：GET /api/chat?text=你好
    """
    text = request.query_params.get("text", "").strip()
    if not text:
        return PlainTextResponse("参数 text 不能为空", status_code=400)

    async with interrupt_lock:
        await start_ai_with_text(text)
    return PlainTextResponse("OK")


# ===== 启动入口 =====
# 教学说明：启动逻辑已移到上方的 lifespan() 上下文管理器中

if __name__ == "__main__":
    # 教学说明：必须关闭 WebSocket 心跳 ping！
    # uvicorn 默认每 20 秒发 ping，ESP32 忙着拉 HTTP 音频流，来不及回 pong
    # → 被判定超时 → WebSocket 断连 → ASR 中断 → 反复重连 → 一堆错误
    # ws_ping_interval=None 禁用 ping，ESP32 就不会被无故踢掉了
    uvicorn.run(app, host="0.0.0.0", port=8081,
                ws_ping_interval=None, ws_ping_timeout=None)
