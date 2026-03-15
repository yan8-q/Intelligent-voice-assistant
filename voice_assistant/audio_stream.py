# audio_stream.py — 音频流广播系统
# 职责：把 TTS 生成的 PCM 音频实时推送给所有连接的 ESP32
# 教学重点：ESP32 通过 HTTP GET /stream.wav 拉取音频流（而不是 WebSocket 推送）
#
# 为什么用 HTTP 拉流而不是 WebSocket 推流？
# - HTTP 拉流：ESP32 主动请求，断线重连简单，兼容性好
# - WebSocket 推流：需要服务端管理连接状态，ESP32 缓冲区容易溢出
# - 结论：拉流更稳定，RobotDuck 项目验证过

import asyncio
from dataclasses import dataclass
from typing import Optional, Set, List

from fastapi import Request
from fastapi.responses import StreamingResponse

# ===== 音频流基础参数 =====
STREAM_SR = 16000   # 采样率：必须与 ESP32 的 I2S 输出采样率一致
STREAM_CH = 1       # 声道数：单声道
STREAM_SW = 2       # 采样宽度：16bit = 2 bytes

# ===== 发送节拍配置 =====
# 教学说明：音频是"连续的"，但网络传输是"分块的"
# 每 10ms 发送一小块音频数据，模拟实时流
STREAM_TICK_MS = 10  # 发送间隔（毫秒）
BYTES_PER_TICK = STREAM_SR * STREAM_SW * STREAM_TICK_MS // 1000  # 10ms@16kHz = 320 bytes

# 兼容旧代码的 20ms 常量
BYTES_PER_20MS_16K = STREAM_SR * STREAM_SW * 20 // 1000  # 640 bytes

# ===== AI 播放任务管理 =====
# 教学说明：同一时间只能有一个 AI 回复在播放
# 新的回复开始前，必须取消旧的播放任务
current_ai_task: Optional[asyncio.Task] = None


def set_current_ai_task(task: Optional[asyncio.Task]):
    """设置当前 AI 播放任务（供 app_main.py 调用）"""
    # 教学说明：之前用 __dict__["current_ai_task"] = task 直接操作模块字典
    # 这是反模式——IDE 无法追踪、类型检查器看不到、搜索也搜不到
    # 改用 setter 函数后，代码可读性好，IDE 能正确跳转
    global current_ai_task
    current_ai_task = task


async def cancel_current_ai():
    """取消当前正在进行的 AI 语音播放任务"""
    global current_ai_task
    task = current_ai_task
    current_ai_task = None
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass


def is_playing_now() -> bool:
    """检查当前是否正在播放 AI 语音"""
    t = current_ai_task
    return (t is not None) and (not t.done())


# ===== /stream.wav 连接管理 =====

@dataclass(frozen=True)
class StreamClient:
    """
    一个音频流客户端（ESP32）的连接状态。
    - q: 音频数据队列，broadcast 往里放，HTTP 响应从里取
    - abort_event: 断开信号
    """
    q: asyncio.Queue
    abort_event: asyncio.Event


stream_clients: "Set[StreamClient]" = set()
STREAM_QUEUE_MAX = 32  # 队列最大长度（约 320ms 缓冲）


def _wav_header_unknown_size(sr=16000, ch=1, sw=2) -> bytes:
    """
    生成一个"无限长度"的 WAV 文件头。

    教学说明：
    - 正常 WAV 文件头包含数据总长度
    - 但我们是实时流，不知道总长度
    - 所以用一个超大的假长度（0x7FFFFFF0），ESP32 会一直读下去
    - 这是 HTTP 音频流的标准做法
    """
    import struct
    byte_rate = sr * ch * sw
    block_align = ch * sw
    data_size = 0x7FFFFFF0
    riff_size = 36 + data_size
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_size, b"WAVE",
        b"fmt ", 16,
        1, ch, sr, byte_rate, block_align, sw * 8,
        b"data", data_size
    )


async def hard_reset_audio(reason: str = "", keep_connections: bool = False):
    """
    一键清场：取消 AI 任务 + 可选断开音频连接。

    教学说明：
    - 用户说了新的话 → 需要打断当前播放 → 调用这个函数
    - keep_connections=True: 保留 ESP32 连接（避免重连延迟）
    - keep_connections=False: 全部断开重来
    """
    if not keep_connections:
        for sc in list(stream_clients):
            try:
                sc.abort_event.set()
            except Exception:
                pass
        stream_clients.clear()

    await cancel_current_ai()

    if reason:
        print(f"[HARD-RESET] {reason}")


async def wait_for_stream_client(timeout: float = 2.0) -> bool:
    """等待至少一个 ESP32 音频客户端连接"""
    import time
    start = time.time()
    while time.time() - start < timeout:
        if len(stream_clients) > 0:
            return True
        await asyncio.sleep(0.05)
    return False


async def send_silence_prebuffer(duration_ms: int = 40):
    """
    发送静音预缓冲，帮助 ESP32 同步 I2S 时钟。

    教学说明：
    - ESP32 的 I2S 播放需要先收到一些数据才能"启动"
    - 发送一小段静音让它准备好，后续音频就不会丢开头
    """
    silence_bytes = BYTES_PER_TICK * (duration_ms // STREAM_TICK_MS)
    if silence_bytes > 0:
        silence = b'\x00' * silence_bytes
        await broadcast_pcm16_realtime(silence)


async def broadcast_pcm16_realtime(pcm16: bytes):
    """
    实时广播 PCM16 音频数据给所有连接的 ESP32。

    教学说明：
    - 按 10ms 节拍分块发送（模拟实时）
    - 如果队列满了，丢弃最旧的数据（保持实时性）
    - 死掉的连接会被自动清理
    """
    loop = asyncio.get_event_loop()
    next_tick = loop.time()
    off = 0
    tick_sec = STREAM_TICK_MS / 1000.0

    while off < len(pcm16):
        take = min(BYTES_PER_TICK, len(pcm16) - off)
        piece = pcm16[off:off + take]

        # 发送给所有活跃的客户端
        dead: List[StreamClient] = []
        for sc in list(stream_clients):
            if sc.abort_event.is_set():
                dead.append(sc)
                continue
            try:
                if sc.q.full():
                    # 队列满了，丢弃最旧的数据
                    try:
                        sc.q.get_nowait()
                    except Exception:
                        pass
                sc.q.put_nowait(piece)
            except Exception:
                dead.append(sc)
        # 清理死掉的连接
        for sc in dead:
            try:
                stream_clients.discard(sc)
            except Exception:
                pass

        # 精确定时：按节拍发送
        next_tick += tick_sec
        now = loop.time()
        if now < next_tick:
            await asyncio.sleep(next_tick - now)
        else:
            next_tick = now
        off += take


# ===== FastAPI 路由注册 =====

def register_stream_route(app):
    """
    注册 /stream.wav 路由。

    教学说明（ESP32 如何获取音频）：
    1. ESP32 启动后，向 Python 服务发起 HTTP GET /stream.wav
    2. 服务端返回一个"无限长"的 WAV 流
    3. ESP32 的 I2S 播放器持续从这个流读取数据并播放
    4. 当有新的 AI 回复时，broadcast_pcm16_realtime() 往队列里放数据
    5. HTTP 响应从队列取数据发给 ESP32
    """

    @app.get("/stream.wav")
    async def stream_wav(_: Request):
        # 强制单连接：断开所有旧连接
        for sc in list(stream_clients):
            try:
                sc.abort_event.set()
            except Exception:
                pass
        stream_clients.clear()

        # 创建新连接
        q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=STREAM_QUEUE_MAX)
        abort_event = asyncio.Event()
        sc = StreamClient(q=q, abort_event=abort_event)
        stream_clients.add(sc)

        async def gen():
            """
            生成器：先发 WAV 头，然后持续发音频数据。

            教学说明（连接复用优化）：
            - 收到 None 哨兵 = 当前 AI 回复播完
            - 旧方案：break 退出 → ESP32 检测断连 → 重新 GET /stream.wav → 200-500ms 延迟
            - 新方案：不退出，继续等待下一轮数据 → 连接复用 → 零重连延迟
            - 只有 abort_event（新连接抢占 / full_system_reset）才真正退出
            - 额外好处：不再触发 Content-Length 不匹配的 RuntimeError
            """
            yield _wav_header_unknown_size(STREAM_SR, STREAM_CH, STREAM_SW)
            try:
                while True:
                    if abort_event.is_set():
                        break
                    try:
                        chunk = await asyncio.wait_for(q.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    if abort_event.is_set():
                        break
                    if chunk is None:
                        # AI 播完了，但不断开连接 → 等下一轮复用
                        continue
                    if chunk:
                        yield chunk
            finally:
                stream_clients.discard(sc)

        # 教学重点：必须设置 Content-Length！
        # Starlette 源码逻辑：如果没有 Content-Length，就自动加 Transfer-Encoding: chunked
        # chunked 编码会在音频数据中插入分块头（如 "140\r\n"..."\r\n"）
        # ESP32 把这些分块头当成 PCM 数据播放 → 沙沙声/杂音
        # 设置一个超大的 Content-Length（≈2GB）后，服务器发送纯原始数据，不加 chunked
        return StreamingResponse(
            gen(),
            media_type="audio/wav",
            headers={"Content-Length": str(0x7FFFFFF0 + 44)},  # WAV数据长度 + 44字节头
        )
