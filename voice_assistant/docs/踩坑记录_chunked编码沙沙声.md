# ESP32 喇叭沙沙声 — 踩坑记录

## 问题现象
ESP32 通过 HTTP GET `/stream.wav` 拉取音频流播放时，喇叭有明显的**沙沙声/杂音**，说话声音不清晰。

## 根本原因：HTTP chunked 编码污染音频数据

### 正常情况
```
[WAV头44字节] [PCM音频数据连续不断...]
```
ESP32 收到的每个字节都是音频采样值，直接送 I2S 播放 → 声音清晰

### 实际发生了什么
Python 后端用 FastAPI/Starlette 的 `StreamingResponse`，**如果不设置 `Content-Length` 头**，
框架会自动加上 `Transfer-Encoding: chunked`。

chunked 编码把数据切块，**每块前面插入十六进制长度标记**：
```
140\r\n                    ← chunked分块头（表示下面有320字节）
[320字节PCM音频数据]
\r\n                       ← 分块结束符
140\r\n                    ← 又一个分块头
[320字节PCM音频数据]
\r\n
...
```

### 为什么会沙沙声
ESP32 的 HTTP 客户端**不懂 chunked 协议**，把所有收到的字节都当成音频数据：
```
"140\r\n" → 被当成PCM采样值 → 产生噪音脉冲
[正常音频]  → 正常播放
"\r\n"     → 又一个噪音脉冲
```
每隔约320字节就混入几个垃圾字节 → **连续的细碎噪音 = 沙沙声**

## 修复方法

在 `audio_stream.py` 的 `StreamingResponse` 中设置一个超大的 `Content-Length`：

```python
# ❌ 修复前：无Content-Length → Starlette自动加chunked → 沙沙声
return StreamingResponse(gen(), media_type="audio/wav")

# ✅ 修复后：有Content-Length → Starlette不加chunked → 纯净音频
return StreamingResponse(
    gen(),
    media_type="audio/wav",
    headers={"Content-Length": str(0x7FFFFFF0 + 44)},  # WAV数据长度 + 44字节头
)
```

### 原理
- Starlette 源码逻辑：没有 `Content-Length` → 自动加 `Transfer-Encoding: chunked`
- 设置一个超大的 `Content-Length`（约2GB）→ Starlette 认为长度已知 → 不加 chunked
- ESP32 收到的就是纯原始 WAV/PCM 数据，没有任何分块头污染

## 验证方法
用 raw socket 检查 HTTP 响应头：
```python
import socket
s = socket.socket()
s.connect(('127.0.0.1', 8081))
s.sendall(b'GET /stream.wav HTTP/1.1\r\nHost: 127.0.0.1:8081\r\n\r\n')
data = s.recv(512)
s.close()
print(data.decode('latin-1'))
```
- 看到 `content-length: 2147483676` 且**没有** `transfer-encoding: chunked` → 修复成功
- 看到 `transfer-encoding: chunked` → 还有问题

## 通用教训
任何通过 HTTP `StreamingResponse` 向 ESP32 推送二进制流（音频、固件等）的场景，
都必须设置 `Content-Length` 头来避免 chunked 编码。这是 ESP32 HTTP 音频流的通用注意事项。
