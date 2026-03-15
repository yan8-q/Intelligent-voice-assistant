// ===== voice_assistant.ino — ESP32-WROOM-32 智能语音助手固件 =====
// 功能：WiFi + WebSocket 音频上行 + HTTP 音频下行播放
//
// 教学说明（整体架构）：
// ┌─────────────────────────────────────────────┐
// │  ESP32-WROOM-32  (双核)                      │
// │                                              │
// │  Core 1 (主循环):                            │
// │    WebSocket 通信 + 麦克风采集上传            │
// │                                              │
// │  Core 0 (HTTP读取任务):                      │
// │    HTTP GET /stream.wav → 环形缓冲区         │
// │                                              │
// │  Core 1 (I2S播放任务):                       │
// │    环形缓冲区 → mono16转stereo32 → I2S播放   │
// └─────────────────────────────────────────────┘
//
// 参考：RobotDuck 项目的双核 + 环形缓冲架构
// 硬件接线：
// INMP441:   SCK=GPIO14, WS=GPIO15, SD=GPIO32
// MAX98357A: BCLK=GPIO26, LRC=GPIO27, DIN=GPIO25

#include <WiFi.h>
#include <WebSocketsClient.h>  // 需要安装 WebSockets by Markus Sattler
#include <driver/i2s.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>
#include <math.h>

// ====================================================================
// WiFi 配置（修改为你的网络）
// ====================================================================
const char* WIFI_SSID    = "啵啵鱼";
const char* WIFI_PASS    = "88888888";
const char* SERVER_HOST  = "172.20.10.2";
const uint16_t SERVER_PORT = 8081;

// ====================================================================
// I2S 引脚定义（你的接线方案）
// ====================================================================

// INMP441 麦克风（I2S 输入）
#define I2S_MIC_SCK   14    // GPIO14 → INMP441 SCK
#define I2S_MIC_WS    15    // GPIO15 → INMP441 WS
#define I2S_MIC_SD    32    // GPIO32 → INMP441 SD

// MAX98357A 功放（I2S 输出）
#define I2S_SPK_BCLK  26   // GPIO26 → MAX98357A BCLK
#define I2S_SPK_LRC   27   // GPIO27 → MAX98357A LRC
#define I2S_SPK_DIN   25   // GPIO25 → MAX98357A DIN

// ====================================================================
// 音频参数
// ====================================================================
#define SAMPLE_RATE     16000   // 采样率 16kHz
#define CHUNK_MS        20      // 麦克风每帧 20ms
#define BYTES_PER_CHUNK (SAMPLE_RATE * CHUNK_MS / 1000 * 2)  // 640 bytes

// ====================================================================
// 环形缓冲区（参考 RobotDuck 架构）
// ====================================================================
// 教学说明：
// HTTP 读取和 I2S 播放在不同的 FreeRTOS 任务中运行
// 环形缓冲区是它们之间的"管道"：
//   HTTP任务 → 写入缓冲区 → I2S任务从缓冲区读取
// 好处：HTTP 网络抖动不会直接影响 I2S 播放连续性

#define RING_BUF_SIZE    16384  // 16KB ≈ 512ms @16kHz mono16（从8KB扩大，吸收TTS分块间隙）
#define AUDIO_CHUNK_SIZE 320    // 每块 10ms @16kHz = 320 字节
#define PREBUFFER_MS     200    // 🆕 预缓冲从100ms提高到200ms，吸收TTS分块间隙（200-500ms）

// 环形缓冲区结构体
typedef struct {
  uint8_t data[RING_BUF_SIZE];  // 数据存储区
  volatile size_t writePos;      // 写指针
  volatile size_t readPos;       // 读指针
  volatile size_t fillLevel;     // 当前填充量
  SemaphoreHandle_t mutex;       // 互斥锁（线程安全）
} AudioRingBuffer;

AudioRingBuffer ringBuf;         // 全局环形缓冲区实例

// 初始化环形缓冲区
void ringBufInit() {
  memset(ringBuf.data, 0, RING_BUF_SIZE);
  ringBuf.writePos = 0;
  ringBuf.readPos = 0;
  ringBuf.fillLevel = 0;
  ringBuf.mutex = xSemaphoreCreateMutex();
}

// 写入环形缓冲区（HTTP任务调用）
size_t ringBufWrite(const uint8_t* src, size_t len) {
  if (xSemaphoreTake(ringBuf.mutex, pdMS_TO_TICKS(10)) != pdTRUE) return 0;
  size_t space = RING_BUF_SIZE - ringBuf.fillLevel;
  size_t toWrite = (len < space) ? len : space;  // 不超过可用空间
  for (size_t i = 0; i < toWrite; i++) {
    ringBuf.data[ringBuf.writePos] = src[i];
    ringBuf.writePos = (ringBuf.writePos + 1) % RING_BUF_SIZE;
  }
  ringBuf.fillLevel += toWrite;
  xSemaphoreGive(ringBuf.mutex);
  return toWrite;
}

// 从环形缓冲区读取（I2S任务调用）
size_t ringBufRead(uint8_t* dst, size_t len) {
  if (xSemaphoreTake(ringBuf.mutex, pdMS_TO_TICKS(10)) != pdTRUE) return 0;
  size_t avail = ringBuf.fillLevel;
  size_t toRead = (len < avail) ? len : avail;
  for (size_t i = 0; i < toRead; i++) {
    dst[i] = ringBuf.data[ringBuf.readPos];
    ringBuf.readPos = (ringBuf.readPos + 1) % RING_BUF_SIZE;
  }
  ringBuf.fillLevel -= toRead;
  xSemaphoreGive(ringBuf.mutex);
  return toRead;
}

// 获取缓冲区当前填充量（加 mutex 保护）
// 教学说明：volatile 只防编译器优化，不保证双核下的原子性
// Core 0 写 fillLevel 和 Core 1 读 fillLevel 可能交叉 → 读到脏数据
// 用 mutex 保护后读取是原子的；获取锁失败返回 0（安全兜底，等同于"缓冲区空"）
size_t ringBufLevel() {
  if (xSemaphoreTake(ringBuf.mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
    size_t level = ringBuf.fillLevel;
    xSemaphoreGive(ringBuf.mutex);
    return level;
  }
  return 0;  // 获取锁失败，返回 0 让调用方认为缓冲区空（安全兜底）
}

// 清空缓冲区
void ringBufClear() {
  if (xSemaphoreTake(ringBuf.mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
    ringBuf.writePos = 0;
    ringBuf.readPos = 0;
    ringBuf.fillLevel = 0;
    xSemaphoreGive(ringBuf.mutex);
  }
}

// ====================================================================
// mono16 → stereo32 转换（参考 RobotDuck）
// ====================================================================
// 教学说明：
// MAX98357A 在 I2S 标准模式下期望 32bit 立体声帧
// 我们的 PCM 数据是 16bit 单声道
// 转换方法：每个 16bit 采样 → 左移16位变成 32bit → 复制到左右声道
// 加 0.7 增益防止削波失真

static inline void mono16_to_stereo32(const int16_t* in, size_t nSamp,
                                       int32_t* outLR, int gain256 = 179) {
  // 教学说明（整数近似浮点，嵌入式常见优化技巧）：
  // 0.7 的增益用整数近似：乘以 179 再右移 8 位（÷256）
  // 179/256 = 0.69921875 ≈ 0.7，误差 < 0.12%，人耳完全听不出
  // 整数乘法+移位比浮点乘法快 3-5 倍（ESP32 的 FPU 不擅长高频调用）
  // gain256 参数：传 256 = 增益1.0，传 179 = 增益0.7
  for (size_t i = 0; i < nSamp; i++) {
    int32_t s = ((int32_t)in[i] * gain256) >> 8;
    // 无需限幅：int16 最大值 32767 × 256 = 8388352 >> 8 = 32767，不会溢出
    int32_t v32 = s << 16;          // 放到高 16 位（MSB对齐）
    outLR[i * 2]     = v32;         // 左声道
    outLR[i * 2 + 1] = v32;         // 右声道（复制）
  }
}

// ====================================================================
// 全局状态
// ====================================================================
WebSocketsClient webSocket;

bool wsConnected   = false;
bool micStreaming   = false;

// 回声抑制（AEC）
// 教学说明：喇叭播放时暂停麦克风上传，防止 ASR 识别到 AI 自己的声音
volatile bool spkHasData     = false;
volatile unsigned long spkLastDataMs = 0;
// 回声抑制静默保护时间
// 教学说明：喇叭停止后，等多久才让麦克风恢复上传
// 原来 1500ms 太长！因为还有多层保护：
//   1. Python 端 "STOP/START" 命令控制 micStreaming 开关
//   2. Python 端 post-play 等待 0.5 秒
//   3. Python 端 ASR 冷却期 3.0 秒（丢弃残余音频识别结果）
// 所以固件端只需要很短的保护（300ms 足够声波衰减）
// RobotDuck 甚至没有这个保护，完全靠 Python 冷却期
#define SPK_SILENCE_GUARD_MS 300

// HTTP 音频流任务控制
volatile bool httpTaskRunning  = false;
volatile bool i2sPlayRunning   = false;
volatile bool needPrebuffer    = true;   // 需要预缓冲

// 麦克风缓冲
uint8_t micBuffer[1024];

// ====================================================================
// I2S 初始化
// ====================================================================

void setup_i2s_mic() {
  // I2S0：麦克风输入（16bit 单声道）
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate = SAMPLE_RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 4;
  cfg.dma_buf_len = 512;
  cfg.use_apll = false;

  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);

  i2s_pin_config_t pins = {};
  pins.mck_io_num   = I2S_PIN_NO_CHANGE;
  pins.bck_io_num   = I2S_MIC_SCK;
  pins.ws_io_num    = I2S_MIC_WS;
  pins.data_in_num  = I2S_MIC_SD;
  pins.data_out_num = I2S_PIN_NO_CHANGE;

  i2s_set_pin(I2S_NUM_0, &pins);
  i2s_zero_dma_buffer(I2S_NUM_0);
  Serial.println("[I2S] 麦克风初始化完成 (I2S0, 16bit mono)");
}

void setup_i2s_speaker() {
  // 教学说明（关键改动！参考 RobotDuck）：
  // 用 32bit 立体声模式，而不是 16bit 单声道
  // 原因：MAX98357A 在标准 I2S 模式下期望 32bit 帧格式
  // 16bit 单声道虽然能发声，但数据对齐可能不稳定，导致杂音
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX);
  cfg.sample_rate = SAMPLE_RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;       // ★ 32bit（原来是16bit）
  cfg.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;        // ★ 立体声（原来是ONLY_LEFT）
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = 512;
  cfg.use_apll = true;              // 🆕 开启APLL精确时钟，减少I2S时钟抖动导致的底噪
  cfg.tx_desc_auto_clear = true;

  i2s_driver_install(I2S_NUM_1, &cfg, 0, NULL);

  i2s_pin_config_t pins = {};
  pins.mck_io_num   = I2S_PIN_NO_CHANGE;
  pins.bck_io_num   = I2S_SPK_BCLK;
  pins.ws_io_num    = I2S_SPK_LRC;
  pins.data_out_num = I2S_SPK_DIN;
  pins.data_in_num  = I2S_PIN_NO_CHANGE;

  i2s_set_pin(I2S_NUM_1, &pins);
  i2s_zero_dma_buffer(I2S_NUM_1);
  Serial.println("[I2S] 喇叭初始化完成 (I2S1, 32bit stereo)");
}

// ====================================================================
// 开机提示音（适配 32bit 立体声）
// ====================================================================

void play_boot_beep() {
  const int freq = 800;
  const int duration_ms = 1000;
  const int num_samples = SAMPLE_RATE * duration_ms / 1000;
  const int chunk_size = 128;  // 每次处理 128 个采样
  int16_t mono[chunk_size];
  int32_t stereo[chunk_size * 2];  // 立体声 = 2倍

  Serial.println("[BEEP] 播放开机提示音（1秒）...");

  for (int i = 0; i < num_samples; i += chunk_size) {
    int count = min(chunk_size, num_samples - i);
    // 生成正弦波（16bit mono）
    for (int j = 0; j < count; j++) {
      float t = (float)(i + j) / SAMPLE_RATE;
      mono[j] = (int16_t)(16000 * sin(2.0 * PI * freq * t));
    }
    // 转换为 32bit stereo
    mono16_to_stereo32(mono, count, stereo, 256);  // gain=1.0 → 256/256
    // 写入 I2S
    size_t bw = 0;
    i2s_write(I2S_NUM_1, stereo, count * 2 * sizeof(int32_t), &bw, portMAX_DELAY);
  }

  delay(100);
  i2s_zero_dma_buffer(I2S_NUM_1);
  Serial.println("[BEEP] 提示音播放完成");
}

// ====================================================================
// WiFi 连接
// ====================================================================

void setup_wifi() {
  Serial.printf("[WiFi] 连接 %s ...\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  int retry = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    retry++;
    if (retry > 30) {
      Serial.println("\n[WiFi] 连接超时，重启...");
      ESP.restart();
    }
  }

  Serial.printf("\n[WiFi] 已连接！IP: %s\n", WiFi.localIP().toString().c_str());
}

// ====================================================================
// WebSocket 回调
// ====================================================================

void webSocketEvent(WStype_t type, uint8_t* payload, size_t length) {
  switch (type) {
    case WStype_DISCONNECTED:
      Serial.println("[WS] 已断开");
      wsConnected = false;
      micStreaming = false;
      break;

    case WStype_CONNECTED:
      Serial.printf("[WS] 已连接: %s\n", (char*)payload);
      wsConnected = true;
      webSocket.sendTXT("START");
      break;

    case WStype_TEXT: {
      String msg = String((char*)payload);
      Serial.printf("[WS] 收到: %s\n", msg.c_str());

      if (msg == "OK:STARTED") {
        micStreaming = true;
        Serial.println("[MIC] 开始上传音频");
      }
      else if (msg == "OK:STOPPED") {
        micStreaming = false;
        Serial.println("[MIC] 停止上传音频");
      }
      else if (msg == "START") {
        // 服务器要求恢复 ASR → 重启麦克风上传
        // delay(20) 足够让 I2S 麦克风稳定（RobotDuck 验证过）
        // 原来 delay(500) 太长，每轮白白浪费 480ms
        micStreaming = false;
        delay(20);
        webSocket.sendTXT("START");
      }
      else if (msg == "STOP") {
        micStreaming = false;
      }
      else if (msg == "RESET") {
        micStreaming = false;
        ringBufClear();  // 清空音频缓冲
        Serial.println("[SYS] 收到重置命令");
      }
      else if (msg == "RESTART") {
        // ASR SDK 出错后重启（delay(50) 足够，RobotDuck 验证过）
        micStreaming = false;
        delay(50);
        webSocket.sendTXT("START");
      }
      break;
    }

    default:
      break;
  }
}

// ====================================================================
// HTTP 音频流读取任务（运行在 Core 0）
// ====================================================================
// 教学说明（参考 RobotDuck 双核架构）：
// 这个任务专门负责从 Python 后端拉取 TTS 音频数据
// 读到的数据写入环形缓冲区，由 I2S 播放任务从另一端取出播放
// 好处：网络抖动被缓冲区吸收，I2S 播放不受影响

void taskHttpRead(void* param) {
  Serial.println("[HTTP] 读取任务启动 (Core 0)");
  WiFiClient cli;
  uint8_t readBuf[640];  // 20ms @16kHz = 640 字节

  while (httpTaskRunning) {
    // ---- 建立连接 ----
    if (!cli.connected()) {
      Serial.printf("[HTTP] 连接 http://%s:%d/stream.wav ...\n", SERVER_HOST, SERVER_PORT);
      if (!cli.connect(SERVER_HOST, SERVER_PORT)) {
        Serial.println("[HTTP] 连接失败，1秒后重试");
        vTaskDelay(pdMS_TO_TICKS(1000));
        continue;
      }

      // 禁用 Nagle 算法：小包立即发送，ACK 更快返回
      // 教学说明：Nagle 算法会把小数据包合并后再发送（最多等200ms）
      // 对音频流来说，我们希望每个 ACK 尽快发出，让服务端更快送下一个包
      cli.setNoDelay(true);

      // 教学重点：用 HTTP/1.0 避免 chunked 编码！
      // HTTP/1.1 的 chunked 编码会在音频数据中插入分块头
      // ESP32 会把这些头当成 PCM 数据播放 → 杂音
      cli.printf("GET /stream.wav HTTP/1.0\r\n");
      cli.printf("Host: %s:%d\r\n", SERVER_HOST, SERVER_PORT);
      cli.printf("\r\n");
      Serial.println("[HTTP] 请求已发送");

      // ---- 跳过 HTTP 响应头 ----
      String headerLine;
      bool headerEnd = false;
      unsigned long t0 = millis();
      while (cli.connected() && !headerEnd && (millis() - t0 < 5000)) {
        if (cli.available()) {
          char c = cli.read();
          headerLine += c;
          if (headerLine.endsWith("\r\n\r\n")) {
            headerEnd = true;
          }
          if (headerLine.length() > 2048) {
            headerEnd = true;  // 防止无限积累
          }
        } else {
          vTaskDelay(pdMS_TO_TICKS(5));
        }
      }

      if (!headerEnd) {
        Serial.println("[HTTP] 未找到响应头结束标记");
        cli.stop();
        vTaskDelay(pdMS_TO_TICKS(1000));
        continue;
      }

      // ---- 跳过 WAV 文件头（44 字节）----
      int skipped = 0;
      t0 = millis();
      while (skipped < 44 && cli.connected() && (millis() - t0 < 3000)) {
        if (cli.available()) {
          cli.read();
          skipped++;
        } else {
          vTaskDelay(pdMS_TO_TICKS(5));
        }
      }

      if (skipped < 44) {
        Serial.println("[HTTP] WAV 头跳过失败");
        cli.stop();
        vTaskDelay(pdMS_TO_TICKS(1000));
        continue;
      }

      // 清空旧数据，标记需要预缓冲
      ringBufClear();
      needPrebuffer = true;
      Serial.println("[HTTP] 头部已跳过，开始接收音频数据");
    }

    // ---- 读取音频数据并写入环形缓冲区 ----
    int avail = cli.available();
    if (avail > 0) {
      // 确保读取偶数字节（16bit PCM = 2 字节一个采样）
      int toRead = min(avail, (int)sizeof(readBuf));
      toRead = toRead & ~1;  // 向下取偶数
      if (toRead > 0) {
        int got = cli.read(readBuf, toRead);
        if (got > 0) {
          ringBufWrite(readBuf, got);
        }
      }
    } else if (!cli.connected()) {
      // 连接断开，准备重连（200ms快速重连，减少每轮对话延迟）
      Serial.println("[HTTP] 连接断开");
      cli.stop();
      vTaskDelay(pdMS_TO_TICKS(200));
      continue;
    }

    // 让出 CPU（1ms 足够，不能太长否则缓冲区会饿）
    vTaskDelay(pdMS_TO_TICKS(1));
  }

  cli.stop();
  Serial.println("[HTTP] 读取任务结束");
  vTaskDelete(NULL);
}

// ====================================================================
// I2S 播放任务（运行在 Core 1）
// ====================================================================
// 教学说明（参考 RobotDuck 双核架构）：
// 这个任务从环形缓冲区取出 PCM 数据
// 转换成 32bit 立体声格式后写入 I2S 播放
// 和 HTTP 读取任务完全解耦，即使网络卡顿也不会影响播放连续性

void taskI2SPlay(void* param) {
  Serial.println("[I2S-PLAY] 播放任务启动 (Core 1)");

  // 缓冲区：160 字节 mono16 = 80 采样 = 5ms
  // 转换后：80 × 2(声道) × 4(字节) = 640 字节 stereo32
  uint8_t inBuf[320];           // 输入：mono16 (10ms)
  int32_t outLR[160 * 2];      // 输出：stereo32

  while (i2sPlayRunning) {
    // ---- 预缓冲等待 ----
    // 教学说明：先攒够一定量的数据再开始播放
    // 防止刚开始播放就缓冲区空了，产生爆音/卡顿
    if (needPrebuffer) {
      size_t prebufBytes = SAMPLE_RATE * 2 * PREBUFFER_MS / 1000;  // 200ms = 6400 字节
      if (ringBufLevel() < prebufBytes) {
        vTaskDelay(pdMS_TO_TICKS(5));
        continue;
      }
      needPrebuffer = false;
      Serial.printf("[I2S-PLAY] 预缓冲完成，缓冲区: %d 字节\n", ringBufLevel());
    }

    // ---- 从环形缓冲区读取 mono16 数据 ----
    // 每次读 5ms = 160 字节（保持低延迟）
    size_t bytesPerChunk = SAMPLE_RATE * 2 * 5 / 1000;  // 5ms = 160 字节
    if (bytesPerChunk > sizeof(inBuf)) bytesPerChunk = sizeof(inBuf);

    size_t got = ringBufRead(inBuf, bytesPerChunk);

    if (got > 0) {
      // 更新回声抑制标志
      // 教学说明：检查音频数据是否有声音（非静音）
      bool hasSound = false;
      size_t nSamp = got / 2;
      int16_t* samples = (int16_t*)inBuf;
      for (size_t i = 0; i < nSamp && !hasSound; i++) {
        if (samples[i] > 50 || samples[i] < -50) {
          hasSound = true;
        }
      }
      if (hasSound) {
        spkHasData = true;
        spkLastDataMs = millis();
      }

      // ---- mono16 → stereo32 转换 ----
      mono16_to_stereo32(samples, nSamp, outLR);  // 默认 gain256=179 ≈ 0.7

      // ---- 写入 I2S ----
      size_t totalBytes = nSamp * 2 * sizeof(int32_t);
      size_t offset = 0;
      while (offset < totalBytes && i2sPlayRunning) {
        size_t written = 0;
        i2s_write(I2S_NUM_1, (uint8_t*)outLR + offset, totalBytes - offset, &written, pdMS_TO_TICKS(30));  // 🆕 从20ms提到30ms，给DMA更多时间消化数据
        if (written > 0) {
          offset += written;
        } else {
          vTaskDelay(pdMS_TO_TICKS(1));
        }
      }
    } else {
      // 缓冲区空，检查是否需要清除播放标志
      if (spkHasData && (millis() - spkLastDataMs > 200)) {
        spkHasData = false;
      }
      // 🆕 缓冲欠载时写入静音帧到I2S，防止DMA饿死产生爆音/咔嗒声
      // 教学说明：I2S DMA 需要持续供数据，如果什么都不写，DMA缓冲耗尽后
      // 喇叭会产生随机噪声或咔嗒声。写入静音帧（全0）保证安静等待。
      if (spkHasData) {
        // 正在播放中但暂时缺数据 → 写 5ms 静音帧填充
        memset(inBuf, 0, bytesPerChunk);
        size_t nSamp = bytesPerChunk / 2;
        mono16_to_stereo32((int16_t*)inBuf, nSamp, outLR);
        size_t totalBytes = nSamp * 2 * sizeof(int32_t);
        size_t written = 0;
        i2s_write(I2S_NUM_1, (uint8_t*)outLR, totalBytes, &written, pdMS_TO_TICKS(20));
      } else {
        vTaskDelay(pdMS_TO_TICKS(2));  // 未在播放状态，短暂等待新数据
      }
    }
  }

  Serial.println("[I2S-PLAY] 播放任务结束");
  vTaskDelete(NULL);
}

// ====================================================================
// 麦克风采集 + WebSocket 上传
// ====================================================================

void process_mic_upload() {
  if (!micStreaming || !wsConnected) return;

  // 回声抑制：喇叭播放中或刚停止，不上传麦克风
  if (spkHasData) return;
  if (spkLastDataMs > 0 && (millis() - spkLastDataMs < SPK_SILENCE_GUARD_MS)) return;

  size_t bytesRead = 0;
  esp_err_t err = i2s_read(I2S_NUM_0, micBuffer, BYTES_PER_CHUNK, &bytesRead, 100 / portTICK_PERIOD_MS);

  if (err == ESP_OK && bytesRead > 0) {
    webSocket.sendBIN(micBuffer, bytesRead);
  }
}

// ====================================================================
// Arduino 主函数
// ====================================================================

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n============================");
  Serial.println("  智能语音助手 ESP32 固件");
  Serial.println("  (双核 + 环形缓冲架构)");
  Serial.println("============================\n");

  // 1. 连接 WiFi
  setup_wifi();

  // 2. 初始化 I2S
  setup_i2s_mic();
  setup_i2s_speaker();

  // 3. 开机提示音
  play_boot_beep();

  // 4. 初始化环形缓冲区
  ringBufInit();

  // 5. 连接 WebSocket
  Serial.printf("[WS] 连接 ws://%s:%d/ws_audio ...\n", SERVER_HOST, SERVER_PORT);
  webSocket.begin(SERVER_HOST, SERVER_PORT, "/ws_audio");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(3000);

  // 6. 启动 HTTP 音频流读取任务（Core 0）
  // 教学说明：xTaskCreatePinnedToCore 可以指定任务运行在哪个核心
  // Core 0 = PRO 核心（负责网络IO）
  // Core 1 = APP 核心（负责用户逻辑 + 音频播放）
  httpTaskRunning = true;
  xTaskCreatePinnedToCore(
    taskHttpRead,     // 任务函数
    "httpRead",       // 任务名称
    8192,             // 栈大小（字节）
    NULL,             // 参数
    2,                // 优先级（比默认的1高）
    NULL,             // 任务句柄
    0                 // ★ 固定在 Core 0
  );

  // 7. 启动 I2S 播放任务（Core 1）
  i2sPlayRunning = true;
  xTaskCreatePinnedToCore(
    taskI2SPlay,      // 任务函数
    "i2sPlay",        // 任务名称
    4096,             // 栈大小
    NULL,             // 参数
    3,                // 优先级（最高，保证音频不卡）
    NULL,             // 任务句柄
    1                 // ★ 固定在 Core 1
  );

  Serial.println("[SYS] 所有任务已启动");
}

void loop() {
  // 主循环只处理 WebSocket 和麦克风（轻量级）
  // 音频播放由独立的 FreeRTOS 任务处理，互不干扰
  webSocket.loop();
  process_mic_upload();

  // WiFi 断线检测
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] 连接丢失，重启...");
    ESP.restart();
  }
}
