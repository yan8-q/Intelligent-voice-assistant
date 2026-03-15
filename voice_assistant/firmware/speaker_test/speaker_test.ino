// ===== 喇叭最简测试 =====
// 只测 MAX98357A 能不能发声，不连WiFi，不连麦克风
// 如果这个都没声音 → 接线有问题

#include <driver/i2s.h>

// MAX98357A 引脚（和主固件一样）
#define SPK_BCLK  26
#define SPK_LRC   27
#define SPK_DIN   25

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n===== 喇叭测试 =====");

  // 初始化 I2S 输出
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX);
  cfg.sample_rate = 16000;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = 512;
  cfg.use_apll = false;
  cfg.tx_desc_auto_clear = true;

  esp_err_t err = i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
  Serial.printf("I2S install: %s\n", esp_err_to_name(err));

  i2s_pin_config_t pins = {};
  pins.mck_io_num   = I2S_PIN_NO_CHANGE;
  pins.bck_io_num   = SPK_BCLK;
  pins.ws_io_num    = SPK_LRC;
  pins.data_out_num = SPK_DIN;
  pins.data_in_num  = I2S_PIN_NO_CHANGE;

  err = i2s_set_pin(I2S_NUM_0, &pins);
  Serial.printf("I2S set_pin: %s\n", esp_err_to_name(err));

  Serial.printf("BCLK=GPIO%d, LRC=GPIO%d, DIN=GPIO%d\n", SPK_BCLK, SPK_LRC, SPK_DIN);
  Serial.println("开始播放 800Hz 正弦波，持续5秒...");
  Serial.println("如果听不到 → 接线有问题！");

  // 播放 5 秒 800Hz 正弦波
  const int freq = 800;
  const int sr = 16000;
  const int duration = 5;  // 5秒
  const int total = sr * duration;
  const int16_t amp = 32000;
  const int chunk = 256;
  int16_t buf[chunk];

  for (int i = 0; i < total; i += chunk) {
    int cnt = min(chunk, total - i);
    for (int j = 0; j < cnt; j++) {
      float t = (float)(i + j) / sr;
      buf[j] = (int16_t)(amp * sin(2.0 * 3.14159265 * freq * t));
    }
    size_t bw = 0;
    i2s_write(I2S_NUM_0, buf, cnt * sizeof(int16_t), &bw, portMAX_DELAY);
  }

  i2s_zero_dma_buffer(I2S_NUM_0);
  Serial.println("播放完成！");
}

void loop() {
  // 每3秒重新播一次，方便测试
  delay(3000);
  Serial.println("再播一次...");

  const int freq = 500;  // 换个频率
  const int sr = 16000;
  const int total = sr * 2;  // 2秒
  const int16_t amp = 32000;
  const int chunk = 256;
  int16_t buf[chunk];

  for (int i = 0; i < total; i += chunk) {
    int cnt = min(chunk, total - i);
    for (int j = 0; j < cnt; j++) {
      float t = (float)(i + j) / sr;
      buf[j] = (int16_t)(amp * sin(2.0 * 3.14159265 * freq * t));
    }
    size_t bw = 0;
    i2s_write(I2S_NUM_0, buf, cnt * sizeof(int16_t), &bw, portMAX_DELAY);
  }
  i2s_zero_dma_buffer(I2S_NUM_0);
}
