# 智能语音助手项目

## 项目概述
基于 ESP32-WROOM-32 的智能语音交互助手，从 RobotDuck 项目精简而来，保留全部语音功能。

## 功能清单
- 实时语音对话（阿里云 FunASR + 通义千问 + CosyVoice v3/v3.5）
- 音色克隆（录制几秒声音即可克隆）
- 方言模式（广东话、四川话等16种）
- 角色/场景模式（脱口秀、rap、客服等）
- 流式 TTS（边想边说，低延迟）
- Web 控制面板（实时 ASR 显示 + 手动文字输入）

## 硬件清单
| 物品 | 型号 |
|------|------|
| 主控板 | ESP32-WROOM-32 开发板 |
| 麦克风 | INMP441 I2S 数字麦克风 |
| 功放+喇叭 | MAX98357A I2S 功放模块 |
| 辅材 | 杜邦线 + 400孔面包板 |

## 硬件接线
```
INMP441:   SCK → GPIO14, WS → GPIO15, SD → GPIO32, VDD → 3V3, GND → GND, L/R → GND
MAX98357A: BCLK → GPIO26, LRC → GPIO27, DIN → GPIO25, VIN → 5V(VIN), GND → GND
```

## 项目结构
```
voice_assistant/
├── .env                ← 阿里云 API 配置
├── requirements.txt    ← Python 依赖
├── app_main.py         ← 主服务（FastAPI, 端口 8081）
├── audio_stream.py     ← 音频流广播
├── voice_adapter.py    ← ASR 适配器
├── voice_core/         ← 语音核心包
│   ├── cosyvoice.py    ← TTS + 克隆
│   ├── dispatcher.py   ← 意图路由
│   ├── state.py        ← 对话状态
│   └── workflows.py    ← 工作流
├── firmware/
│   └── voice_assistant.ino ← ESP32 固件
├── templates/
│   └── index.html      ← Web 面板
└── static/
    └── app.js
```

## 快速启动
1. 编辑 `.env` 填入阿里云 API Key
2. `pip install -r requirements.txt`
3. `python app_main.py`
4. Arduino IDE 烧录 `firmware/voice_assistant.ino`（修改 WiFi 配置）
5. 打开 http://localhost:8081 查看控制面板

## 代码规范
- 永远用中文注释
- 代码永远要加注释
