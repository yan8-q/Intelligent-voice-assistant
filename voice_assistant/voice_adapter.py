# voice_adapter.py — ASR 适配器
# 职责：把阿里云 FunASR 的回调接口适配成 WebSocket 可用的形式
# 教学重点：ASR 是"回调式"的（数据到了就调你的函数），需要适配成我们的架构
#
# 数据流向：
# ESP32 麦克风 → WebSocket 二进制帧 → send_audio_frame() → 阿里云 ASR
#                                                              ↓
#                                          on_partial(中间结果) / on_final(最终结果)

import os
from typing import Optional, Callable

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult


class WebSocketASRCallback(RecognitionCallback):
    """
    ASR 回调处理器：接收阿里云 FunASR 的识别结果。

    教学说明（ASR 两种结果）：
    - partial（中间结果）：用户还在说话，识别结果会不断更新
      比如："你" → "你好" → "你好吗" （实时显示在界面上）
    - final（最终结果）：一句话说完了，识别结果确定
      比如："你好吗"（这时候才触发 AI 回复）
    """

    def __init__(
        self,
        on_partial: Callable[[str], None],  # 中间结果回调
        on_final: Callable[[str], None],    # 最终结果回调
    ) -> None:
        super().__init__()
        self._last_partial: str = ""
        self._sentences: list[str] = []
        self.on_partial = on_partial
        self.on_final = on_final

    def on_event(self, result: RecognitionResult) -> None:
        """ASR 引擎每次有新结果时调用这个方法"""
        sentence = result.get_sentence()
        if not isinstance(sentence, dict):
            return
        text = sentence.get("text")
        if not text:
            return

        self._last_partial = str(text).strip()
        if self._last_partial:
            try:
                self.on_partial(self._last_partial)
            except Exception as e:
                print(f"[ASR] on_partial 错误: {e}", flush=True)

        # 判断是否是一句话的结尾
        if RecognitionResult.is_sentence_end(sentence):
            final_text = self._last_partial
            self._sentences.append(final_text)
            self._last_partial = ""
            try:
                self.on_final(final_text)
            except Exception as e:
                print(f"[ASR] on_final 错误: {e}", flush=True)

    def get_final_text(self) -> str:
        """获取所有识别到的句子（拼接）"""
        if self._last_partial:
            self._sentences.append(self._last_partial)
            self._last_partial = ""
        final = " ".join([t.strip() for t in self._sentences if t and t.strip()]).strip()
        self._sentences.clear()
        return final


class WebSocketASREngine:
    """
    ASR 引擎：接收 WebSocket 音频数据，送入阿里云 FunASR 进行实时识别。

    教学说明（完整数据流）：
    1. ESP32 的 INMP441 麦克风采集 16kHz PCM 音频
    2. ESP32 通过 WebSocket 把音频数据发给 Python 服务
    3. Python 调用 send_audio_frame() 把数据送入阿里云 ASR
    4. ASR 识别出文字后，通过回调函数通知我们
    5. on_final 触发后 → 意图路由 → LLM 对话 → TTS → 播放
    """

    def __init__(self, api_key: str, sample_rate: int = 16000, ws_url: str = None,
                 model: str = "fun-asr-realtime"):
        """
        初始化 ASR 引擎。

        教学说明（model 参数）：
        - fun-asr-realtime: 通用实时识别，支持7种方言+26种口音
        - gummy-chat-v1: 对话场景专用（可选升级）
        - 通过 .env 的 ASR_MODEL 配置，方便切换模型测试
        """
        if not api_key:
            raise RuntimeError("ASR: 缺少 DASHSCOPE_API_KEY")
        dashscope.api_key = api_key
        if ws_url:
            dashscope.base_websocket_api_url = ws_url

        self.model = model              # ASR 模型名，从 .env 读取
        self.sample_rate = sample_rate
        self.recognition: Optional[Recognition] = None
        self.callback: Optional[WebSocketASRCallback] = None

    def start(
        self,
        on_partial: Callable[[str], None],
        on_final: Callable[[str], None],
    ) -> Recognition:
        """启动 ASR 识别会话"""
        self.callback = WebSocketASRCallback(on_partial, on_final)
        self.recognition = Recognition(
            model=self.model,                   # ASR 模型，从 .env 配置读取
            format="pcm",                       # 音频格式：原始 PCM
            sample_rate=self.sample_rate,        # 采样率：16kHz
            semantic_punctuation_enabled=False,  # 不自动加标点（语音对话不需要）
            disfluency_removal_enabled=True,    # 🆕 去赘词：去掉"嗯""那个"等口语废话
            callback=self.callback,
        )
        self.recognition.start()
        return self.recognition

    def send_audio_frame(self, audio_bytes: bytes) -> None:
        """发送一帧音频数据到 ASR 引擎"""
        if self.recognition:
            try:
                self.recognition.send_audio_frame(audio_bytes)
            except Exception:
                # 教学说明：stop_current_recognition() 后台停止了 Recognition 对象
                # 但 ws_audio 的本地变量还没更新（要等 ESP32 收到 STOP 命令才清）
                # 这段时间内 send_audio_frame 会连续报错，直接清 None 阻止后续尝试
                self.recognition = None

    def stop(self) -> None:
        """停止 ASR 识别"""
        if self.recognition:
            try:
                self.recognition.stop()
            except Exception:
                pass
            self.recognition = None
