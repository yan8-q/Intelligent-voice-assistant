# cosyvoice.py — TTS 语音合成 + 音色克隆引擎
# 职责：文本转语音（流式/非流式）、录音克隆音色、OSS上传
# 教学重点：CosyVoice 是阿里云的 TTS 服务，支持流式合成和音色克隆

from __future__ import annotations

import json
import os
import re
import time
import uuid
import wave
from pathlib import Path
from typing import Optional, List

import dashscope
import oss2  # type: ignore  # 阿里云 OSS SDK，用于上传克隆音频样本
from dashscope.audio.tts_v2 import (
    AudioFormat,
    SpeechSynthesizer,
    VoiceEnrollmentService,  # 音色克隆服务
    ResultCallback,
)

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # type: ignore

# 加载 .env 文件中的环境变量
_THIS_DIR = Path(__file__).resolve().parent
_ENV_PATH = _THIS_DIR.parent / ".env"  # 项目根目录的 .env
if load_dotenv is not None and _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH, override=True)


def _env(name: str, default: str = "") -> str:
    """读取环境变量的工具函数"""
    return os.getenv(name, default).strip()


def _missing_env(names: List[str]) -> List[str]:
    """检查哪些环境变量缺失"""
    return [n for n in names if not _env(n)]


def _strip_emojis_for_tts(s: str) -> str:
    """
    移除 emoji 字符，防止 TTS 崩溃。

    教学说明：
    - LLM 回复经常包含 emoji（如 😊）
    - Windows 的 GBK 编码处理不了这些字符
    - CosyVoice 遇到 emoji 也会出错
    - 解决方案：只保留 BMP 范围内的字符（Unicode <= 0xFFFF）
    """
    return "".join(ch for ch in s if ord(ch) <= 0xFFFF)


def _normalize_for_tts(s: str) -> str:
    """
    清洗 markdown 符号，让文本更适合 TTS 朗读。
    比如去掉 **加粗**、`代码`、### 标题 等。
    """
    s = s.replace("**", "").replace("`", "")
    s = s.replace("#", "")
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# 基础方言音色注册用的朗读文本（需要足够长以提取音色特征，至少5秒）
# 教学说明：CosyVoice 克隆需要至少 5 秒有效语音。
# 内容包含多种声调和韵母组合，帮助模型更好地提取说话人音色特征。
_BASE_DIALECT_SAMPLE_TEXT = (
    "今天天气真不错，阳光明媚，微风轻拂。"
    "我喜欢在公园里散步，看着孩子们在草地上快乐地奔跑。"
    "远处的山峦若隐若现，湖面上倒映着蓝天白云的影子。"
    "生活中处处都是美好的风景，只要我们用心去感受。"
)


class CosyVoiceEngine:
    """
    CosyVoice TTS 引擎。

    教学说明：
    - tts_to_wav(): 非流式合成，一次性生成完整 WAV 文件
    - enroll_voice_from_pcm(): 从 ESP32 录音数据克隆音色
    - _oss_upload(): 上传音频样本到阿里云 OSS（克隆需要一个公网 URL）

    流式合成（speak_stream）在 app_main.py 中直接使用 SpeechSynthesizer 实现，
    因为需要和 WebSocket 广播配合，不在这个类里做。
    """

    def __init__(self, api_key: str, tts_model: str, default_voice: str,
                 sample_rate: int = 16000,
                 dialect_clone_model: Optional[str] = None) -> None:
        """
        初始化 CosyVoice TTS 引擎。

        教学说明（dialect_clone_model 参数）：
        - tts_model: 默认 TTS 模型（如 cosyvoice-v3-plus），用于普通话+系统音色
        - dialect_clone_model: 方言克隆专用模型（如 cosyvoice-v3.5-plus）
          v3.5 不支持系统音色，但原生支持 10 种方言，克隆音色质量远胜 v3
          方言克隆音色注册在 v3.5 上 → 方言 instruction 效果更好
        """
        if not api_key:
            raise RuntimeError("CosyVoice: 缺少 DASHSCOPE_API_KEY")
        dashscope.api_key = api_key

        self.tts_model = tts_model          # 模型名，如 cosyvoice-v3-plus
        self.default_voice = default_voice  # 默认音色，如 longanhuan
        self.sample_rate = sample_rate      # 采样率，16000 Hz
        # 🆕 方言克隆专用模型（v3.5-plus 方言质量更好）
        self.dialect_clone_model = dialect_clone_model or tts_model
        # 基础方言音色缓存（由 ensure_base_dialect_voice() 填充）
        self._base_dialect_voice_id: Optional[str] = None

    # ==================== 非流式 TTS ====================

    def tts_to_wav(self, text: str, voice: str, instruction: Optional[str], out_path: str,
                   model: Optional[str] = None) -> str:
        """
        非流式 TTS：文本 → WAV 文件。

        教学重点：
        - model 参数允许覆盖默认模型（方言音色需要 v3-flash 而非 v3-plus）
        - 如果 instruction 导致 428 错误（克隆音色不支持），会自动降级重试
        - 428 是阿里云返回的"参数不支持"错误码
        """
        # 允许调用方覆盖模型（方言音色需要 cosyvoice-v3-flash）
        _model = model or self.tts_model

        def _call(instr: Optional[str]):
            """内部调用函数，方便重试"""
            tts = SpeechSynthesizer(
                model=_model,
                voice=voice,
                format=AudioFormat.WAV_16000HZ_MONO_16BIT,
                instruction=instr,
            )
            audio_bytes = tts.call(text)
            # 获取响应信息（用于错误诊断）
            last = None
            rid = None
            get_resp = getattr(tts, "get_response", None)
            if callable(get_resp):
                last = get_resp()
            get_rid = getattr(tts, "get_last_request_id", None)
            if callable(get_rid):
                rid = get_rid()
            return audio_bytes, last, rid

        # 第一次尝试（带 instruction）
        audio_bytes, last, rid = _call(instruction)

        # 如果失败且是 428 错误，降级重试（不带 instruction）
        if audio_bytes is None:
            err_msg = ""
            if isinstance(last, dict):
                err_msg = str(last.get("header", {}).get("error_message", ""))
            if "428" in err_msg:
                audio_bytes2, last2, rid2 = _call(None)
                if audio_bytes2 is not None:
                    print("[TTS] 警告: 该音色不支持 instruction，已降级为 instruction=None")
                    audio_bytes, last, rid = audio_bytes2, last2, rid2

        if audio_bytes is None:
            raise RuntimeError(
                f"TTS 失败: SpeechSynthesizer.call() 返回 None\n"
                f"- model={_model}\n"
                f"- voice={voice}\n"
                f"- instruction={instruction}\n"
                f"- request_id={rid}\n"
                f"- response={last}\n"
            )

        # 保存到文件
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        return out_path

    # ==================== 音色克隆 ====================

    def _oss_upload(self, local_path: str, remote_prefix: str) -> str:
        """
        上传文件到阿里云 OSS，返回公网 URL。

        教学说明：
        - 音色克隆需要一个公网可访问的 WAV 文件 URL
        - 所以先把录音上传到 OSS，再把 URL 传给克隆 API
        - 需要配置 OSS_ACCESS_KEY_ID 等环境变量
        """
        # 教学说明：改用签名 URL 后不再需要 OSS_PUBLIC_BASE
        required = ["OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET", "OSS_ENDPOINT", "OSS_BUCKET"]
        miss = _missing_env(required)
        if miss:
            raise RuntimeError("OSS 未配置，缺少环境变量: " + ", ".join(miss))

        auth = oss2.Auth(_env("OSS_ACCESS_KEY_ID"), _env("OSS_ACCESS_KEY_SECRET"))
        bucket = oss2.Bucket(auth, _env("OSS_ENDPOINT"), _env("OSS_BUCKET"))

        p = Path(local_path)
        # 生成唯一的远程路径：prefix/日期/随机ID.wav
        key = f"{remote_prefix}/{time.strftime('%Y%m%d')}/{uuid.uuid4().hex}{p.suffix.lower()}"
        bucket.put_object_from_file(key, str(p))
        # 教学说明（签名 URL vs 公开 URL）：
        # 原来用 OSS_PUBLIC_BASE 拼普通 URL → 需要 bucket 设为 public-read 才能访问
        # 签名 URL 在 URL 里带临时授权参数（?Expires=xxx&Signature=xxx）
        # 有效期 1 小时，CosyVoice 克隆只需下载一次，绰绰有余
        # 好处：不需要改 bucket 权限，也更安全（URL 过期后自动失效）
        signed_url = bucket.sign_url('GET', key, 86400)  # 有效期 86400 秒 = 24 小时
        return signed_url

    def _save_pcm_to_wav(self, pcm_data: bytes, out_path: str, sample_rate: int = 16000) -> str:
        """将 PCM 原始数据（int16 mono）保存为标准 WAV 文件"""
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)        # 单声道
            wf.setsampwidth(2)        # 16bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return out_path

    def upload_voice_sample(self, pcm_data: bytes) -> str:
        """
        即时克隆：只上传音频到 OSS，返回签名 URL。

        教学说明（即时克隆 vs 注册克隆）：
        - 注册克隆（enroll_voice_from_pcm）：上传 → 训练 → 轮询等 1-2 分钟 → voice_id
        - 即时克隆（本方法）：上传 → 返回 URL → 把 URL 当 voice 参数传给 TTS → 零等待
        CosyVoice v3-plus 支持把音频 URL 直接作为 voice 参数，
        模型会实时提取说话人的音色特征，不需要后台训练。
        签名 URL 有效期 24 小时，过期后重新"克隆"即可。
        """
        if not pcm_data or len(pcm_data) < 16000:  # 至少 0.5 秒
            raise RuntimeError("录音数据太短，请确保麦克风正常工作")

        out_dir = Path("runtime")
        out_dir.mkdir(parents=True, exist_ok=True)
        wav_path = out_dir / f"clone_{uuid.uuid4().hex}.wav"

        # 1. PCM → WAV 文件
        self._save_pcm_to_wav(pcm_data, str(wav_path), sample_rate=self.sample_rate)
        print(f"[CLONE] 已保存录音: {wav_path} ({len(pcm_data)} 字节)")

        # 2. 上传到 OSS，获取签名 URL
        url = self._oss_upload(str(wav_path), remote_prefix="voice_assistant/voice_samples")
        print(f"[CLONE] 即时克隆 URL: {url[:80]}...")

        # 3. 清理本地临时文件
        try:
            wav_path.unlink()
        except Exception:
            pass

        return url  # 这个 URL 直接当 voice 参数传给 SpeechSynthesizer

    def enroll_voice_from_pcm(self, pcm_data: bytes, prefix: str = "myvoice",
                              target_model: Optional[str] = None) -> str:
        """
        从 ESP32 麦克风录音数据克隆音色。

        教学说明（克隆流程）：
        1. ESP32 录制 7 秒音频 → PCM 数据通过 WebSocket 发到 Python
        2. Python 把 PCM 保存为 WAV 文件
        3. 上传 WAV 到阿里云 OSS，获得公网 URL
        4. 调用 VoiceEnrollmentService.create_voice() 提交克隆请求
        5. 轮询 query_voice() 等待克隆完成（通常 1-2 分钟）
        6. 返回 voice_id，后续 TTS 用这个 ID 就能用克隆音色

        教学说明（target_model 参数）：
        voice_id 绑定到 target_model！v3-plus 的 voice_id 不能用于 v3.5-plus。
        方言克隆时传 cosyvoice-v3.5-plus，用户克隆时用默认的 tts_model。

        Args:
            pcm_data: int16 mono 16kHz 的 PCM 音频数据
            prefix: 音色名称前缀
            target_model: 目标模型（默认用 self.tts_model，方言克隆用 v3.5-plus）
        Returns:
            voice_id: 克隆成功的音色ID
        """
        if not pcm_data or len(pcm_data) < 16000:  # 至少 0.5 秒
            raise RuntimeError("录音数据太短，请确保麦克风正常工作")

        out_dir = Path("runtime")
        out_dir.mkdir(parents=True, exist_ok=True)
        wav_path = out_dir / f"enroll_{uuid.uuid4().hex}.wav"

        # 1. PCM → WAV 文件
        self._save_pcm_to_wav(pcm_data, str(wav_path), sample_rate=self.sample_rate)
        print(f"[CLONE] 已保存录音: {wav_path} ({len(pcm_data)} 字节)")

        # 2. 上传到 OSS
        url = self._oss_upload(str(wav_path), remote_prefix="voice_assistant/voice_samples")
        print(f"[CLONE] 样本URL: {url}")

        # 3. 提交克隆请求
        # 教学说明：target_model 决定这个 voice_id 能被哪个模型使用
        _target = target_model or self.tts_model
        service = VoiceEnrollmentService()
        voice_id = service.create_voice(target_model=_target, prefix=prefix, url=url)
        print(f"[CLONE] create_voice 提交成功：{voice_id}")

        # 4. 轮询等待完成（最多 150 秒）
        # 教学说明：轮询间隔从 5 秒改为 2 秒，更快检测到完成状态
        # 总次数 75 次 × 2 秒 = 150 秒超时不变
        for i in range(75):
            info = service.query_voice(voice_id=voice_id)
            status = (info or {}).get("status")
            print(f"[CLONE] 状态({i+1}/75): {status}")
            if status == "OK":
                return voice_id
            if status == "UNDEPLOYED":
                raise RuntimeError("音色审核失败（UNDEPLOYED），请换更干净的朗读样本重试。")
            time.sleep(2)

        raise RuntimeError("音色创建超时，请稍后重试。")

    # ==================== 基础方言音色（预注册）====================

    def ensure_base_dialect_voice(self) -> Optional[str]:
        """
        确保基础方言音色已注册，返回 voice_id。

        教学说明（为什么需要基础方言音色）：
        - CosyVoice 只有 4 种方言专用音色（广东话/东北话/陕西话/闽南话）
        - 其余 12 种方言（四川话/上海话/河南话等）没有专用音色
        - 系统音色（longanhuan）不支持方言 instruction（会 428 错误）
        - 但克隆音色支持方言 instruction（如"请用四川话表达。"）
        - 所以我们用默认音色生成一段朗读音频，注册为克隆音色
        - 这个克隆音色就成了"万能方言底座"，配合方言 instruction 可以说任何方言
        - voice_id 是永久的，注册一次写入文件，后续启动直接读取

        Returns:
            voice_id（成功）或 None（失败，降级为文字模拟）
        """
        cache_path = Path("runtime/base_dialect_voice.json")

        # 1. 尝试从缓存读取（voice_id 是永久的，无需重新注册）
        # 教学说明（缓存模型检查）：
        #   voice_id 是绑定到 target_model 的！v3-plus 注册的 voice_id 不能用于 v3.5-plus。
        #   如果 .env 中 DIALECT_CLONE_MODEL 改了（比如从 v3-plus 升级到 v3.5-plus），
        #   旧缓存的 voice_id 就失效了，需要删除缓存重新注册。
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                voice_id = data.get("voice_id", "")
                cached_model = data.get("tts_model", "")
                # 检查缓存的模型是否匹配当前配置
                if voice_id and cached_model == self.dialect_clone_model:
                    print(f"[DIALECT] 从缓存加载基础方言音色: {voice_id} (model={cached_model})", flush=True)
                    self._base_dialect_voice_id = voice_id
                    return voice_id
                elif voice_id and cached_model != self.dialect_clone_model:
                    # 模型已升级（如 v3-plus → v3.5-plus），旧 voice_id 不兼容
                    print(f"[DIALECT] 方言模型已升级: {cached_model} → {self.dialect_clone_model}，需重新注册", flush=True)
                    cache_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"[DIALECT] 缓存文件读取失败: {e}", flush=True)

        # 2. 检查 OSS 是否配置（注册克隆必须上传到 OSS）
        required_env = ["OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET", "OSS_ENDPOINT", "OSS_BUCKET"]
        miss = _missing_env(required_env)
        if miss:
            print(f"[DIALECT] OSS 未配置（缺少 {miss}），无法注册基础方言音色", flush=True)
            return None

        # 3. 用默认音色生成 TTS 音频作为克隆样本
        # 教学说明：用 longanhuan（默认系统音色）朗读一段文字，
        # 把这段音频注册为克隆音色，后续就能对这个音色下方言 instruction
        print("[DIALECT] 正在生成基础方言音色样本...", flush=True)
        sample_wav = Path("runtime") / "base_dialect_sample.wav"
        try:
            self.tts_to_wav(
                text=_BASE_DIALECT_SAMPLE_TEXT,
                voice=self.default_voice,
                instruction=None,
                out_path=str(sample_wav),
                model=self.tts_model,
            )
        except Exception as e:
            print(f"[DIALECT] 生成样本音频失败: {e}", flush=True)
            return None

        # 4. 读取 WAV 文件为 PCM 数据
        try:
            with wave.open(str(sample_wav), "rb") as wf:
                pcm_data = wf.readframes(wf.getnframes())
            print(f"[DIALECT] 样本音频: {len(pcm_data)} 字节 = {len(pcm_data)/32000:.1f}秒", flush=True)
        except Exception as e:
            print(f"[DIALECT] 读取样本 WAV 失败: {e}", flush=True)
            return None

        # 5. 注册为克隆音色（这一步最耗时，需要 1-2 分钟）
        # 教学说明（v3.5 升级）：
        #   注册时 target_model 用 dialect_clone_model（v3.5-plus）而非默认 tts_model（v3-plus）
        #   v3.5 原生支持 10 种方言，克隆音色+方言 instruction 效果远胜 v3
        try:
            voice_id = self.enroll_voice_from_pcm(
                pcm_data, prefix="basedlct",
                target_model=self.dialect_clone_model  # 🆕 用 v3.5-plus 注册
            )
            print(f"[DIALECT] 基础方言音色注册成功: {voice_id} (model={self.dialect_clone_model})", flush=True)
        except Exception as e:
            print(f"[DIALECT] 注册失败: {e}", flush=True)
            return None
        finally:
            # 清理临时样本文件
            try:
                sample_wav.unlink(missing_ok=True)
            except Exception:
                pass

        # 6. 保存到缓存文件（后续启动直接读取，不需要重新注册）
        try:
            import datetime
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({
                    "voice_id": voice_id,
                    "created_at": datetime.date.today().isoformat(),
                    "source_voice": self.default_voice,
                    "tts_model": self.dialect_clone_model,  # 🆕 记录实际注册的模型
                }, f, ensure_ascii=False, indent=2)
            print(f"[DIALECT] 已保存缓存: {cache_path}", flush=True)
        except Exception as e:
            print(f"[DIALECT] 保存缓存失败（不影响使用）: {e}", flush=True)

        self._base_dialect_voice_id = voice_id
        return voice_id

    def validate_voice_id(self, voice_id: str) -> bool:
        """
        验证 voice_id 是否仍然有效。

        教学说明：
        voice_id 理论上是永久的，但阿里云可能因各种原因使其失效（如长期不用被回收）。
        用 VoiceEnrollmentService.query_voice() 查询状态，
        只有 status == "OK" 才算有效。
        """
        try:
            service = VoiceEnrollmentService()
            info = service.query_voice(voice_id=voice_id)
            status = (info or {}).get("status")
            print(f"[DIALECT] 验证 voice_id={voice_id}: status={status}", flush=True)
            return status == "OK"
        except Exception as e:
            print(f"[DIALECT] 验证 voice_id 失败: {e}", flush=True)
            return False
