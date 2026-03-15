# test_enroll_clone.py — 测试注册克隆（enrollment）是否能成功
# 目的：逐步定位 ensure_base_dialect_voice() 的失败点
#
# 教学说明：
# 注册克隆流程有 4 个关键步骤，任一步骤都可能失败：
#   1. 用默认音色生成 TTS 样本 → WAV 文件
#   2. 上传 WAV 到 OSS → 签名 URL
#   3. 调用 create_voice() 注册 → voice_id
#   4. 轮询 query_voice() 等待 status == "OK"
# 本脚本每一步都独立 try-except，精确定位出错位置。

import os
import sys
import time
import wave
import json
import uuid
from pathlib import Path

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
except Exception:
    pass

import dashscope
from dashscope.audio.tts_v2 import (
    AudioFormat, SpeechSynthesizer, VoiceEnrollmentService
)

API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
dashscope.api_key = API_KEY

SAMPLE_TEXT = (
    "今天天气真不错，阳光明媚，微风轻拂。"
    "我喜欢在公园里散步，看着孩子们在草地上快乐地奔跑。"
    "远处的山峦若隐若现，湖面上倒映着蓝天白云的影子。"
    "生活中处处都是美好的风景，只要我们用心去感受。"
)
OUT_DIR = Path("runtime")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def step1_generate_sample() -> bytes:
    """步骤1：用默认音色生成 TTS 音频样本"""
    print("\n[Step 1] 用 longanhuan 生成 TTS 样本...")
    tts = SpeechSynthesizer(
        model="cosyvoice-v3-plus",
        voice="longanhuan",
        format=AudioFormat.WAV_16000HZ_MONO_16BIT,
    )
    audio_bytes = tts.call(SAMPLE_TEXT)
    if not audio_bytes:
        print("  ❌ TTS 生成失败")
        return b""
    print(f"  ✅ TTS 成功，{len(audio_bytes)} 字节")

    # 保存 WAV
    wav_path = OUT_DIR / "base_dialect_sample_test.wav"
    with open(wav_path, "wb") as f:
        f.write(audio_bytes)
    print(f"  📁 保存到: {wav_path}")

    # 提取 PCM（去掉 WAV 头）
    with wave.open(str(wav_path), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())
    print(f"  PCM: {len(pcm)} 字节 = {len(pcm)/32000:.1f} 秒")
    return pcm


def step2_upload_oss(pcm_data: bytes) -> str:
    """步骤2：PCM → WAV → 上传 OSS → 签名 URL"""
    import oss2

    print("\n[Step 2] 上传到 OSS...")
    # 先保存为 WAV
    wav_path = OUT_DIR / f"enroll_test_{uuid.uuid4().hex[:8]}.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_data)
    print(f"  WAV 文件: {wav_path} ({wav_path.stat().st_size} 字节)")

    auth = oss2.Auth(
        os.getenv("OSS_ACCESS_KEY_ID", "").strip(),
        os.getenv("OSS_ACCESS_KEY_SECRET", "").strip(),
    )
    bucket = oss2.Bucket(
        auth,
        os.getenv("OSS_ENDPOINT", "").strip(),
        os.getenv("OSS_BUCKET", "").strip(),
    )
    key = f"voice_assistant/dialect_test/enroll_{uuid.uuid4().hex}.wav"
    bucket.put_object_from_file(key, str(wav_path))
    signed_url = bucket.sign_url("GET", key, 86400)
    print(f"  ✅ 上传成功")
    print(f"  URL: {signed_url[:80]}...")
    return signed_url


def step3_create_voice(url: str) -> str:
    """步骤3：调用 create_voice() 提交注册请求"""
    print("\n[Step 3] 提交克隆注册请求...")
    service = VoiceEnrollmentService()
    voice_id = service.create_voice(
        target_model="cosyvoice-v3-plus",
        prefix="testdlct",
        url=url,
    )
    print(f"  ✅ 注册请求已提交: {voice_id}")
    return voice_id


def step4_poll_status(voice_id: str) -> bool:
    """步骤4：轮询等待注册完成"""
    print("\n[Step 4] 轮询等待注册完成...")
    service = VoiceEnrollmentService()
    for i in range(100):  # 最多 200 秒
        info = service.query_voice(voice_id=voice_id)
        status = (info or {}).get("status")
        print(f"  [{i+1}/100] status={status}")
        if status == "OK":
            print(f"  ✅ 注册成功！voice_id = {voice_id}")
            return True
        if status == "UNDEPLOYED":
            print(f"  ❌ 审核失败（UNDEPLOYED）")
            return False
        time.sleep(2)
    print(f"  ❌ 超时（200秒）")
    return False


def step5_test_dialect(voice_id: str):
    """步骤5：用注册好的克隆音色测试方言 instruction"""
    print("\n[Step 5] 测试克隆音色 + 方言 instruction...")

    for dialect in ["四川话", "上海话"]:
        print(f"\n  测试: {dialect}")
        tts = SpeechSynthesizer(
            model="cosyvoice-v3-plus",
            voice=voice_id,
            format=AudioFormat.WAV_16000HZ_MONO_16BIT,
            instruction=f"请用{dialect}表达。",
        )
        audio = tts.call("今天天气真不错，我们一起出去耍嘛，吃个火锅安逸得很。")
        if audio and len(audio) > 100:
            out = OUT_DIR / f"dialect_test/clone_{dialect}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as f:
                f.write(audio)
            print(f"  ✅ {dialect} 成功！{len(audio)} 字节 → {out}")
        else:
            err_info = ""
            get_resp = getattr(tts, "get_response", None)
            if callable(get_resp):
                resp = get_resp()
                if isinstance(resp, dict):
                    err_info = str(resp.get("header", {}).get("error_message", ""))
            print(f"  ❌ {dialect} 失败: {err_info}")


def main():
    print("=" * 60)
    print("  注册克隆（Enrollment）完整流程测试")
    print("=" * 60)

    # Step 1
    try:
        pcm = step1_generate_sample()
        if not pcm:
            return
    except Exception as e:
        print(f"  ❌ Step 1 异常: {e}")
        return

    # Step 2
    try:
        url = step2_upload_oss(pcm)
        if not url:
            return
    except Exception as e:
        print(f"  ❌ Step 2 异常: {e}")
        return

    # Step 3
    try:
        voice_id = step3_create_voice(url)
        if not voice_id:
            return
    except Exception as e:
        print(f"  ❌ Step 3 异常: {e}")
        return

    # Step 4
    try:
        ok = step4_poll_status(voice_id)
        if not ok:
            return
    except Exception as e:
        print(f"  ❌ Step 4 异常: {e}")
        return

    # Step 5: 测试方言
    try:
        step5_test_dialect(voice_id)
    except Exception as e:
        print(f"  ❌ Step 5 异常: {e}")
        return

    # 保存成功的 voice_id
    print("\n" + "=" * 60)
    print(f"  🎉 完整测试成功！voice_id = {voice_id}")
    print(f"  可以将此 voice_id 写入 runtime/base_dialect_voice.json")
    print("=" * 60)

    # 保存到缓存文件
    import datetime
    cache_path = OUT_DIR / "base_dialect_voice.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({
            "voice_id": voice_id,
            "created_at": datetime.datetime.now().isoformat(),
            "source_voice": "longanhuan",
            "tts_model": "cosyvoice-v3-plus",
        }, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 已保存缓存: {cache_path}")


if __name__ == "__main__":
    main()
