# test_dialect_tts.py — 方言 TTS 诊断测试脚本
# 目的：测试不同 模型 × 音色 × instruction 组合，找出哪条路径能让方言真正发音
#
# 教学说明：
# CosyVoice 的 instruction 支持有严格限制：
#   - 系统音色（如 longanhuan）：只支持情感/角色 instruction（方言 → 428 错误）
#   - 克隆音色（voice_id / URL）：只支持方言 instruction（情感 → 428 错误）
#   - 方言音色（如 longanyue_v3）：不支持任何 instruction（本身自带方言）
# 本脚本要验证以上规则在最新 API 版本下是否仍然成立，
# 并测试即时克隆（URL 作为 voice 参数）是否支持方言 instruction。

import os
import sys
import time
import wave
import uuid
from pathlib import Path

# Windows 控制台 UTF-8
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
except Exception:
    pass

import dashscope
from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer

# ---- 配置 ----
API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
if not API_KEY:
    print("❌ 缺少 DASHSCOPE_API_KEY，请编辑 .env 文件")
    sys.exit(1)
dashscope.api_key = API_KEY

# 测试用的文本（包含方言特征词，方便听出效果）
TEST_TEXT = "今天天气真不错，我们一起出去耍嘛，吃个火锅安逸得很。"
# 输出目录
OUT_DIR = Path("runtime/dialect_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def test_tts(label: str, model: str, voice: str, instruction: str = None) -> bool:
    """
    测试一种 TTS 组合，返回是否成功。

    教学说明：
    - SpeechSynthesizer.call() 返回 bytes = 成功
    - 返回 None = 失败（通常是 428 错误：该音色不支持此 instruction）
    - 成功时保存 WAV 文件到 runtime/dialect_test/ 方便人工试听
    """
    print(f"\n{'='*60}")
    print(f"  测试: {label}")
    print(f"  模型: {model}")
    print(f"  音色: {voice[:60]}{'...' if len(voice) > 60 else ''}")
    print(f"  指令: {instruction or '(无)'}")
    print(f"{'='*60}")

    try:
        tts = SpeechSynthesizer(
            model=model,
            voice=voice,
            format=AudioFormat.WAV_16000HZ_MONO_16BIT,
            instruction=instruction,
        )
        t0 = time.time()
        audio_bytes = tts.call(TEST_TEXT)
        elapsed = time.time() - t0

        if audio_bytes and len(audio_bytes) > 100:
            # 保存到文件，方便人工试听
            safe_label = label.replace(" ", "_").replace("/", "_")
            out_path = OUT_DIR / f"{safe_label}.wav"
            with open(out_path, "wb") as f:
                f.write(audio_bytes)
            print(f"  ✅ 成功！音频 {len(audio_bytes)} 字节，耗时 {elapsed:.1f}s")
            print(f"  📁 已保存: {out_path}")
            return True
        else:
            # 获取错误信息
            err_info = ""
            get_resp = getattr(tts, "get_response", None)
            if callable(get_resp):
                resp = get_resp()
                if isinstance(resp, dict):
                    err_info = str(resp.get("header", {}).get("error_message", ""))
            print(f"  ❌ 失败：返回 None")
            if err_info:
                print(f"  错误信息: {err_info}")
            if "428" in err_info:
                print(f"  → 428 错误：该音色不支持此 instruction 类型")
            return False
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return False


def create_instant_clone_url() -> str:
    """
    创建即时克隆用的音频 URL。

    教学说明（即时克隆流程）：
    1. 用默认音色 longanhuan 生成一段普通话朗读音频
    2. 上传到 OSS，获取签名 URL
    3. 把这个 URL 直接作为 voice 参数传给 TTS
    4. CosyVoice 会实时提取该音频的音色特征，效果等同克隆
    5. 不需要 enrollment 注册，不需要等 1-2 分钟
    """
    import oss2

    # 检查 OSS 配置
    required = ["OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET", "OSS_ENDPOINT", "OSS_BUCKET"]
    missing = [k for k in required if not os.getenv(k, "").strip()]
    if missing:
        print(f"  ⚠️ OSS 未配置（缺少 {missing}），跳过即时克隆测试")
        return ""

    # Step 1: 用默认音色生成朗读样本
    sample_text = (
        "今天天气真不错，阳光明媚，微风轻拂。"
        "我喜欢在公园里散步，看着孩子们在草地上快乐地奔跑。"
        "远处的山峦若隐若现，湖面上倒映着蓝天白云的影子。"
        "生活中处处都是美好的风景，只要我们用心去感受。"
    )
    print("\n  [即时克隆] 正在生成样本音频...")
    tts = SpeechSynthesizer(
        model="cosyvoice-v3-plus",
        voice="longanhuan",
        format=AudioFormat.WAV_16000HZ_MONO_16BIT,
    )
    audio_bytes = tts.call(sample_text)
    if not audio_bytes:
        print("  ❌ 生成样本音频失败")
        return ""
    print(f"  [即时克隆] 样本音频: {len(audio_bytes)} 字节")

    # 保存到临时 WAV 文件
    tmp_wav = OUT_DIR / f"clone_sample_{uuid.uuid4().hex[:8]}.wav"
    with open(tmp_wav, "wb") as f:
        f.write(audio_bytes)

    # Step 2: 上传到 OSS
    print("  [即时克隆] 上传到 OSS...")
    auth = oss2.Auth(
        os.getenv("OSS_ACCESS_KEY_ID", "").strip(),
        os.getenv("OSS_ACCESS_KEY_SECRET", "").strip(),
    )
    bucket = oss2.Bucket(auth, os.getenv("OSS_ENDPOINT", "").strip(), os.getenv("OSS_BUCKET", "").strip())
    key = f"voice_assistant/dialect_test/{uuid.uuid4().hex}.wav"
    bucket.put_object_from_file(key, str(tmp_wav))
    # 签名 URL（24 小时有效）
    signed_url = bucket.sign_url("GET", key, 86400)
    print(f"  [即时克隆] 签名 URL: {signed_url[:80]}...")

    # 清理临时文件
    try:
        tmp_wav.unlink()
    except Exception:
        pass

    return signed_url


def main():
    """
    运行完整的方言 TTS 测试矩阵。

    教学说明（测试策略）：
    我们要回答3个关键问题：
    Q1: v3-flash 模型的系统音色（longanhuan）能否接受方言 instruction？
        → 如果能，这是最简方案（零克隆依赖）
    Q2: v3-plus 模型的系统音色（longanhuan）能否接受方言 instruction？
        → 之前确认不能（428），这里再验证一下
    Q3: 即时克隆 URL 作为 voice 参数时，能否接受方言 instruction？
        → 如果能，这比注册克隆更快更可靠
    """
    print("=" * 60)
    print("  CosyVoice 方言 TTS 诊断测试")
    print("=" * 60)

    results = {}

    # ---- 测试组 1：系统音色 + 方言 instruction ----
    print("\n\n📋 测试组 1：系统音色 + 方言 instruction")
    print("  目的：验证系统音色是否支持方言 instruction")

    results["v3-flash+longanhuan+四川话"] = test_tts(
        label="v3flash_longanhuan_sichuan",
        model="cosyvoice-v3-flash",
        voice="longanhuan",
        instruction="请用四川话表达。",
    )

    results["v3-flash+longanhuan+上海话"] = test_tts(
        label="v3flash_longanhuan_shanghai",
        model="cosyvoice-v3-flash",
        voice="longanhuan",
        instruction="请用上海话表达。",
    )

    results["v3-plus+longanhuan+四川话"] = test_tts(
        label="v3plus_longanhuan_sichuan",
        model="cosyvoice-v3-plus",
        voice="longanhuan",
        instruction="请用四川话表达。",
    )

    # ---- 测试组 2：即时克隆 + 方言 instruction ----
    print("\n\n📋 测试组 2：即时克隆 URL + 方言 instruction")
    print("  目的：验证即时克隆音色是否支持方言 instruction")

    clone_url = create_instant_clone_url()
    if clone_url:
        results["即时克隆+v3-plus+四川话"] = test_tts(
            label="instant_clone_v3plus_sichuan",
            model="cosyvoice-v3-plus",
            voice=clone_url,
            instruction="请用四川话表达。",
        )

        results["即时克隆+v3-plus+上海话"] = test_tts(
            label="instant_clone_v3plus_shanghai",
            model="cosyvoice-v3-plus",
            voice=clone_url,
            instruction="请用上海话表达。",
        )

        results["即时克隆+v3-plus+无instruction"] = test_tts(
            label="instant_clone_v3plus_no_instr",
            model="cosyvoice-v3-plus",
            voice=clone_url,
            instruction=None,
        )
    else:
        print("  ⏭️ 跳过即时克隆测试（OSS 未配置）")

    # ---- 测试组 3：对照组（确认已知可行的组合）----
    print("\n\n📋 测试组 3：对照组（确认已知可行的组合）")

    results["v3-flash+粤语专用音色"] = test_tts(
        label="v3flash_cantonese_voice",
        model="cosyvoice-v3-flash",
        voice="longanyue_v3",
        instruction=None,  # 方言音色不需要 instruction
    )

    results["v3-plus+longanhuan+情感"] = test_tts(
        label="v3plus_longanhuan_emotion",
        model="cosyvoice-v3-plus",
        voice="longanhuan",
        instruction="你说话的情感是happy。",
    )

    # ---- 汇总结果 ----
    print("\n\n" + "=" * 60)
    print("  📊 测试结果汇总")
    print("=" * 60)
    for name, ok in results.items():
        status = "✅ 成功" if ok else "❌ 失败"
        print(f"  {status}  {name}")

    # ---- 给出建议 ----
    print("\n" + "=" * 60)
    print("  💡 修复建议")
    print("=" * 60)

    if results.get("v3-flash+longanhuan+四川话") and results.get("v3-flash+longanhuan+上海话"):
        print("  🎉 路径A可行：v3-flash 系统音色支持方言 instruction！")
        print("  → 最简方案：非专用方言用 longanhuan + v3-flash + 方言instruction")
        print("  → 不需要任何克隆，零等待，最可靠")
    elif results.get("即时克隆+v3-plus+四川话") and results.get("即时克隆+v3-plus+上海话"):
        print("  👍 路径B可行：即时克隆 URL 支持方言 instruction！")
        print("  → 次优方案：启动时生成样本→上传OSS→URL作为voice参数")
        print("  → 3-5秒完成，比注册克隆快得多")
    else:
        print("  ⚠️ 路径A/B均不可行，需要走路径C：修复注册克隆机制")
        print("  → 检查 enrollment API 是否有变化")
        print("  → 加强错误日志定位失败原因")

    print(f"\n  🔊 所有测试音频已保存到: {OUT_DIR.resolve()}")
    print("  请用播放器试听，确认方言口音是否真实\n")


if __name__ == "__main__":
    main()
