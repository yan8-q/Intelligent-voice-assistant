# test_dialect_tts2.py — 扩大测试范围
# 教学说明：第一轮只测了 longanhuan + v3-flash/v3-plus，范围太窄。
# 这次测试更多 模型 × 音色 × instruction 组合，找出能用的路径。

import os
import sys
import time
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
from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

TEST_TEXT = "今天天气真不错，我们一起出去耍嘛，吃个火锅安逸得很。"
OUT_DIR = Path("runtime/dialect_test2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def test(label, model, voice, instruction=None):
    """测试一种 TTS 组合"""
    print(f"\n--- {label} ---")
    print(f"    model={model}, voice={voice}, instr={instruction or '(无)'}")
    try:
        tts = SpeechSynthesizer(
            model=model, voice=voice,
            format=AudioFormat.WAV_16000HZ_MONO_16BIT,
            instruction=instruction,
        )
        t0 = time.time()
        audio = tts.call(TEST_TEXT)
        dt = time.time() - t0

        if audio and len(audio) > 100:
            safe = label.replace(" ", "_").replace("/", "_")
            out = OUT_DIR / f"{safe}.wav"
            with open(out, "wb") as f:
                f.write(audio)
            print(f"    ✅ 成功 ({len(audio)}B, {dt:.1f}s) → {out}")
            return True
        else:
            # 获取错误信息
            err = ""
            get_resp = getattr(tts, "get_response", None)
            if callable(get_resp):
                r = get_resp()
                if isinstance(r, dict):
                    err = str(r.get("header", {}).get("error_message", ""))
            print(f"    ❌ 失败: {err}")
            return False
    except Exception as e:
        print(f"    ❌ 异常: {e}")
        return False


def main():
    print("=" * 60)
    print("  CosyVoice 方言 TTS 扩展测试")
    print("=" * 60)

    results = {}

    # ===== 测试不同模型 =====
    models = [
        "cosyvoice-v3-flash",
        "cosyvoice-v3-plus",
        "cosyvoice-v3.5-flash",
        "cosyvoice-v3.5-plus",
        "cosyvoice-v2",
    ]

    # ===== 测试不同音色 =====
    voices = ["longanhuan", "longanyang"]

    # ===== 测试组1：各模型 × 各音色 × 方言instruction =====
    print("\n\n📋 测试组1：各模型 × 各音色 × 四川话instruction")
    for model in models:
        for voice in voices:
            key = f"{model}+{voice}+四川话"
            results[key] = test(key, model, voice, "请用四川话表达。")

    # ===== 测试组2：v3.5 模型 × 情感instruction（确认模型本身是否可用）=====
    print("\n\n📋 测试组2：v3.5 模型基础可用性")
    for model in ["cosyvoice-v3.5-flash", "cosyvoice-v3.5-plus"]:
        # 无instruction
        key = f"{model}+longanhuan+无instr"
        results[key] = test(key, model, "longanhuan", None)
        # 情感instruction
        key = f"{model}+longanhuan+情感"
        results[key] = test(key, model, "longanhuan", "你说话的情感是happy。")

    # ===== 测试组3：v3-flash 其他方言（不用专用音色，用通用音色+instruction）=====
    print("\n\n📋 测试组3：v3-flash + longanyang + 各种方言")
    for dialect in ["四川话", "上海话", "河南话"]:
        key = f"v3flash+longanyang+{dialect}"
        results[key] = test(key, "cosyvoice-v3-flash", "longanyang", f"请用{dialect}表达。")

    # ===== 测试组4：v3-flash 童声/其他音色 + 方言instruction =====
    print("\n\n📋 测试组4：v3-flash 其他支持Instruct的音色 + 方言")
    # longhuhu_v3 标注支持 Instruct
    key = "v3flash+longhuhu_v3+四川话"
    results[key] = test(key, "cosyvoice-v3-flash", "longhuhu_v3", "请用四川话表达。")

    # ===== 汇总 =====
    print("\n\n" + "=" * 60)
    print("  📊 测试结果汇总")
    print("=" * 60)

    # 先显示成功的
    success = {k: v for k, v in results.items() if v}
    fail = {k: v for k, v in results.items() if not v}

    if success:
        print("\n  ✅ 成功的组合：")
        for k in success:
            print(f"    ✅ {k}")

    print(f"\n  ❌ 失败的组合（{len(fail)}个）：")
    for k in fail:
        print(f"    ❌ {k}")

    if success:
        print("\n  💡 找到可行路径！请试听上面成功的音频文件。")
    else:
        print("\n  ⚠️ 所有组合都失败，需要走注册克隆路径。")

    print(f"\n  🔊 音频保存在: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
