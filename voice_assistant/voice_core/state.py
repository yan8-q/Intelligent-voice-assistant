# state.py — 对话状态管理
# 职责：追踪当前音色、方言、角色/场景、对话历史
# 教学重点：dataclass 是 Python 的"结构体"，用来存储一组相关数据

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 情绪类型（字符串别名，方便类型提示）
Emotion = str  # neutral|fearful|angry|sad|surprised|happy|disgusted

# 方言专用系统音色映射（阿里云 cosyvoice-v3-flash 预置音色）
# 教学说明：
#   阿里云为常见方言准备了专用音色，这些音色本身就说对应方言，不需要 instruction。
#   就像请了一个说粤语的播音员，而不是让普通话播音员模仿粤语。
#   这些音色只在 cosyvoice-v3-flash 模型下可用（v3-plus 不支持）。
#   没有专用音色的方言（如四川话、上海话）只能靠 LLM 文字模拟或克隆音色+instruction。
DIALECT_VOICES: Dict[str, str] = {
    "广东话": "longanyue_v3",    # 欢脱粤语男
    "东北话": "longlaotie_v3",   # 东北直率男
    "陕西话": "longshange_v3",   # 原味陕北男
    "闽南话": "longanmin_v3",    # 清纯萝莉女
}
# 方言音色需要的 TTS 模型（v3-flash 支持方言音色，v3-plus 不支持）
DIALECT_TTS_MODEL = "cosyvoice-v3-flash"

# 方言别名 → 规范名 映射
# 教学说明：
#   用户可能说"粤语"而不是"广东话"，LLM 返回的 dialect 参数也可能是别名。
#   如果不归一化，"粤语" not in DIALECT_VOICES → 有专用音色却用不上！
#   所有入口（关键词预检、LLM 路由结果）都要通过这个表归一化。
DIALECT_ALIASES: Dict[str, str] = {
    "粤语": "广东话",
    "白话": "广东话",
    "川话": "四川话",
    "沪语": "上海话",
    "台语": "闽南话",
}


def normalize_dialect(name: str) -> str:
    """将方言别名归一化为规范名（如 '粤语' → '广东话'）"""
    return DIALECT_ALIASES.get(name, name)


@dataclass
class ChatState:
    """
    对话状态：记录当前语音助手的所有"模式"设置和对话记忆。

    教学说明：
    - default_voice: 系统默认音色（阿里云预置）
    - current_voice: 当前使用的音色（可能是克隆的）
    - is_cloned_voice: 是否正在用克隆音色（影响 TTS instruction 的构建）
    - dialect: 当前方言模式（如"四川话"）
    - scene/role/style_hint: 角色扮演相关
    - messages: 对话历史（LLM 需要上下文才能连贯对话）
    - max_turns: 最多保留几轮对话（防止 token 超限）
    """

    # ---- 音色状态 ----
    default_voice: str = "longanhuan"       # 系统默认音色
    current_voice: str = "longanhuan"       # 当前音色（克隆后会变）
    is_cloned_voice: bool = False           # 是否已克隆
    has_base_dialect_voice: bool = False    # 是否有基础方言音色可用（全局资源，不随 reset 清除）

    # ---- 风格模式（持久化，直到用户说"恢复默认"）----
    dialect: Optional[str] = None           # 方言，如 "四川话"、"广东话"
    scene: Optional[str] = None             # 场景，如 "脱口秀表演"
    role: Optional[str] = None              # 角色，如 "温和客服"
    style_hint: Optional[str] = None        # 风格提示，如 "rap"、"押韵"

    # ---- 对话记忆 ----
    messages: List[Dict[str, Any]] = field(default_factory=list)
    max_turns: int = 8                      # 最多保留8轮对话

    def reset_to_default(self) -> None:
        """恢复所有设置到默认状态"""
        self.current_voice = self.default_voice
        self.is_cloned_voice = False
        self.dialect = None
        self.scene = None
        self.role = None
        self.style_hint = None

    def cancel_clone(self) -> None:
        """只取消克隆音色，保留方言/角色/场景设置。
        教学说明：用户说"取消克隆"时只想恢复默认声音，
        不应该连方言模式、角色扮演一起清掉。
        """
        self.current_voice = self.default_voice
        self.is_cloned_voice = False

    def cancel_dialect(self) -> None:
        """只取消方言模式，保留克隆音色/角色/场景设置。
        教学说明：用户说"说回普通话"时只想取消方言，
        不应该连克隆的声音一起清掉。
        """
        self.dialect = None

    def set_cloned_voice(self, voice_id: str) -> None:
        """设置克隆音色"""
        self.current_voice = voice_id
        self.is_cloned_voice = True

    def get_tts_voice_and_model(self, default_model: str) -> tuple:
        """
        获取当前应使用的 TTS 音色和模型。

        教学说明（方言的三种路径）：
        1. 已克隆音色 → 用克隆音色 + 默认模型（方言通过 instruction 实现）
        2. 有方言专用音色 → 用方言音色 + v3-flash（音色自带方言，无需 instruction）
        3. 没有方言音色 → 用默认音色 + 默认模型（方言靠 LLM 文字模拟）

        Returns:
            (voice, model) 元组
        """
        # 已克隆：用克隆音色，模型不变
        if self.is_cloned_voice:
            return (self.current_voice, default_model)
        # 有方言专用音色：切换到方言音色 + v3-flash 模型
        if self.dialect and self.dialect in DIALECT_VOICES:
            return (DIALECT_VOICES[self.dialect], DIALECT_TTS_MODEL)
        # 默认：用当前音色和默认模型
        return (self.current_voice, default_model)

    def build_tts_instruction(self, emotion: Emotion) -> Optional[str]:
        """
        构建 TTS 的 instruction 参数。

        教学重点（CosyVoice instruction 规则）：

        v3 系列规则（系统音色 + v3-flash 方言音色）：
        - 系统音色（如 longanhuan）：只支持 情感/角色/场景 instruction
        - 克隆音色（voice_id on v3-plus）：只支持 方言/小语种 instruction
        - 方言专用音色（如 longanyue_v3）：不需要 instruction（音色自带方言）
        - 混用会 428！系统音色+方言→428，v3克隆音色+情感→428

        v3.5 系列新规则（通过 _resolve_dialect_tts 应用）：
        - 🆕 v3.5 克隆音色支持"任意指令"——方言+情感可以同时控制
        - 这个组合在 _resolve_dialect_tts() 中构建，不走本方法
        """
        parts: List[str] = []

        if not self.is_cloned_voice:
            # 方言专用音色（如 longanyue_v3）完全不支持 Instruct！
            # 传任何 instruction（包括情感）都会 428 错误。
            # 所以使用方言音色时直接返回 None。
            if self.dialect and self.dialect in DIALECT_VOICES:
                return None
            # === 普通系统音色（如 longanhuan）：支持情感/角色/场景 instruction ===
            if self.role:
                parts.append(f"你说话的角色是{self.role}，你说话的情感是{emotion}。")
            elif self.scene:
                parts.append(f"你正在进行{self.scene}，你说话的情感是{emotion}。")
            else:
                parts.append(f"你说话的情感是{emotion}。")
            return " ".join(parts).strip() if parts else None

        # === 用户克隆音色（v3-plus）：只允许方言/小语种 instruction ===
        # 教学说明：用户自己录音克隆的音色走 v3-plus，仍然只支持方言 instruction
        if self.dialect:
            parts.append(f"请用{self.dialect}表达。")

        return " ".join(parts).strip() if parts else None

    # ---- 对话记忆管理 ----

    def ensure_system(self, system_prompt: str) -> None:
        """确保 messages 列表开头有 system 消息（每轮更新）"""
        if not self.messages or self.messages[0].get("role") != "system":
            self.messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            self.messages[0]["content"] = system_prompt

    def add_user(self, text: str) -> None:
        """添加用户消息"""
        self.messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        """添加助手回复"""
        self.messages.append({"role": "assistant", "content": text})
        self._trim()

    def pop_last_user_if_orphan(self) -> bool:
        """
        如果 messages 末尾是 user 消息（没有对应的 assistant 回复），就删掉它。

        教学说明（为什么需要这个方法）：
        - chat_answer_stream() 的流程：先 add_user() → 流式生成 → 最后 add_assistant()
        - 如果用户打断 AI（CancelledError），add_assistant() 永远不会执行
        - 这导致 messages 末尾留下一个"孤儿" user 消息
        - 多次打断后：[system, user1, assistant1, user2, user3, user4, ...]
        - 连续多个 user 消息会让 LLM API 报错或回复混乱
        - 所以在被打断时调用此方法，清理掉那个孤儿 user 消息

        返回值：True = 确实删掉了一条, False = 末尾不是 user 或列表为空
        """
        if self.messages and self.messages[-1].get("role") == "user":
            removed = self.messages.pop()
            print(f"[STATE] 清理孤儿用户消息: \"{removed['content'][:30]}...\"", flush=True)
            return True
        return False

    def _trim(self) -> None:
        """
        修剪对话历史，只保留最近 max_turns 轮。

        教学说明：
        - system 消息永远保留（第一条）
        - 每轮 = 1条user + 1条assistant = 2条消息
        - 所以保留 max_turns * 2 条非 system 消息
        """
        if not self.messages:
            return
        # 分离 system 消息
        system = self.messages[:1] if self.messages[0].get("role") == "system" else []
        rest = self.messages[1:] if system else self.messages
        # 只保留最近的 N 条
        keep = self.max_turns * 2
        rest = rest[-keep:]
        self.messages = system + rest
