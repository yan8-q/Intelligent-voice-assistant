# workflows.py — 工作流执行器
# 职责：根据意图路由结果执行具体操作（重置/方言/角色/克隆/聊天）
# 教学重点：Workflows 是"动作执行层"，Dispatcher 是"决策层"

from __future__ import annotations

from voice_core.dispatcher import IntentDispatcher, RouteDecision
from voice_core.cosyvoice import CosyVoiceEngine
from voice_core.state import ChatState


class Workflows:
    """
    工作流执行器：根据 RouteDecision 执行对应的操作。

    教学说明（为什么要分 Dispatcher 和 Workflows）：
    - Dispatcher 负责"理解"用户说了什么（意图识别）
    - Workflows 负责"执行"具体操作（状态修改 + 回复生成）
    - 分开的好处：意图识别逻辑和执行逻辑解耦，方便测试和维护

    注意：克隆和流式对话在 app_main.py 中直接处理（因为涉及 WebSocket），
    这里的 run() 主要处理简单的非流式场景。
    """

    def __init__(self, api_key: str, base_url: str, cosy: CosyVoiceEngine, dispatcher: IntentDispatcher) -> None:
        self.cosy = cosy
        self.dispatcher = dispatcher

    def run(self, decision: RouteDecision, raw_user_text: str, state: ChatState) -> str:
        """
        执行工作流，返回文本回复。

        教学说明：
        - reset: 恢复默认设置
        - dialect: 设置方言模式
        - role_scene: 设置角色/场景模式
        - default: 普通聊天（非流式）
        - clone: 这里不处理（在 app_main.py 中处理，因为需要 ESP32 录音）
        """

        # === 恢复默认 ===
        if decision.intent == "reset":
            state.reset_to_default()
            return "好的，已恢复默认模式和默认音色。"

        # === 设置方言 ===
        if decision.intent == "dialect":
            if decision.dialect:
                from voice_core.state import normalize_dialect
                state.dialect = normalize_dialect(decision.dialect)
            if not decision.query:
                return f"好的，接下来我会用{state.dialect}和你聊。"
            # 有具体问题，直接回答
            return self.dispatcher.chat_answer(decision.query, state, decision.emotion)

        # === 设置角色/场景 ===
        if decision.intent == "role_scene":
            if decision.role:
                state.role = decision.role
            if decision.scene:
                state.scene = decision.scene
            if decision.style_hint:
                state.style_hint = decision.style_hint
            if not decision.query:
                return "好的，收到。你想让我说点什么？"
            return self.dispatcher.chat_answer(decision.query, state, decision.emotion)

        # === 默认聊天 ===
        query = decision.query or raw_user_text
        return self.dispatcher.chat_answer(query, state, decision.emotion)
