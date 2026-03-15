# dispatcher.py — 意图分发器 + LLM 对话
# 职责：判断用户意图（聊天/克隆/方言/角色/重置）+ 生成回复
# 教学重点：用 OpenAI 兼容接口的 tool_choice 让 LLM 自动选择工作流

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Iterator

from openai import OpenAI

from voice_core.state import ChatState, Emotion

# ==================== 常量定义 ====================

# 情绪列表（LLM 从中选择，用于控制 TTS 语气）
EMOTIONS = ["neutral", "fearful", "angry", "sad", "surprised", "happy", "disgusted"]

# 支持的方言列表（CosyVoice 支持的方言）
DIALECTS = [
    "广东话", "东北话", "甘肃话", "贵州话", "河南话", "湖北话", "江西话", "闽南话", "宁夏话",
    "山西话", "陕西话", "山东话", "上海话", "四川话", "天津话", "云南话"
]

# 场景和角色列表
SCENES = ["闲聊对话", "比赛解说", "深夜电台广播", "剧情解说", "诗歌朗诵", "科普知识推广", "产品推广", "脱口秀表演"]
ROLES = ["温和客服"]


@dataclass
class RouteDecision:
    """
    路由决策结果：LLM 分析用户输入后的判断。

    教学说明：
    - intent: 意图类型，决定走哪个工作流
    - emotion: 情绪类型，影响 TTS 语气
    - query: 实际要回答的问题（可能和原始输入不同）
    - dialect/scene/role/style_hint: 各种模式参数
    """
    intent: str  # default | clone | dialect | role_scene | reset
    emotion: Emotion = "neutral"
    query: str = ""
    dialect: Optional[str] = None
    scene: Optional[str] = None
    role: Optional[str] = None
    style_hint: Optional[str] = None


def _safe_json_load(s: str) -> Dict[str, Any]:
    """安全解析 JSON，失败返回空字典"""
    try:
        return json.loads(s)
    except Exception:
        return {}


class IntentDispatcher:
    """
    意图分发器：分析用户输入 → 选择工作流 → 生成回复。

    教学说明（核心设计）：
    1. route(): 用 LLM 的 function calling 判断用户意图
       - 定义 5 个"工具函数"（workflow_default/clone/dialect/role_scene/reset）
       - LLM 根据用户输入自动选择调用哪个工具
       - 这比手写关键词匹配智能得多

    2. chat_answer_stream(): 流式生成文本回复
       - 逐 token 返回（yield），配合流式 TTS 实现"边想边说"
       - 大幅降低首次响应延迟
    """

    def __init__(self, api_key: str, base_url: str, router_model: str, text_model: str) -> None:
        # 使用 OpenAI 兼容接口连接阿里云百炼
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.router_model = router_model  # 路由模型（轻量，如 qwen-turbo）
        self.text_model = text_model      # 对话模型（可以和路由模型相同）
        self.tools = self._build_tools()  # 工具定义列表

    def _build_tools(self) -> List[Dict[str, Any]]:
        """
        构建 function calling 的工具定义。

        教学说明：
        - 每个工具对应一个意图/工作流
        - LLM 会根据用户输入的语义自动选择调用哪个工具
        - 比如用户说"用四川话聊天"，LLM 会选择 workflow_dialect
        """
        return [
            # 工具1：默认聊天
            {"type": "function", "function": {"name": "workflow_default",
                "description": "默认模式：常规问答/闲聊，没有触发其它模式时使用。",
                "parameters": {"type": "object",
                    "properties": {
                        "emotion": {"type": "string", "enum": EMOTIONS},
                        "query": {"type": "string"}
                    },
                    "required": ["emotion", "query"]}}},

            # 工具2：音色克隆
            {"type": "function", "function": {"name": "workflow_clone",
                "description": "模仿/克隆音色：当用户说'模仿/学一下xx的音色/克隆音色/用xx的音色说'等时使用。",
                "parameters": {"type": "object",
                    "properties": {
                        "emotion": {"type": "string", "enum": EMOTIONS},
                        "query": {"type": "string"},
                        "seconds": {"type": "integer", "default": 7}
                    },
                    "required": ["emotion"]}}},

            # 工具3：方言模式
            {"type": "function", "function": {"name": "workflow_dialect",
                "description": "方言模式：当用户说'用<方言>和我聊/用<方言>说'等时使用。",
                "parameters": {"type": "object",
                    "properties": {
                        "emotion": {"type": "string", "enum": EMOTIONS},
                        "dialect": {"type": "string", "enum": DIALECTS},
                        "query": {"type": "string"}
                    },
                    "required": ["emotion", "dialect"]}}},

            # 工具4：角色/场景模式
            {"type": "function", "function": {"name": "workflow_role_scene",
                "description": "角色/场景模式：'你是xx/在xx场景/唱rap/脱口秀…'等。",
                "parameters": {"type": "object",
                    "properties": {
                        "emotion": {"type": "string", "enum": EMOTIONS},
                        "role": {"type": "string", "enum": ROLES},
                        "scene": {"type": "string", "enum": SCENES},
                        "style_hint": {"type": "string"},
                        "query": {"type": "string"}
                    },
                    "required": ["emotion"]}}},

            # 工具5：恢复默认
            {"type": "function", "function": {"name": "workflow_reset",
                "description": "恢复默认：'换回默认模式/恢复默认音色/别模仿了'等。",
                "parameters": {"type": "object",
                    "properties": {"emotion": {"type": "string", "enum": EMOTIONS}},
                    "required": ["emotion"]}}},
        ]

    def route(self, user_text: str, state: ChatState) -> RouteDecision:
        """
        意图路由：分析用户输入，决定走哪个工作流。

        教学说明：
        - 把当前状态（是否已克隆、当前方言等）告诉 LLM
        - LLM 结合上下文选择最合适的工具
        - temperature=0.1 让选择更确定性（不要太随机）
        """
        system = (
            "你是语音助手项目的意图分发器。"
            "从工具中选择最合适的工作流，并给出emotion、query等参数。"
        )
        # 把当前状态传给 LLM，帮助它做更准确的判断
        state_hint = {
            "is_cloned_voice": state.is_cloned_voice,
            "dialect": state.dialect,
            "role": state.role,
            "scene": state.scene,
        }

        resp = self.client.chat.completions.create(
            model=self.router_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"当前状态: {json.dumps(state_hint, ensure_ascii=False)}\n用户输入: {user_text}"},
            ],
            tools=self.tools,
            tool_choice="auto",
            temperature=0.1,
        )

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        # 如果 LLM 没有选择任何工具，默认走聊天
        if not tool_calls:
            return RouteDecision(intent="default", emotion="neutral", query=user_text)

        # 解析 LLM 选择的工具和参数
        tc = tool_calls[0]
        fn_name = tc.function.name
        args = _safe_json_load(tc.function.arguments or "{}")

        emotion = str(args.get("emotion", "neutral")).strip()
        if emotion not in EMOTIONS:
            emotion = "neutral"

        # 根据工具名称构建路由决策
        if fn_name == "workflow_default":
            return RouteDecision(intent="default", emotion=emotion, query=str(args.get("query", "")).strip())

        if fn_name == "workflow_clone":
            q = str(args.get("query", "")).strip()
            seconds = args.get("seconds", 7)
            return RouteDecision(intent="clone", emotion=emotion, query=q, style_hint=str(seconds))

        if fn_name == "workflow_dialect":
            return RouteDecision(
                intent="dialect", emotion=emotion,
                query=str(args.get("query", "")).strip(),
                dialect=str(args.get("dialect", "")).strip(),
            )

        if fn_name == "workflow_role_scene":
            role = args.get("role")
            scene = args.get("scene")
            style_hint = args.get("style_hint")
            q = str(args.get("query", "")).strip()
            return RouteDecision(
                intent="role_scene", emotion=emotion, query=q,
                role=str(role).strip() if isinstance(role, str) and role.strip() else None,
                scene=str(scene).strip() if isinstance(scene, str) and scene.strip() else None,
                style_hint=str(style_hint).strip() if isinstance(style_hint, str) and style_hint.strip() else None,
            )

        if fn_name == "workflow_reset":
            return RouteDecision(intent="reset", emotion=emotion)

        # 兜底：走默认聊天
        return RouteDecision(intent="default", emotion=emotion, query=user_text)

    # ==================== LLM 对话生成 ====================

    def _build_chat_system_prompt(self, state: ChatState, emotion: Emotion) -> str:
        """
        构建聊天的 system prompt。

        教学说明：
        - system prompt 告诉 LLM 它是什么角色、该怎么说话
        - 根据当前方言/角色/场景动态调整
        - 未克隆时，方言通过 LLM 文本实现（让 LLM 用方言词汇）
        - 已克隆后，方言通过 TTS instruction 实现（LLM 正常说，TTS 变方言）
        """
        import datetime
        today = datetime.date.today().strftime("%Y年%m月%d日")
        # 教学说明（system prompt 的长度控制策略）：
        # 原来写"简洁"→ LLM 把所有回复压到最短 → 用户说"讲30秒笑话"也只给一句
        # 改成"默认2-4句"→ 日常对话自然简短，用户要求长内容时 LLM 会服从
        # 关键原则：system prompt 是"宪法级"指令，LLM 会优先服从它而非用户消息
        system = (
            "你是语音对话助手。输出要口语化，适合直接朗读。\n"
            "默认回复2到4句话。如果用户要求更详细或更长的内容，按用户要求的长度来。\n"
            f"今天的日期是{today}。\n"
            f"本轮情感类型：{emotion}。请用对应语气表达（用词/语气/标点体现）。\n"
            "如果问题不清楚，只问1个最关键的澄清问题。\n"
            "不要使用 emoji 表情符号。\n"
        )

        # 方言处理：
        # - 有方言专用音色（如 longanyue_v3 粤语）：音色自带方言发音，LLM 用标准普通话
        # - 没有专用音色且未克隆：靠 LLM 文字模拟方言口吻（如"俺"代替"我"）
        # - 已克隆：方言通过 TTS instruction 实现，LLM 用标准普通话
        from voice_core.state import DIALECT_VOICES
        if state.dialect and not state.is_cloned_voice and state.dialect not in DIALECT_VOICES:
            system += f"\n接下来请用{state.dialect}的口吻表达，可夹带少量典型方言词汇。\n"

        if state.role:
            system += f"\n你正在扮演：{state.role}。\n"
        if state.scene:
            system += f"\n当前场景：{state.scene}。\n"

        # 特殊风格提示
        if state.style_hint:
            hint = state.style_hint.lower()
            if "rap" in hint or "押韵" in hint or "唱" in hint:
                system += "\n用户想听中文rap：输出 8~16 行，每行尽量押韵，不要写括号舞台说明。\n"
            if "脱口秀" in hint:
                system += "\n用户想听脱口秀：输出 120~200 字，包含一个包袱或反转。\n"

        return system

    def chat_answer(self, user_text: str, state: ChatState, emotion: Emotion) -> str:
        """非流式对话：一次性返回完整回复"""
        system = self._build_chat_system_prompt(state, emotion)
        state.ensure_system(system)
        state.add_user(user_text)

        resp = self.client.chat.completions.create(
            model=self.text_model,
            messages=state.messages,
            temperature=0.7,
        )
        ans = (resp.choices[0].message.content or "").strip()
        if ans:
            state.add_assistant(ans)
        return ans

    def chat_answer_stream(self, user_text: str, state: ChatState, emotion: Emotion) -> Iterator[str]:
        """
        流式对话：逐 token 返回回复。

        教学说明（流式的好处）：
        - 非流式：等 LLM 生成完整回复 → 再 TTS → 再播放（延迟 3-5 秒）
        - 流式：LLM 边生成 → TTS 边合成 → 边播放（延迟 < 1 秒）

        yield 关键字让这个函数变成"生成器"，调用方用 for 循环逐个获取 token。
        """
        system = self._build_chat_system_prompt(state, emotion)
        state.ensure_system(system)
        state.add_user(user_text)

        # 教学说明（异常处理策略）：
        # LLM API 可能因 网络超时 / token超限 / 429限流 / 服务异常 而失败
        # 不加 try-except → 异常裸抛到调用方 → 日志看不出错误来源
        # 加了之后：日志明确记录失败原因，已收到的部分回复仍然保存到对话记忆
        try:
            stream = self.client.chat.completions.create(
                model=self.text_model,
                messages=state.messages,
                temperature=0.7,
                stream=True,  # 开启流式输出
            )
        except Exception as e:
            print(f"\n[LLM] API 调用失败: {e}", flush=True)
            raise  # 重新抛出，让调用方知道出错了

        parts: List[str] = []
        print("[ASSIST] ", end="", flush=True)

        try:
            for chunk in stream:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if not delta:
                    continue
                parts.append(delta)
                print(delta, end="", flush=True)  # 终端实时打印
                yield delta  # 返回给调用方
        except Exception as e:
            # 流迭代中断（网络断开、服务端关闭连接等）
            # 已收到的部分回复仍然保存，不至于丢失整段对话
            print(f"\n[LLM] 流迭代中断: {e}", flush=True)
            raise
        finally:
            print()  # 换行
            # 教学说明：即使中途出错，已收到的部分文本也要存入对话记忆
            # 否则 messages 里有 user 没有 assistant → 下次对话 LLM 会困惑
            final = "".join(parts).strip()
            if final:
                state.add_assistant(final)  # 写入对话记忆
