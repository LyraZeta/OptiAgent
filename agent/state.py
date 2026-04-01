"""
智能体状态定义 (Step 7)。

【核心方法】：
1. 采用 LangGraph 强制推荐的 `TypedDict` 定义状态空间图流转共享状态 (State)。
2. 对对话记录字段 `messages` ，利用了特殊的注解类型结合归约函数 `Annotated[..., operator.add]`（追加模式）。使得状态机在面临“多步思考-调用多次工具”的长程动作时，不仅不丢失上下文，大模型更具备了完美的历史执行记忆。
"""
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage


# -----------------------------------------------------------------------------
# 【架构机制精讲】：
# LangGraph 中的 State 是所有图节点 (Node) 在有向图执行流转期间所共享的“数据总线板”。
# - 强烈推荐使用 `TypedDict`：因为它提供了静态类型提示约束，但不额外增加运行时开销。
# - 最核心魔法：`Annotated[Sequence[BaseMessage], operator.add]` 中的合并器 (Reducer)。
#   如果不声明 `operator.add` 归约策略，一旦大模型或者工具节点返回 `{"messages": [new_item]}`，
#   默认行为会整体覆写整个消息列表导致上下文记忆清零。
#   增加了附加归约后，底层的 State Channel 会自动执行 `state["messages"].extend([new_item])` 追加。
#   从而完美保留：Human(用户) -> AI(思考策略/tool_calls) -> Tool(执行结果文本) -> AI(总结) 的完整执行链。
# -----------------------------------------------------------------------------
class AgentState(TypedDict):
    """
    智能体状态定义。
    此字典用于在整个 LangGraph 循环流转（大模型推理节点 -> 工具执行节点）中保存核心状态。
    `messages` 列表通过 annotated 使用 operator.add 进行追加（不覆盖），完整保留多轮执行历史和工具的返回信息。
    """
    # 消息记录（用户输入、大模型思考过程记录、工具执行结果、最后回答）
    messages: Annotated[Sequence[BaseMessage], operator.add]

