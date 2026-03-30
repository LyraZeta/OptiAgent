"""
智能体工作流设计核心状态机 (Step 6 & 7)。

【面试亮点 / 核心方法】：
1. 【先进的框架范式】：告别 LangChain 老旧且极难 debug 的 `AgentExecutor` 黑盒，全面拥抱 【LangGraph 细粒度有向图状态流】。
2. 【原生工具调用】：采用大模型原生的 `bind_tools` (Tool Calling / Function Calling 功能)，抛弃了原先靠大模型生成 JSON 解释的硬编码方式，调用极其稳定。
3. 【可视化图节点与流路由】：清晰定义图论层面的节点（Nodes：大模型主推理点 agent_node 和 工具执行点 tool_node）和条件边（Conditional Routing： `should_continue` 判断逻辑）。让整个复杂 AI 任务的流转不仅性能卓越，还能完美落成项目架构图，是今年最流行的 Agent 端到端组装写法。
"""
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import os
from dotenv import load_dotenv

load_dotenv()

# 引入状态定义和工具组件
from agent.state import AgentState
from tools.rag_tool import search_optics_manual

# 初始化工具列表
# search_optics_manual：用于语义检索光学原理和PDF公式（RAG模式）
tools = [search_optics_manual]

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3-flash-preview-free")
OPENAI_COMPAT_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "https://aihubmix.com/v1")
OPENAI_COMPAT_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY")
LLM_TIMEOUT_SECONDS = 240
LLM_MAX_RETRIES = 1

# Step 6: 定义Tool Calling绑定到主要大模型
# 这里使用您配置的 gemini-3-flash-preview-free，通过 bind_tools 为大语言模型赋予工具识别与调用能力。
llm = ChatOpenAI(
    model_name=LLM_MODEL,
    temperature=0,
    openai_api_base=OPENAI_COMPAT_BASE_URL,
    openai_api_key=OPENAI_COMPAT_API_KEY,
    request_timeout=LLM_TIMEOUT_SECONDS,
    max_retries=LLM_MAX_RETRIES,
)

# 【核心概念】：`bind_tools` (原生工具绑定)
# 这一步是使得传统闲聊模型升级为 Agent 的临门一脚。
# 底层实现是把 python 函数的 params schema（通过 Pydantic 推导）转换为对应的 JSON Schema 结构，
# 注册进入大模型 API 的 params.tools 发送给服务端。告诉模型：“你有这些额外的外挂抓手”。
llm_with_tools = llm.bind_tools(tools)

# Step 7: 绘制 LangGraph 状态机及各节点处理逻辑

def agent_node(state: AgentState):
    """
    大模型思考节点：
    读取当前的 message 历史，提交给已被赋予工具调用能力的大模型。
    大模型看情况可能回复普通文本，也可能触发 function call 请求。
    """
    messages = state["messages"]
    # 每次给大模型补充一个系统核心定位设定，帮助稳定发挥
    sys_prompt = (
        "你是 OptiAgent，一个极其专业的光学设计辅助专家。\n"
        "遇到理论、设计经验、宏命令或软件操作查询时，或是查询具体玻璃型号参数(阿贝数、折射率、密度)时，必须调用搜索文档工具（search_optics_manual）。\n"
        "你可以根据问题类型可选地传入 source_types 参数进行精确路由（如 CamLibrary代表镜头库, Glasscat代表玻璃知识, Macro代表宏命令。注：Manual代表底层手册，会自动包含进所有请求中，无需特意指定）。\n"
        "在返回最终答案时，必须清晰指出你引用了哪些数据来源和元数据（如手册名称、检索策略或 RRF 排序分数等）。"
    )
    
    # 【架构解读】：注入 System Prompt (系统人设提示词) 的技巧
    # 在 LangGraph/LangChain 原生循环中，不要把 SystemMessage 一直留在 `st.session_state` 开头，
    # 那样会让数组变得很长且容易管理混乱。每次 invoke 时通过临时拼装列表：
    # `[('system', sys_prompt)] + messages` 能确保最高优先级的规则指令持续刺激大模型的注意力分布。
    # 由于 messages 中并不自带系统提示词，在实际调用时我们通过这种组合技巧加入
    try:
        response = llm_with_tools.invoke([("system", sys_prompt)] + messages)
    except Exception as e:
        err_text = str(e)
        if "429" in err_text or "free model quota" in err_text.lower():
            raise RuntimeError(
                "大模型调用失败：第三方平台免费额度已用尽（429）。"
                "请在 aihubmix 控制台充值，或更换可用模型后重试。"
            ) from e
        # 返回可读的失败信息，避免上层界面无限等待且无反馈
        raise RuntimeError(
            "大模型调用失败（请检查 OPENAI_COMPAT_API_KEY / OPENAI_COMPAT_BASE_URL / LLM_MODEL 配置）: "
            f"{e}"
        ) from e
    # 将新的模型返回直接 append 进状态字典里
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    条件路由判断器：
    用来判断 agent_node 返回的结果。
    1. 如果包含工具调用请求 (tool_calls) -> 路由走向 "tools"。
    2. 如果是直接的结论性文字，没有工具调用 -> 路由走向最终出口 "__end__"。
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    
    # 【流转网络拓扑精讲】：条件边 Conditional Edge 的意义
    # 有向无环图 (或带环图) 的运转需要“探针”来决定下一步走什么分支。
    # - 当我们通过 invoke 返回的最后一条 `AIMessage` 判断如果携带有 `.tool_calls` 列表，
    #   意味着大模型自己评估后，觉得自己算不出来，必须借助工具。这时候返回的字面量 "tools" 会被框架映射到下一步去执行 `tool_node`。
    # - 否则，说明模型已经通过自身权重知识储备或者根据现有工具返回的数据得出了纯文字解答。
    #   不需要再进行环境互动，直奔内置全局极点 `__end__` 走。
    if last_message.tool_calls:
        return "tools"
    return "end"


# 开始组装状态机
workflow = StateGraph(AgentState)

# 定义核心节点
# 1. 大模型推理节点
workflow.add_node("agent", agent_node)
# 2. 从 prebuilt 库直接加载极轻量级的内置工具执行节点
# 它会自动解析 message 里的 function calling 要求，去执行并收集运行结果转为 tool_message
tool_node = ToolNode(tools)
workflow.add_node("tools", tool_node)

# 设置网络结构和走向边
workflow.set_entry_point("agent")

# 条件路由：从 agent 出发，根据 should_continue() 判断走工具环境还是直接结束对话。
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# 工具在执行完毕拿到结果后，必须无条件回到大模型进行再次反思、提炼、总结
workflow.add_edge("tools", "agent")

# 编译生成状态图大脑
graph = workflow.compile()

