"""
主前端应用程序 (Step 8)。

【面试亮点 / 核心方法】：
1. 采用了 Streamlit 快速构建交互式 Web UI。
2. 引入了“可解释性”设计：通过 `st.expander` (折叠面板) 截获并展示 Agent 的中间思考过程（Tool Calls、SQL语句、检索来源等）。这在工业落地中极大地提升了用户对 AI 系统的信任度，也是面试官非常看重的“工程化思维与黑盒透明化”。
"""
import warnings
warnings.filterwarnings("ignore", message=".*Accessing `__path__`.*")

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent.graph import graph

GRAPH_TIMEOUT_SECONDS = 240


st.set_page_config(page_title="OptiAgent (光学设计智能体)", layout="wide")

st.title("🔭 OptiAgent: 光学设计智能体")
st.write("欢迎使用 OptiAgent！我可以查阅光学手册、操作光学软件、查询光学玻璃参数。")

# 初始化前端对话历史记忆
if "messages" not in st.session_state:
    st.session_state.messages = []

# 初始化供 LangGraph 底层流转使用的原始 Message 对象列表历史
if "agent_state_messages" not in st.session_state:
    st.session_state.agent_state_messages = []

# 渲染已有的对话
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        with st.chat_message("assistant"):
            if "thought_process" in msg and msg["thought_process"]:
                with st.expander("🤔 AI 思考过程 & 检索调用记录"):
                    st.markdown(msg["thought_process"])
            st.write(msg["content"])

user_query = st.chat_input("请输入您的问题（例如：什么是球差？ 或者 找折射率大于1.7的肖特玻璃有哪些？）")
if user_query:
    # 记录并渲染用户输入
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # 将用户的输入转化为 LangChain 的 HumanMessage，用于投喂给后台大脑
    user_msg = HumanMessage(content=user_query)
    st.session_state.agent_state_messages.append(user_msg)

    with st.status("🧠 Agent 正在思考和规划...", expanded=True) as status_box:
        new_messages = []
        try:
            # 采用流式输出更新UI思考过程
            for event in graph.stream({"messages": st.session_state.agent_state_messages}, stream_mode="updates"):
                for node_name, node_state in event.items():
                    if node_name == "agent":
                        status_box.update(label="🤖 大脑正在分析和推理...")
                        status_box.write("✨ 智能体进行逻辑推导...")
                    elif node_name == "tools":
                        status_box.update(label="🔧 正在使用外挂工具查询数据...")
                        # 尝试提取刚调用的工具信息
                        msgs = node_state.get("messages", [])
                        if not isinstance(msgs, list):
                            msgs = [msgs]
                        if msgs:
                            last_msg = msgs[-1]
                            if str(type(last_msg).__name__) == "ToolMessage":
                                status_box.write(f"- 收到来自 `{last_msg.name}` 的返回结果。")
                    
                    if "messages" in node_state:
                        msgs = node_state["messages"]
                        if not isinstance(msgs, list):
                            msgs = [msgs]
                        new_messages.extend(msgs)
            
            status_box.update(label="✅ 执行完成！", state="complete", expanded=False)
        except Exception as e:
            err_msg = f"调用后端智能体失败：{e}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": err_msg,
                "thought_process": "",
            })
            status_box.update(label="❌ 执行出现异常", state="error")
            with st.chat_message("assistant"):
                st.error(err_msg)
            st.stop()
    
    # 将网络中产生的新行为和新答复记录下来
    st.session_state.agent_state_messages.extend(new_messages)
    
    thought_log = ""
    final_answer = ""
    
    import re

    # 动态解析产生的新消息：是大模型生成的中间推导(AIMessage)，还是工具干活吐出来的结果(ToolMessage)？
    for m in new_messages:
        if isinstance(m, AIMessage):
            content_str = m.content if isinstance(m.content, str) else ""
            
            # 兼容：部分大模型API直接通过额外字段返回 reasoning_content (如 o1 或 标准接入的 deepseek-reasoner)
            reasoning_content = m.additional_kwargs.get("reasoning_content", "")
            if not reasoning_content and hasattr(m, 'response_metadata'):
                reasoning_content = m.response_metadata.get("model_extra", {}).get("reasoning_content", "") or m.response_metadata.get("reasoning_content", "")
            
            if reasoning_content:
                thought_log += f"**🧠 模型内部推理思考:**\n{reasoning_content}\n\n"

            # 兼容：部分大模型API会将思考过程写在正文的 <think>...</think> 中
            think_match = re.search(r'<think>(.*?)</think>', content_str, flags=re.DOTALL)
            if think_match:
                think_content = think_match.group(1).strip()
                if not reasoning_content:
                    thought_log += f"**🧠 模型内部推理思考:**\n{think_content}\n\n"
                # 剔除原文中的 think 标签及其内部文字，让最终呈现给用户的答案更干净
                content_str = re.sub(r'<think>.*?</think>', '', content_str, flags=re.DOTALL).strip()
            
            # 捕获模型想要使用 Tool 的念头
            if hasattr(m, 'tool_calls') and m.tool_calls:
                for tc in m.tool_calls:
                    thought_log += f"**🛠️ 决定调用工具:** `{tc['name']}`\n\n"
                    thought_log += f"**📥 工具输入参数:** `{tc['args']}`\n\n"
            # 捕获模型最后给出人类的话
            if content_str:
                final_answer += content_str + "\n"
        elif isinstance(m, ToolMessage):
            # 捕获工具最终传回来的数据资料（为了避免太长刷屏，做个截断截取）
            content_str = m.content if isinstance(m.content, str) else str(m.content)
            thought_log += f"**📤 工具返回结果缩略:**\n```text\n{content_str[:500]}...\n```\n\n---\n"
    
    if not final_answer.strip():
        final_answer = "对不起，我虽然执行了工具，但未能成功生成最终文字回复。"

    # 【状态更新揭秘】
    # 最新状态已经通过 extend(new_messages) 追加到 agent_state_messages 中了。
    # 因为底层图流转是以 agent_state_messages 为共享总线的，前端的 st.session_state.messages 
    # 只是拿来构建 UI界面的气泡而已，底层状态链不可被破坏。

    # 包装好这一轮的文字与“白盒思考日志”，推入前端聊天列表
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "thought_process": thought_log
    })

    # 在界面上实时输出本次的大模型回答与过程
    with st.chat_message("assistant"):
        if thought_log:
            with st.expander("🤔 AI 思考过程 & 检索调用记录"):
                st.markdown(thought_log)
        st.write(final_answer)
