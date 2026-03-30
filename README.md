# 🔭 OptiAgent: 离线驱动的光学设计领域 AI 智能体

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Workflow-orange)](https://python.langchain.com/docs/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/Chroma-Vector%20DB-blueviolet)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OptiAgent** 是一个专为光学工程师和研究人员打造的**复合型 RAG (检索增强生成) 智能体系统**。采用基于 **LangGraph** 有向图的状态机架构，专为处理大规模复杂光学系统手册、镜头玻璃库及宏代码参数设计。

> 💡 **核心优势：专为 HPC 和军工内网脱水部署设计。** 完整剥离在线 API 依赖，通过自研的 PyMuPDF 启发式逻辑截断引擎解决极度受限断网设备的 RAG 幻觉，无需云端 GPU 也能高速运行。

---

## 🎨 系统架构与核心链路

本项目高度实践了细粒度的节点流转控制，以“所见即所得”的极高工程化水准控制大模型的决策链。

```text
       [用户请求 / 提问: "找折射率大于1.7的肖特玻璃"]
             │
             ▼
    ┌─────────────────┐       (具备 tool_calls 触发条件)      ┌─────────────────┐
    │  LangGraph 🧠   │ ────────────────────────────────────> │  Tool Node 🛠️  │
    │  Agent Node     │                                       │ (RAG 离线检索)  │
    └─────────────────┘ <──────────────────────────────────── └─────────────────┘
      (分析提炼并总结)            返回工具执行文献与 RRF 分数
             │
             ▼
    [Streamlit 前端: 带可解释性白盒展示面板]
```

## 🌟 核心独创功能 (Features)

1. **纯净的离线 HPC 适配 (Offline-First for HPC):**
   完全抛弃了极其臃肿的 `Unstructured` 生态依赖链（往往引发无数 Linux 核心缺失），利用轻量级库加模型推理，可在超算/封闭服务器环境中丝滑独立运行。（自带自动化 Wheels 集成上传脚本）。
2. **启发式隐式脊梁重建引擎 (ISR 抽取引擎):**
   在处理 PDF 数据提取与注入库时，不再“定长硬切”，而是采用类似于 SpineDoc 的思想：**跨页合并章节化知识簇**。利用版面字号分析自动感知文档脊梁防折断，极大提升大模型摄入 RAG 参数信息的完备性。
3. **混合检索引擎与重排 (Hybrid RRF + BGE Reranker):**
   面对光学领域的大量零件与缩写（如 N-BK7, MTF），抛弃单一向量算法的劣势。结合了 **BM25 倒排索引找精准词**与 **BGE 本地小模型寻找语义拓展**。最后经特征倒数关联融合 (RRF)，交由 `bge-reranker-base` 进行二次精排去噪，Top 3 送入模型！
4. **LangGraph 工业级状态流总线:**
   拥抱最新的 `bind_tools`，抛除了传统提示词让大模型强制输出 JSON 造成的崩溃黑盒，拥有原生调用工具抓手的稳定能力，每步皆可审计。
5. **内建自动化评估基准 (Ragas Evaluation):**
   提供独立完整的测试入口，自动化运行多维度的评估（即：不撒谎的忠实度和上下文召回精度），并自动生成 CSV 报告。（兼容最新 Ragas 0.2+ 列名规范）。

---

## 📂 项目模块地图

*   **`data_prep/` (知识摄入引擎):**
    *   `parse_pdf.py`: 【数据管道入口】负责遍历目录下所有的光学手册与文献，进行提取、分块聚合、打上来源标签 Metadata，并将最终的高质量晶体注入本地 ChromaDB。
*   **`tools/` (Agent 手脚扩展):**
    *   `rag_tool.py`: 定义对外暴露的高级搜索函数，包含上述混合搜索与重排序算法，它是整个系统能够“认字”的关键。
*   **`agent/` (大脑控制中枢):**
    *   `state.py`: 利用 LangGraph 定义共享全局的 Annotated Reducer 数据总线。
    *   `graph.py`: 拼装节点与走向边，处理 System Prompt 的每一次激活动作。
*   **`eval/` (系统量化基准):**
    *   `evaluate.py`: 利用 Ragas 框架进行的防幻觉评分工具。
*   **根目录工具与前端:**
    *   `app.py`: Streamlit 用户交互入口（带内部运算面板）。
*   **`hpc_scripts/` (超算离线部署):**
    *   包含所有自动化部署组件脚本。**若需要在无网络的高性能计算 (HPC) 运行，可以使用该文件夹内的脚本**实现底层依赖无网安装、向量权重上传以及多节点任务下发等能力（如 `deploy_wheels.sh` 等）。

---

## 🛠️ 快速开始

### 1. 环境准备

推荐使用 Python 3.10+ 环境。您可以克隆本仓库后：

```bash
git clone https://github.com/your-username/OptiAgent.git
cd OptiAgent

# 使用 pip 快速安装
pip install -r requirements.txt
```

*(也附带了 `optiagent_env.yml` 供 conda 重度使用者快速复刻底层。)*

### 2. 配置大模型接入环境变量

在根目录下创建（或基于现有） `.env` 文件。虽然 RAG 参数查询全离线，但这套系统仍然需要一个智慧极高的核心大脑(如通过 vLLM 本地部署模型 或 第三方 API):

```dotenv
# 主要模型
LLM_MODEL = "模型名称"
OPENAI_COMPAT_BASE_URL = "https://你的接口地址/v1"
OPENAI_COMPAT_API_KEY = "你的 API Key"

# 用于评估模型
EMBEDDING_LLM_MODEL = "模型名称"
EMBEDDING_OPENAI_COMPAT_BASE_URL = "https://你的接口地址/v1"
EMBEDDING_OPENAI_COMPAT_API_KEY = "你的 API Key"
```

### 3. 数据初始化入库 (关键)

准备好您的待读资料（存放入 `data/` 目录下相应的文件夹），执行知识数据编译，该代码会自动进行 PDF 分辨重建与分块嵌入（耗时视文档数量而定）：

```bash
python data_prep/parse_pdf.py
```
> 完成后，`data/chroma_db` 中将拥有完全建立好的离线超维度字典！

### 4. 驱动问答控制台

通过 Streamlit 唤起交互前端：

```bash
streamlit run app.py
```

终端会弹出本地网页端口 (默认 `http://localhost:8501`)，你现在可以向它询问复杂的镜头理论与操作指南，并随时打开其折叠面板监督它的查表细节！

### 5. 高性能计算 (HPC) 离线运行

**若需要在无网络的高性能计算集群运行，可以使用 `hpc_scripts/` 文件夹下提供的环境分发脚本。**
由于超算普遍没有外网，您可以先在本地连网机器下载好所需 Python 环境的源包（`.whl`），然后使用该目录下的脚本（如 `deploy_wheels.sh` 和 `deploy_env_to_server.sh`）以 `rsync` 打包上传并冷部署。

脚本使用环境变量，执行前请配置：
```bash
export HPC_USER=your_username
export HPC_HOST=your_hpc_address

# 随后运行任一部署脚本，如：
bash hpc_scripts/deploy_to_server.sh
```

---

## 💡 开发与量化

您可以随时执行以下命令评估当前模型的回答有效性和文档捞取精度：
```bash
python eval/evaluate.py
```
它会在 `eval/results/` 中生成直观的指标 CSV 报告。

## 📜 许可证 (License)
本项目基于 [MIT License](LICENSE) 发布。
