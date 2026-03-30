"""
RAG 系统量化评估测试脚本 (Step 10)。

【面试亮点 / 核心方法】：
向面试官展示强悍的“闭环工程能力与数据驱动思维”。主动引入 Ragas 评估框架。
包含 5 个代表性测试样例，主要评估两个核心维度：
1. 评估【Context Precision (上下文相关性)】：证明“混合检索 + RRF + Reranker”精准捞出了相关片段拿到了高分。
2. 评估【Faithfulness (忠实度)】：测量系统生成的自然语言回复是否真的忠于被检索上来的文献，克服“幻觉 (Hallucination)”。
"""

import os
import json
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset

# 确保加载环境变量以调用大模型 API 做 LLM-as-a-Judge 评估
load_dotenv()

# 设置 Ragas 使用的评价大模型（必须兼容 OpenAI API 格式）
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_COMPAT_BASE_URL")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_COMPAT_API_KEY")

# 引入 Ragas 核心指标
try:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from ragas import evaluate
    import warnings
    warnings.filterWarning = warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    from ragas.metrics import context_precision, faithfulness, answer_relevancy
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    print("⚠️ 缺少 ragas 或 langchain-openai，请执行: pip install ragas langchain-openai datasets")
    exit(1)

# 构建精选测试集（Ground Truth）
TEST_QUESTIONS = [
    {
        "question": "什么是球差？",
        "ground_truth": "球差是指近轴光线与边缘光线经过同一个球面折射后，不能汇聚在同一点的现象。"
    },
    {
        "question": "Zemax中怎么看操作数？",
        "ground_truth": "在 Zemax 的评价函数编辑器 (Merit Function Editor) 中，可以通过输入具体操作数(如TTHI, TRAC等)并按回车查看对应的功能和参数设置。"
    },
    {
        "question": "阿贝数的作用是什么？",
        "ground_truth": "阿贝数代表材料的色散特性，主要用来衡量光学玻璃在不同波长下的折射率变化。"
    },
    {
        "question": "如何设计一个大光圈镜头？",
        "ground_truth": "设计大光圈镜头时通常需要复杂的光学结构(例如双高斯结构)，并且重点校正球差与彗差。"
    },
    {
        "question": "解释下赛德尔像差包含哪些？",
        "ground_truth": "赛德尔像差(Seidel Aberrations)主要包含五种：球差、彗差、像散、场曲和畸变。"
    },
    {
        "question": "什么是MTF？",
        "ground_truth": "调制传递函数（MTF，Modulation Transfer Function）用来定量表示光学系统对各个空间频率上的细节解析能力和对比度转换能力。"
    },
    {
        "question": "Zemax中如何设置玻璃材料？",
        "ground_truth": "在镜头数据编辑器(Lens Data Editor)的Glass列中，直接输入材料名称（如N-BK7）或双击该单元格打开玻璃库选择所需的材料。"
    },
    {
        "question": "什么是数值孔径(NA)？",
        "ground_truth": "数值孔径（Numerical Aperture, NA）表征了镜头收集光线的能力，它是介质折射率与边缘光线角度正弦值的乘积。"
    },
    {
        "question": "光学设计中的畸变对图像有什么影响？",
        "ground_truth": "畸变会引起像形状的变形，它不影响图像的清晰度，但会使本来直线的物体在像面上显现出弯曲的形式（如桶形畸变或枕形畸变）。"
    },
    {
        "question": "OpticStudio中默认的视场角类型有哪些？",
        "ground_truth": "OpticStudio中常见的视场类型包括角度(Angle)、物高(Object Height)、近轴像高(Paraxial Image Height)和实际像高(Real Image Height)。"
    }
]

def generate_test_results():
    """
    通过调用真实构建的 Agent/RAG 管道，自动生成每个问题的预测答案和召回上下文。
    """
    from tools.rag_tool import search_optics_manual
    
    print("🚀 正在收集 RAG 系统预测数据...")
    # 适配 Ragas 0.2+ 最新的字段名规范：
    # question -> user_input
    # answer -> response
    # contexts -> retrieved_contexts
    # ground_truth -> reference
    data = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": []}
    
    for item in TEST_QUESTIONS:
        query = item["question"]
        print(f"👉 提问: {query}")
        
        # 1. 模拟系统检索 (记录 Contexts)
        try:
            retrieval_output = search_optics_manual.invoke(query)
            # 简单按照分隔符提取文档（实际项目中可以改写返回列表格式的专门评估接口）
            contexts = [c for c in retrieval_output.split("--- 文档片段") if c.strip()][:3]
        except Exception as e:
            print(f"检索失败: {e}")
            contexts = ["检索失败或无知识库环境。"]
            
        # 2. 模拟 LLM 根据 Contexts 回答
        # (这里为了简化依赖，直接用一个直出的 LLM，实际应调用 agent/graph.py 的图)
        judge_llm = ChatOpenAI(
            model_name=os.getenv("LLM_MODEL", "deepseek-chat"),
            temperature=0
        )
        prompt = f"基于以下文档回答问题。如果文档无关，请回答不知道。\n\n文档：{contexts}\n\n问题：{query}"
        answer = judge_llm.invoke(prompt).content
        
        data["user_input"].append(query)
        data["response"].append(answer)
        data["retrieved_contexts"].append(contexts)
        data["reference"].append(item["ground_truth"])
        
    return Dataset.from_dict(data)


def run_evaluation():
    # 数据集准备
    dataset = generate_test_results()
    
    # 评测所需模型对象
    eval_llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_COMPAT_API_KEY"),
        base_url=os.getenv("OPENAI_COMPAT_BASE_URL"),
        model_name=os.getenv("LLM_MODEL", "deepseek-chat"), 
        temperature=0
    )
    # 这里为了防断网/没有外部向量 API，直接复用当前离线目录的 bge-small-zh-v1.5
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bge_model_path = os.path.join(base_dir, "models", "bge-small-zh-v1.5")
    
    eval_embeddings = HuggingFaceEmbeddings(
        model_name=bge_model_path,
        encode_kwargs={"normalize_embeddings": True},
    )
    
    print("\n🔍 开始评测核心指标 (Faithfulness, Context Precision)...")
    
    # 执行评估
    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy
        ],
        llm=eval_llm,
        embeddings=eval_embeddings,
        raise_exceptions=False, # 如果遇到大模型调用异常，Ragas 会继续执行并填充 NaN。如果想调查报错原因，可改为 True。
    )
    
    df = result.to_pandas()
    
    print("\n📊 评估结果汇总:")
    
    # 动态匹配列名（兼容不同版本的 Ragas 返回的键名问题）
    available_cols = df.columns.tolist()
    
    # 尝试取出我们关心的列，如果不存在则忽略，避免 KeyError
    target_cols = ["question", "user_input", "context_precision", "faithfulness", "answer_relevancy"]
    display_cols = [c for c in target_cols if c in available_cols]
    
    if display_cols:
        print(df[display_cols].head(5))
    else:
        print(df.head(5))
    
    # 保存结果
    os.makedirs("eval/results", exist_ok=True)
    try:
        df.to_csv("eval/results/ragas_report.csv", index=False)
        print("\n✅ 详细测试报告已保存至 eval/results/ragas_report.csv")
    except Exception as e:
        print(f"\n⚠️ 保存测试报告失败: {e}")


if __name__ == "__main__":
    run_evaluation()
