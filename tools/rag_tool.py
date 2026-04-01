"""
文档检索工具 (Step 4)。

【核心方法】：
1. 引入了【混合检索 (Hybrid Search)】模块：将基于 BM25 的稀疏检索（擅长抓取特殊专业词汇或零件号如 N-BK7）与基于特征的密集向量检索（擅长捕捉模糊的语义关联如“色散”、“像差”）相结合，通常权重各占 0.5。
2. 引入了【重排序 (Reranker) 机制】：使用专门的打分模型 `BAAI/bge-reranker-base`（交叉编码器 cross-encoder），对混合检索初步捞回的 Top-10 文档进行语义交叉二次打分，仅将最核心无相关杂音的 Top-3 喂给大模型。解决“召回多但相关性差”造成的 LLM 幻觉和 token 浪费问题，在业界非常硬核。
"""
import os
import logging
import re
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # 使用本地模型路径加载 HuggingFaceEmbeddings，避免在线下载导致的 httpx client closed 错误
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder  # HuggingFaceCrossEncoder 是一个交叉编码器模型类，用于对查询和文档进行交叉编码打分，常用于检索结果的重排序（reranking）。在这个工具中，我们使用了 `BAAI/bge-reranker-base` 模型作为交叉编码器，对混合检索得到的 Top-10 文档进行二次打分，选出最相关的 Top-3 文档返回给大模型，提升最终回答的准确性和相关性。
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

load_dotenv()
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# 配置向量数据库路径
DB_DIR = "data/chroma_db"
ALL_SOURCE_TYPES = {"Manual", "CamLibrary", "Glasscat", "Macro"}
SOURCE_TYPE_ALIASES = {
    "camlibrary": "CamLibrary",
    "camera": "CamLibrary",
    "lens": "CamLibrary",
    "glasscat": "Glasscat",
    "glass": "Glasscat",
    "macro": "Macro",
    "manual": "Manual",
}


@dataclass
class _RetrievalBundle: # 包含检索器实例和检索模式（如 "rrf_hybrid", "vector_only", "bm25_only"）
    retriever: BaseRetriever    # 实际执行检索的对象，可能是向量检索、BM25检索或混合检索
    mode: str   # 检索模式描述字符串


def _parse_source_tokens(raw: str | None) -> set[str]:
    if not raw:
        return set()

    tokens = [t.strip() for t in re.split(r"[,，\s]+", raw) if t.strip()]   # 例子：输入 "CamLibrary, Glasscat Macro"，经过正则分割和清洗后得到 tokens=["CamLibrary", "Glasscat", "Macro"]，再通过别名映射解析成标准化的来源类型集合 {"CamLibrary", "Glasscat", "Macro"}。
    resolved = set()
    for token in tokens:
        if token in ALL_SOURCE_TYPES:
            resolved.add(token)
            continue
        alias = SOURCE_TYPE_ALIASES.get(token.lower())  # 支持大小写不敏感的别名映射，例如 "camera" 会被解析成 "CamLibrary"，"glass" 会被解析成 "Glasscat"。
        if alias:
            resolved.add(alias)
    return resolved # 将用户输入的原始字符串解析成标准化的来源类型集合，支持多种分隔符和别名映射。


def _extract_source_hint(query: str) -> tuple[str, set[str]]:   # tuple的格式是 (cleaned_query, hinted_sources)，cleaned_query 是去掉路由提示后的查询文本，hinted_sources 是从 query 中提取的显式路由提示来源集合。
    """支持在 query 中写 source=CamLibrary,Macro 作为路由提示。"""
    pattern = re.compile(r"(?:#|@)?sources?\s*[:=]\s*([A-Za-z,，\s]+)", re.IGNORECASE)
    matched = pattern.search(query)
    if not matched:
        return query.strip(), set() # 如果没有找到路由提示，直接返回原始查询和一个空的来源集合。

    hinted_sources = _parse_source_tokens(matched.group(1))
    cleaned_query = pattern.sub("", query).strip()  # 例子：输入 "source=CamLibrary,Macro 球差校正怎么设置"，提取到 hinted_sources={"CamLibrary", "Macro"}，cleaned_query="球差校正怎么设置"
    return cleaned_query, hinted_sources


def _normalize_allowed_sources(explicit_sources: set[str]) -> list[str] | None: # 永远添加 Manual
    if not explicit_sources:
        return None

    # 业务硬约束：Manual 永远参与召回，不能被过滤掉。
    effective = set(explicit_sources)
    effective.add("Manual")
    return sorted(effective)


def _build_bm25_retriever_from_chroma(allowed_sources: list[str] | None = None, embeddings=None) -> BM25Retriever | None:
    """从本地 Chroma 持久化库加载文档，构建纯关键词 BM25 检索器（无需联网）。"""
    try:
        local_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        
        all_docs = []
        all_metas = []
        offset = 0
        limit = 500
        
        while True:
            payload = local_store.get(limit=limit, offset=offset, include=["documents", "metadatas"])
            docs = payload.get("documents") or []
            metas = payload.get("metadatas") or []
            
            if not docs:
                break
                
            all_docs.extend(docs)
            all_metas.extend(metas)
            offset += limit
            
            if len(docs) < limit:
                break

        if not all_docs:
            return None

        built_docs = []
        for i, content in enumerate(all_docs):
            if not content:
                continue
            metadata = all_metas[i] if i < len(all_metas) and all_metas[i] else {}
            if allowed_sources and metadata.get("source_type") not in allowed_sources:
                continue
            built_docs.append(Document(page_content=content, metadata=metadata))

        if not built_docs:
            return None

        bm25 = BM25Retriever.from_documents(built_docs)
        bm25.k = 10
        return bm25
    except Exception as e:
        print(f"⚠️ BM25 构建失败: {e}")
        return None


class RRFRetriever(BaseRetriever):
    """RRF 混合检索器：并行融合向量检索与 BM25 结果。"""

    vector_retriever: BaseRetriever | None = None
    bm25_retriever: BaseRetriever | None = None
    k: int = 60 # RRF 参数，控制融合时的衰减因子
    top_k: int = 10

    def _get_relevant_documents(self, query: str, *, run_manager=None): # 这个方法会被 ContextualCompressionRetriever 调用，输入是用户的查询文本，输出是经过 RRF 融合排序后的相关文档列表。
        vector_docs = []
        bm25_docs = []

        if self.vector_retriever is not None:
            try:
                vector_docs = self.vector_retriever.invoke(query)   # 调用向量检索器获取初步的相关文档列表，可能包含一些与查询语义相关但不包含关键词的文档。
            except Exception as e:
                print(f"⚠️ 向量检索失败: {e}")

        if self.bm25_retriever is not None:
            try:
                bm25_docs = self.bm25_retriever.invoke(query)
            except Exception as e:
                print(f"⚠️ BM25 检索失败: {e}")

        if not vector_docs and not bm25_docs:
            return []

        doc_scores = {} # 计算每个文档的 RRF 分数，key 是文档的唯一标识（这里使用 page_content 的 hash），value 是累积的 RRF 分数。由于同一文档可能在向量检索和 BM25 检索中都被召回，因此需要累积分数来体现其综合相关性。
        doc_objects = {}    # 存储文档对象的字典，key 是文档的唯一标识（这里使用 page_content 的 hash），value 是 Document 对象。这样在计算 RRF 分数时可以同时保留文档内容和元信息，方便后续排序和返回给调用者。

        for rank, doc in enumerate(vector_docs):    # 遍历向量检索的结果列表，rank 是文档的排名（从 0 开始），doc 是 Document 对象。对于每个文档，根据其在向量检索中的排名计算 RRF 分数，并累加到 doc_scores 中。同时将文档对象存储在 doc_objects 中，以便后续根据 doc_id 获取文档内容和元信息。
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            rrf_score = 1.0 / (self.k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score

        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            rrf_score = 1.0 / (self.k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score

        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        fused_docs = []
        for doc_id, final_score in sorted_items[: self.top_k]:
            doc = doc_objects[doc_id]
            doc.metadata = {
                **(doc.metadata or {}),
                "rrf_score": round(final_score, 6),
                "search_method": "rrf_hybrid",
            }
            fused_docs.append(doc)

        return fused_docs   # 返回经过 RRF 融合排序后的文档列表，包含了向量检索和 BM25 检索的综合相关性评分，供后续的重排序模块使用。类型是 List[Document]，每个 Document 对象包含 page_content 和 metadata。

def build_advanced_retriever(allowed_sources: list[str] | None = None) -> _RetrievalBundle | None:
    """
    构建高级检索器：包含 混合检索（BM25 + 向量）与 BGE-Reranker 重排序
    """
    if not os.path.exists(DB_DIR):
        print("⚠️ 警告：未找到本地 Chroma 数据库，请先执行数据入库阶段。")
        return None

    # 第一部分：基础向量检索 (Top-10) 与 Embedding 初始化
    vector_retriever = None
    embeddings = None
    try:
        # 使用本地已有的 bge-small 模型路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bge_model_path = os.path.join(base_dir, "models", "bge-small-zh-v1.5")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=bge_model_path,
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)   # 从本地 Chroma 数据库加载向量检索器，注意这里的 embedding_function 需要与 BM25 检索时使用的保持一致，确保两者在同一语义空间中进行检索和融合。
        search_kwargs = {"k": 10}
        if allowed_sources:
            search_kwargs["filter"] = {"source_type": {"$in": allowed_sources}}
        vector_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)    # 构建向量检索器实例，支持根据 allowed_sources 进行过滤，例如如果 allowed_sources=["CamLibrary", "Macro"]，则只会检索 source_type 为 "CamLibrary" 或 "Macro" 的文档。
    except Exception as e:
        print(f"⚠️ 向量模型加载失败，退化为 BM25 检索: {e}")

    # 构建离线可用的 BM25 检索（必须传递相同的 embeddings，防止 Chroma 内部强制下载默认模型导致 bindings 报错）
    bm25_retriever = _build_bm25_retriever_from_chroma(allowed_sources, embeddings)

    # 第二部分：混合检索融合策略（优先使用 RRF）
    retrieval_mode = "single"
    if vector_retriever and bm25_retriever:
        base_retriever = RRFRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            k=60,
            top_k=10,
        )
        retrieval_mode = "rrf_hybrid"   # rrf_hybrid 表示使用了 Reciprocal Rank Fusion 的混合检索策略，融合了向量检索和 BM25 检索的结果，能够兼顾两者的优势，提高召回的相关性和多样性。
    elif vector_retriever:
        base_retriever = vector_retriever
        retrieval_mode = "vector_only"
    elif bm25_retriever:
        base_retriever = bm25_retriever
        retrieval_mode = "bm25_only"
    else:
        print("⚠️ 未能构建可用检索器（向量与 BM25 都失败）。")
        return None

    # 第三部分：BGE-Reranker 重排序 (提取混合检索结果中最符合语义的前 3 条)
    try:
        # 使用本地已有的 reranker 模型路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reranker_local_path = os.path.join(base_dir, "models", "bge-reranker-base")
        
        cross_encoder = HuggingFaceCrossEncoder(
            model_name=reranker_local_path,
        )
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)

        # 注意：ContextualCompressionRetriever 内部会先调用 base_retriever 获取 Top-10 文档，然后将查询与每个文档内容拼接成输入，送给 cross-encoder 模型进行交叉编码打分，最后选出得分最高的 Top-3 文档返回给调用者。
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )
        # 最终返回一个包含了混合检索和重排序机制的检索器实例，以及一个描述当前检索模式的字符串（如 "rrf_hybrid+rerank"），供后续调用时使用。
        return _RetrievalBundle(retriever=compression_retriever, mode=f"{retrieval_mode}+rerank")
    except Exception as e:
        print(f"⚠️ Reranker 加载失败，退回普通检索: {e}")
        return _RetrievalBundle(retriever=base_retriever, mode=retrieval_mode)

# 懒加载检索器（避免首次导入时多次加载模型导致 httpx client closed 等问题）
_retriever_cache = {}   # 用法：_retriever_cache[cache_key] = retrieval_bundle

@tool
def search_optics_manual(query: str, source_types: str = "") -> str:
    """
    【AI Tool 机制解密：大模型的眼睛】
    这是一个被 @tool 装饰器包裹的工具。当这串 Python Docstring 传递给大模型时，
    大模型不仅能看懂这函数的用途（何时调用），还能根据类型标注理解必须传入哪些参数。
    当业务场景判定需要查询本地领域知识库时，模型会生成特定 JSON 请求，触发这里的离线高级检索。
    
    查询光学手册和知识库。
    当你需要回答有关光学理论（如像差、焦距）、光学公式或复杂定义的问题时，调用这个工具。
    输入参数是用户的搜索问题。

    可选参数 source_types 用于路由检索来源（如 "CamLibrary,Glasscat"）。
    注意：Manual 始终会参与检索，不能被过滤掉。

    也支持在 query 中携带路由提示，例如：
    "source=CamLibrary,Macro 球差校正怎么设置"。
    """
    hinted_query, hinted_sources = _extract_source_hint(query)  # 从查询中提取路由提示和清洗后的查询文本
    explicit_sources = _parse_source_tokens(source_types) | hinted_sources  # 例子：如果 source_types="Glasscat" 且 query 中提示 source=CamLibrary,Macro，那么 explicit_sources={"Glasscat", "CamLibrary", "Macro"}
    allowed_sources = _normalize_allowed_sources(explicit_sources) # 永远添加 Manual

    cache_key = tuple(allowed_sources) if allowed_sources else ("__ALL__",)
    retrieval_bundle = _retriever_cache.get(cache_key)  # 例子：第一次调用时 allowed_sources={"CamLibrary", "Macro"}，cache_key=("CamLibrary", "Macro")，检索器构建成功后被缓存；第二次调用时如果 allowed_sources 仍然是 {"CamLibrary", "Macro"}，则直接从 _retriever_cache 中获取之前构建好的检索器实例，避免重复加载模型和构建检索器的开销。
    if retrieval_bundle is None:
        retrieval_bundle = build_advanced_retriever(allowed_sources=allowed_sources)    # 构建检索器实例
        _retriever_cache[cache_key] = retrieval_bundle  # 将构建好的检索器实例缓存起来

    if retrieval_bundle is None:
        return "未能连接到文档知识库。请检查数据是否入库。"

    if not hinted_query:
        return "请提供检索问题内容。"
    
    # 执行高级检索
    results = retrieval_bundle.retriever.invoke(hinted_query)
    
    if not results:
        return "手册中未能找到与该问题相关的内容。"

    # 将检索到的分块拼装后返回给大模型去阅读总结
    formatted_docs = [] # 格式化检索结果，包含文档内容和元信息（如来源类型、来源名称、检索方法、RRF 分数等），以便大模型能够理解每个文档片段的背景和相关性，提升后续回答的准确性和丰富度。
    for i, doc in enumerate(results):
        source = doc.metadata.get("source", "unknown")
        source_type = doc.metadata.get("source_type", "unknown")
        rrf_score = doc.metadata.get("rrf_score", "n/a")
        search_method = doc.metadata.get("search_method", retrieval_bundle.mode)
        formatted_docs.append(
            f"--- 文档片段 {i+1} | source_type={source_type} | source={source} | method={search_method} | rrf_score={rrf_score} ---\n{doc.page_content}"
        )
        
    return "\n\n".join(formatted_docs)
