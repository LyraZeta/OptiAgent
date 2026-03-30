"""
OptiAgent 知识库构建模块 - 文档解析与向量化入库

本模块负责处理 data 目录下的多源知识文档（涵盖光学手册、镜头库、玻璃库、宏预设等），并将清洗与提取后的文本存储到 Chroma 向量数据库中。

其核心特性在于：
1. 完全离线化：避免网络 API 依赖，适应超算或内网 HPC 环境的部署需求。
2. 启发式骨架重建 (ISR, 类似 SpineDoc 思想)：使用 PyMuPDF 对长篇 PDF 进行章节层级的智能切分防截断，保留知识晶体语义。
3. 多模态回退机制：针对纯文本和代码类文档，使用自适应多编码的纯文本读取组件作为基础数据入口。

调用入口：`ingest_all_sources_to_vector_db()`
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 配置基础日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# 常量与路径配置
DATA_ROOT = Path("data")
SOURCE_DIRS = ["Manual", "CamLibrary", "Glasscat", "Macro"]
# SOURCE_DIRS = ["Manual_all", "CamLibrary_all", "Glasscat_all", "Macro_all"]
DB_DIR = "data/chroma_db"
PARSED_MD_DIR = "data/parsed_md"

# 文件类型过滤控制
SKIP_EXTENSIONS = {".exe", ".dll", ".so", ".bin", ".zip", ".tar", ".gz"}
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".xls", ".xlsx",
    ".cfg", ".ses", ".agf", ".tsv", ".json", ".xml", ".html", ".htm"
}

def _print_progress(prefix: str, current: int, total: int, detail: str = "", width: int = 30) -> None:
    """在终端渲染单行进度条输出。"""
    if total <= 0:
        print(f"{prefix}: 0/0")
        return

    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    percent = ratio * 100
    message = f"\r{prefix} [{bar}] {current}/{total} ({percent:5.1f}%)"
    if detail:
        message += f" | {detail}"

    print(message, end="\r" if current < total else "\n", flush=True)

def _save_intermediate_markdown(docs: List[Document], file_path: Path, data_root: Path, output_root: Path) -> Optional[Path]:
    """
    【数据透明度设计】将单文档解析出的中间分块结果（知识晶体）转存为 Markdown 文件，便于审查解析质量。
    这在 RAG (检索增强生成) 系统中非常重要，因为向量入库前的内容决定了模型回答的上限上限（Garbage in, garbage out）。
    
    Args:
        docs (List[Document]): 已经解析提取好 metadata 的 Document 列表。
        file_path (Path): 原文件路径。
        data_root (Path): 数据根目录。
        output_root (Path): 缓存输出根目录。
    """
    if not docs:
        return None

    try:
        rel = file_path.relative_to(data_root)
        out_dir = output_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{file_path.name}.md"

        lines = [
            f"# OpiAgent Parsed Record: {rel.as_posix()}",
            "",
            f"- **Source File**: {file_path.name}",
            f"- **Total Chunks**: {len(docs)}",
            "",
        ]

        for idx, doc in enumerate(docs, start=1):
            meta = doc.metadata or {}
            lines.extend([
                f"## Chunk Block {idx}",
                f"- **Strategy**: {meta.get('chunk_strategy', 'unknown')}",
                f"- **Element Type**: {meta.get('element_type', 'unknown')}",
                f"- **Page/Position**: {meta.get('page_number', 'unknown')}",
                ""
            ])
            
            # 如果是 Spine_Logic 提取，记录它对应的结构章节名
            if "spine_chapter" in meta:
                lines.append(f"- **Spine Framework**: *{meta['spine_chapter']}*\n")
                
            lines.append(f"{doc.page_content.strip()}\n")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path
    except Exception as e:
        logger.warning(f"保存中间 Markdown 缓存记录失败: {file_path.name}, Error: {e}")
        return None

    try:
        content_type = "application/pdf" if file_path.suffix.lower() == ".pdf" else None
        elements = partition(
            filename=str(file_path),
            content_type=content_type,
            strategy="fast",
        )
        return _elements_to_documents(elements, file_path)
    except Exception as e:
        logger.debug(f"Unstructured Partition 解析失败: {file_path}, Error: {e}")
        return []

def _iter_input_files(data_root: Path, source_dirs: List[str]) -> List[Path]:
    """遍历指定数据目录，采集待入库文件清单。"""
    files: List[Path] = []
    for folder in source_dirs:
        folder_path = data_root / folder
        if not folder_path.exists():
            logger.warning(f"跳过不存在的来源目录: {folder_path}")
            continue
        for path in folder_path.rglob("*"):
            if path.is_file():
                files.append(path)
    return sorted(files)

def _text_fallback_loader(file_path: Path) -> List[Document]:
    """兜底解析器：当专用解析组件或 Unstructured 不可用/报错时，回退到系统自适应多编码纯文本读取。"""
    encodings = ["utf-8", "gb18030", "latin-1"]
    for enc in encodings:
        try:
            return TextLoader(str(file_path), encoding=enc).load()
        except Exception:
            continue
    logger.error(f"多编码兜底纯文本读取全部失败: {file_path.name}")
    return []

def _pdf_loader(file_path: Path) -> List[Document]:
    """
    【架构揭秘：ISR 隐式脊梁重建引擎】
    OptiAgent 专有离线解析引擎 (PyMuPDF)。
    由于我们是在本地离线超算环境，大模型 RAG 最怕遇到的是“一段话被暴力按页码/字数切断”导致语义丢失。
    此引擎复刻了 SpineDoc 的隐式逻辑脊梁重建 (ISR) 功能：跨页合并章节化知识簇，防止截断，极大提升完整率。
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("未找到关键依赖 PyMuPDF, 请使用 'pip install PyMuPDF' 配置引擎。")
        return []

    docs: List[Document] = []
    try:
        doc = fitz.open(str(file_path))
        
        # 1. 尝试获取 PDF 原生大纲目录 (TOC) 作为显式逻辑树
        toc = doc.get_toc()
        
        # 2. 如果不存在显式目录，启动 ISR (Implicit Spine Reconstruction) 模块启发式提取脊梁
        if not toc:
            for page_num, page in enumerate(doc, start=1):
                blocks = page.get_text("dict").get("blocks", [])
                for b in blocks:
                    if "lines" not in b:
                        continue
                    for line in b["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            # 基于经验法则：大字号(>11pt)且以特定形式开头的判定为文档级章节锚点
                            if span["size"] > 11 and re.match(r'^(第[一二三四五六七八九十百千万0-9]+[章部分节]|Chapter\s*\d+|\d+(\.\d+)+\s+)', text, re.IGNORECASE):
                                toc.append([1, text, page_num])

        # 3. 按页挂载脊梁属性字典映射
        toc.sort(key=lambda x: x[2]) 
        page_to_chapter = {}
        current_chapter = "未分类综述/引言"
        toc_idx = 0
        
        for page_num in range(1, len(doc) + 1):
            while toc_idx < len(toc) and page_num >= toc[toc_idx][2]:
                current_chapter = toc[toc_idx][1]
                toc_idx += 1
            page_to_chapter[page_num] = current_chapter

        # 4. 执行跨页矩阵融合：确保位于极长逻辑块的语句不会被硬暴力翻页斩断
        chapter_texts = {}
        chapter_pages = {}
        
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            
            ch_title = page_to_chapter.get(page_num, "碎片章节")
            chapter_texts.setdefault(ch_title, []).append(text)
            chapter_pages.setdefault(ch_title, []).append(str(page_num))
        
        # 5. 封装携带精准物理溯源页码的朗链知识晶体
        for chapter_title, texts in chapter_texts.items():
            combined_text = "\n\n".join(texts)
            pages_list = chapter_pages[chapter_title]
            
            page_range = f"{pages_list[0]}-{pages_list[-1]}" if len(pages_list) > 1 else pages_list[0]
                
            docs.append(Document(
                page_content=combined_text,
                metadata={
                    "spine_chapter": chapter_title,
                    "page_number": page_range,
                    "chunk_strategy": "spine_logic_reconstruction",
                    "element_type": "LogicChapter",
                    "loader": "PyMuPDF_ISR_Engine",
                    "source_file": file_path.name,
                }
            ))
    except Exception as e:
        logger.error(f"PyMuPDF 引擎解析发生严重错误: {file_path}, Error: {e}")
        return docs

    return docs

def _load_file(file_path: Path) -> List[Document]:
    """核心路由：负责分发各种数据的处理流。"""
    suffix = file_path.suffix.lower()
    if suffix in SKIP_EXTENSIONS:
        return []

    # 专用拦截器：PDF走自定义高效引擎管道
    if suffix == '.pdf':
        return _pdf_loader(file_path)

    # 纯文本文档兜底为系统自适应多编码读取
    if suffix in TEXT_EXTENSIONS:
        return _text_fallback_loader(file_path)

    return []

def _enrich_metadata(docs: List[Document], file_path: Path, data_root: Path) -> List[Document]:
    """统一向数据块元图注入全局标准字段（来源、来源分类、扩展名等）。"""
    rel_path = file_path.relative_to(data_root).as_posix()
    source_type = rel_path.split("/", 1)[0] if "/" in rel_path else rel_path
    
    for doc in docs:
        doc.metadata = {
            **(doc.metadata or {}),
            "source": rel_path,
            "file_name": file_path.name,
            "source_type": source_type,
            "extension": file_path.suffix.lower(),
        }
    return docs

def ingest_all_sources_to_vector_db(
    data_root: str = str(DATA_ROOT),
    source_dirs: Optional[List[str]] = None,
    db_dir: str = DB_DIR,
    reset_db: bool = True,
    save_intermediate_md: bool = True,
    md_output_dir: str = PARSED_MD_DIR,
) -> None:
    """
    【架构总入口：数据清洗与索引建立】
    将所有模块来源合并检索清洗并实例化向量表征入库更新 ChromaDB。
    步骤：1. 遍历文件 -> 2. _load_file 按扩展名路由引擎提取 -> 3. _save_intermediate_markdown 保存调试文档
    -> 4. _enrich_metadata 注入元数据 -> 5. 字符切分兜底 -> 6. 灌入 HuggingFace 离线 embedding 模型 -> 入库本地。
    """
    root = Path(data_root)
    folders = source_dirs or SOURCE_DIRS
    md_root = Path(md_output_dir)

    files = _iter_input_files(root, folders)
    if not files:
        logger.error(f"未找到可入库的合法源文件，请检查 '{data_root}' 结构！")
        return

    loaded_docs: List[Document] = []
    skipped_files = 0
    saved_md_files = 0

    if save_intermediate_md:
        md_root.mkdir(parents=True, exist_ok=True)

    print(f"🚀 [OptiAgent] 正在启动文档解析核心，预扫文件总数: {len(files)}")

    for idx, f in enumerate(files, start=1):
        file_docs = _load_file(f)
        if not file_docs:
            skipped_files += 1
            _print_progress("解析进度", idx, len(files), f"跳过异常文件: {f.name[:20]}")
            continue

        if save_intermediate_md:
            if _save_intermediate_markdown(file_docs, f, root, md_root):
                saved_md_files += 1

        loaded_docs.extend(_enrich_metadata(file_docs, f, root))
        _print_progress("解析进度", idx, len(files), f"完成吞吐: {f.name[:20]}")

    if not loaded_docs:
        logger.critical("所有文件均解析失败，拒绝执行下沉写入任务。")
        return

    # 对由于回退导致依然过长的切片进行兜底物理强制阻断处理
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", ". ", " ", ""],
    )
    
    print("\n🔪 正在执行字符级边界截断修正 (对于超长兜底文档)...")
    chunks = splitter.split_documents(loaded_docs)

    if reset_db and os.path.exists(db_dir):
        logger.info(f"触发了重新构建策略：清空历史旧库池 {db_dir}")
        shutil.rmtree(db_dir)

    # 动态载入 BGE 中国/通用优化版嵌入模型
    base_dir = Path(__file__).parent.parent.resolve()
    bge_model_path = base_dir / "models" / "bge-small-zh-v1.5"
    
    # 强制在服务端不携带设备分配指令，借由 accelerate 或 huggingface API 自主寻找可用资源（彻底解决 Torch Meta 异常）
    embeddings = HuggingFaceEmbeddings(
        model_name=str(bge_model_path),
        encode_kwargs={"normalize_embeddings": True},
    )

    print("📦 正在生成表征级向并持久化至 Chroma...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir,
    )

    print("-" * 50)
    print(f"✅ 处理统计: 成功扫描 {len(files)} 个，失败被滤除 {skipped_files} 个。")
    print(f"✅ 有效文档集群: {len(loaded_docs)} 个逻辑晶体。")
    if save_intermediate_md:
         print(f"📁 调试中间结果: ({saved_md_files} 个) 已投递至 {md_root}")
    print(f"✅ 系统总计向 DB 承载向量切片: {len(chunks)} 条。")
    print(f"🎉 RAG 向量底层准备就绪！路径锚定 -> {db_dir}")

def process_pdf_manual(pdf_path: str = "data/Manual", db_dir: str = DB_DIR) -> None:
    """历史 API 遗留兼容适配管道接口。"""
    manual_path = Path(pdf_path)
    source_dirs = [manual_path.parent.name] if manual_path.is_file() else ([manual_path.name] if manual_path.exists() else ["Manual"])

    ingest_all_sources_to_vector_db(
        data_root="data",
        source_dirs=source_dirs,
        db_dir=db_dir,
    )

if __name__ == "__main__":
    try:
        ingest_all_sources_to_vector_db()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户强制终止解析构建中断。")
    except Exception as e:
        logger.exception(f"解析入库期间发生未捕获级别的错误崩溃：{e}")
