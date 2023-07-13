# -*- coding: utf-8 -*-
import datetime
from typing import List
from tqdm import tqdm
import os
import logging

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from functools import lru_cache
from pypinyin import lazy_pinyin
from pydantic import BaseModel

from modules.vectorstores import MyFAISS
from modules.textsplitter import ChineseTextSplitter
from modules.textsplitter.textsplitter import split_text
from modules.utils.gpu import clear_torch_cache
from modules.agent import bing_search
from modules.models import BaseLLM


# patch HuggingFaceEmbeddings to make it hashable
def _embeddings_hash(self):
    return hash(self.model_name)


HuggingFaceEmbeddings.__hash__ = _embeddings_hash

# 缓存知识库数量
CACHED_VS_NUM = 1
# Embedding model name
EMBEDDING_MODEL = "text2vec"
# 知识库检索时返回的匹配内容条数
VECTOR_SEARCH_TOP_K = 2
# 匹配后单段上下文长度
CHUNK_SIZE = 250
# 知识检索内容相关度 Score, 数值范围约为0-1100，如果为0，则不生效，经测试设置为小于500时，匹配结果更精准
VECTOR_SEARCH_SCORE_THRESHOLD = 0
# Embedding running device
EMBEDDING_DEVICE = "cuda"
# 在以下字典中修改属性值，以指定本地embedding模型存储位置
# 如将 "text2vec": "GanymedeNil/text2vec-large-chinese" 修改为 "text2vec": "User/Downloads/text2vec-large-chinese"
# 此处请写绝对路径
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "m3e-small": "moka-ai/m3e-small",
    "m3e-base": "moka-ai/m3e-base",
}
# 文本分句长度
SENTENCE_SIZE = 100
# 知识库默认存储路径
KB_ROOT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "knowledge_base"
)
# LLM streaming reponse
STREAMING = True


# will keep CACHED_VS_NUM of vector store caches
@lru_cache(CACHED_VS_NUM)
def load_vector_store(vs_path, embeddings):
    return MyFAISS.load_local(vs_path, embeddings)


def tree(filepath, ignore_dir_names=None, ignore_file_names=None):
    """返回两个列表，第一个列表为 filepath 下全部文件的完整路径, 第二个为对应的文件名"""
    if ignore_dir_names is None:
        ignore_dir_names = []
    if ignore_file_names is None:
        ignore_file_names = []
    ret_list = []
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            logging.info("路径不存在")
            return None, None
        elif (
            os.path.isfile(filepath)
            and os.path.basename(filepath) not in ignore_file_names
        ):
            return [filepath], [os.path.basename(filepath)]
        elif (
            os.path.isdir(filepath)
            and os.path.basename(filepath) not in ignore_dir_names
        ):
            for file in os.listdir(filepath):
                fullfilepath = os.path.join(filepath, file)
                if (
                    os.path.isfile(fullfilepath)
                    and os.path.basename(fullfilepath) not in ignore_file_names
                ):
                    ret_list.append(fullfilepath)
                if (
                    os.path.isdir(fullfilepath)
                    and os.path.basename(fullfilepath) not in ignore_dir_names
                ):
                    ret_list.extend(
                        tree(fullfilepath, ignore_dir_names, ignore_file_names)[0]
                    )
    return ret_list, [os.path.basename(p) for p in ret_list]


def remove_garbled_characters(text):
    try:
        # 尝试将字符串解码为UTF-8
        text = text.decode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        # 如果解码失败或者输入不是字符串类型，就直接返回原始文本
        return text

    # 去除非中文字符
    cleaned_text = "".join([char for char in text if "\u4e00" <= char <= "\u9fa5"])

    return cleaned_text


def generate_prompt(
    related_docs: List[str],
    query: str,
    prompt_template: str,
) -> str:
    context = "\n".join(
        [remove_garbled_characters(doc.page_content) for doc in related_docs]
    )
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt


def search_result2docs(search_results):
    docs = []
    for result in search_results:
        doc = Document(
            page_content=result["snippet"] if "snippet" in result.keys() else "",
            metadata={
                "source": result["link"] if "link" in result.keys() else "",
                "filename": result["title"] if "title" in result.keys() else "",
            },
        )
        docs.append(doc)
    return docs


class LocalDocQA(BaseModel):
    llm: BaseLLM = None
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE
    chunk_conent: bool = True
    score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD

    def init_cfg(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        embedding_device=EMBEDDING_DEVICE,
        top_k=VECTOR_SEARCH_TOP_K,
    ):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_dict[embedding_model],
            model_kwargs={"device": embedding_device},
        )
        self.top_k = top_k

    def init_knowledge_vector_store(
        self,
        filepath: str or List[str],
        vs_path: str or os.PathLike,
        sentence_size,
        using_zh_title_enhance,
    ):
        loaded_files = []
        failed_files = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                logging.info("路径不存在")
                return None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                try:
                    docs = split_text(filepath, sentence_size, using_zh_title_enhance)
                    logging.info(f"{file} 已成功加载")
                    loaded_files.append(filepath)
                except Exception as e:
                    logging.error(e)
                    logging.info(f"{file} 未能成功加载")
                    return None
            elif os.path.isdir(filepath):
                docs = []
                for fullfilepath, file in tqdm(
                    zip(*tree(filepath, ignore_dir_names=["tmp_files"])), desc="加载文件"
                ):
                    try:
                        docs += split_text(
                            fullfilepath, sentence_size, using_zh_title_enhance
                        )
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        logging.error(e)
                        failed_files.append(file)

                if len(failed_files) > 0:
                    logging.info("以下文件未能成功加载：")
                    for file in failed_files:
                        logging.info(f"{file}\n")

        else:
            docs = []
            for file in filepath:
                try:
                    docs += split_text(file, sentence_size, using_zh_title_enhance)
                    logging.info(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    logging.error(e)
                    logging.info(f"{file} 未能成功加载")
        if len(docs) > 0:
            logging.info("文件加载完毕，正在生成向量库")
            if (
                vs_path
                and os.path.isdir(vs_path)
                and "index.faiss" in os.listdir(vs_path)
            ):
                vector_store = load_vector_store(vs_path, self.embeddings)
                vector_store.add_documents(docs)
                clear_torch_cache()
            else:
                if not vs_path:
                    vs_path = os.path.join(
                        KB_ROOT_PATH,
                        f"""{"".join(lazy_pinyin(os.path.splitext(file)[0]))}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""",
                        "vector_store",
                    )
                vector_store = MyFAISS.from_documents(
                    docs, self.embeddings
                )  # docs 为Document列表
                clear_torch_cache()

            vector_store.save_local(vs_path)
            return vs_path, loaded_files
        else:
            logging.info("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
            return None, loaded_files

    def one_knowledge_add(
        self, vs_path, one_title, one_conent, one_content_segmentation, sentence_size
    ):
        try:
            if not vs_path or not one_title or not one_conent:
                logging.info("知识库添加错误，请确认知识库名字、标题、内容是否正确！")
                return None, [one_title]
            docs = [
                Document(page_content=one_conent + "\n", metadata={"source": one_title})
            ]
            if not one_content_segmentation:
                text_splitter = ChineseTextSplitter(
                    pdf=False, sentence_size=sentence_size
                )
                docs = text_splitter.split_documents(docs)
            if os.path.isdir(vs_path) and os.path.isfile(vs_path + "/index.faiss"):
                vector_store = load_vector_store(vs_path, self.embeddings)
                vector_store.add_documents(docs)
            else:
                vector_store = MyFAISS.from_documents(
                    docs, self.embeddings
                )  ##docs 为Document列表
            clear_torch_cache()
            vector_store.save_local(vs_path)
            return vs_path, [one_title]
        except Exception as e:
            logging.error(e)
            return None, [one_title]

    def get_knowledge_based_answer(
        self,
        query,
        vs_path,
        prompt_template,
        chat_history=[],
        streaming: bool = STREAMING,
        llm_model: BaseLLM = None,
    ):
        self.llm = llm_model
        vector_store = load_vector_store(vs_path, self.embeddings)
        vector_store.chunk_size = self.chunk_size
        vector_store.chunk_conent = self.chunk_conent
        vector_store.score_threshold = self.score_threshold
        related_docs_with_score = vector_store.similarity_search_with_score(
            query, k=self.top_k
        )
        clear_torch_cache()
        if len(related_docs_with_score) > 0:
            prompt = generate_prompt(related_docs_with_score, query, prompt_template)
        else:
            prompt = query

        logging.info(f"generate_prompt: \n{prompt}")

        for answer_result in self.llm.generateAnswer(
            prompt=prompt, history=chat_history, streaming=streaming
        ):
            yield answer_result

    def get_search_result_based_answer(
        self,
        query,
        prompt_template,
        chat_history=[],
        streaming: bool = STREAMING,
        llm_model: BaseLLM = None,
    ):
        self.llm = llm_model
        result_docs = bing_search(query)
        prompt = generate_prompt(result_docs, query, prompt_template)

        for answer_result in self.llm.generateAnswer(
            prompt=prompt, history=chat_history, streaming=streaming
        ):
            yield answer_result

    def delete_file_from_vector_store(self, filepath: str or List[str], vs_path):
        vector_store = load_vector_store(vs_path, self.embeddings)
        status = vector_store.delete_doc(filepath)
        return status

    def update_file_from_vector_store(
        self,
        filepath: str or List[str],
        vs_path,
        docs: List[Document],
    ):
        vector_store = load_vector_store(vs_path, self.embeddings)
        status = vector_store.update_doc(filepath, docs)
        return status

    def list_file_from_vector_store(self, vs_path, fullpath=False):
        vector_store = load_vector_store(vs_path, self.embeddings)
        docs = vector_store.list_docs()
        if fullpath:
            return docs
        else:
            return [os.path.split(doc)[-1] for doc in docs]
