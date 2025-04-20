#调api
import faiss
import numpy as np
import os
import pickle
import logging
from openai import OpenAI  # 使用 OpenAI 客户端
from loader import load_documents  # 使用改进后的 loader 模块
from config import config

# ========================
# 模块初始化配置
# 配置日志输出
# ========================
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 初始化 OpenAI 客户端
client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)

def get_embedding_from_api(text):
    """通过 API 获取嵌入向量，确保文本长度在有效范围内"""
    max_input_length = 2048  # OpenAI 最大输入长度
    # 如果文本过长，进行拆分
    if len(text) > max_input_length:
        chunks = [text[i:i + max_input_length] for i in range(0, len(text), max_input_length)]
        embeddings = []
        for chunk in chunks:
            response = client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=chunk
            )
            # 获取嵌入向量
            embeddings.append(np.array(response.data[0].embedding))  # 使用 response.data[0].embedding 获取向量
        return np.mean(embeddings, axis=0)  # 返回所有分块嵌入向量的平均值
    else:
        response = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=text
        )
        # 获取嵌入向量
        return np.array(response.data[0].embedding)  # 使用 response.data[0].embedding 获取向量

def rebuild_index():
    """重新加载所有文档，并重建 FAISS 索引"""
    print("🔄 开始重建 FAISS 索引...")

    # 使用配置中的知识库目录
    knowledge_dir = config.REFERENCE_FOLDER  # 这里改为 `REFERENCE_FOLDER`

    # 通过 loader 并发加载所有文档
    docs = list(load_documents(knowledge_dir))
    if not docs:
        print("⚠️ 没有找到文档，索引未更新。")
        return "⚠️ 没有找到文档，索引未更新。"

    print(f"📂 加载了 {len(docs)} 个文档")  # 打印文档数量

    # 提取每篇文档的内容和文件名
    texts = [doc["content"] for doc in docs]
    filenames = [doc["filename"] for doc in docs]

    # 将文档内容转化为嵌入向量
    embeddings = np.array([get_embedding_from_api(text) for text in texts])  # 获取嵌入向量

    # 使用 FAISS 建立索引
    dim = embeddings.shape[1]  # 向量维度
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # 保存 FAISS 索引和文档文件名列表
    faiss.write_index(index, str(config.FAISS_CACHE / "docs.index"))

    with open(str(config.FAISS_CACHE / "filenames.pkl"), "wb") as f:
        pickle.dump(filenames, f)

    print("✅ FAISS 索引已成功重建！")
    return "✅ FAISS 索引已成功重建！"


# 如果 `main.py` 直接运行，则自动创建索引
if __name__ == "__main__":
    rebuild_index()


