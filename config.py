#改用api调用方式
from pathlib import Path
import os
from openai import OpenAI

BASE_DIR = Path(__file__).parent

'''
需要配置环境变量，参考
https://help.aliyun.com/zh/model-studio/developer-reference/configure-api-key-through-environment-variables?spm=a2c4g.11186623.help-menu-2400256.d_2_0_1.394872a3CgQzh8#e4cd73d544i3r

'''
class config:
    # API 服务配置
    API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-default")  # 默认API Key
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 初始化API客户端
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 模型API名称配置
    EMBEDDING_MODEL = "text-embedding-v2"  # 替换为实际支持的embedding模型名称
    LLM_MODEL_NAME = "deepseek-r1-distill-qwen-1.5b"  # 大模型名称https://help.aliyun.com/zh/model-studio/new-free-quota?spm=a2c4g.11186623.0.0.5b444823SiWTr2#view-quota
    #LLM_MODEL_NAME = "deepseek-r1"
    # 默认模型相关参数
    DEFAULT_MAX_LENGTH = 4096
    CHUNK_SIZE = 1000
    OVERLAP = 200

    # FAISS 索引缓存目录
    FAISS_CACHE = BASE_DIR / "cache" / "faiss_index"

    # 参考文档文件夹配置
    REFERENCE_FOLDER = BASE_DIR / "knowledge_base"
    REFERENCE_FOLDER.mkdir(parents=True, exist_ok=True)
    IDENTITY_FILE = REFERENCE_FOLDER / "identity.md"

    # 对话参数
    MAX_HISTORY = 5
    STREAM_SEGMENT_SIZE = 5
    STREAM_DELAY = 0.1
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Maddie")

    def __init__(self):
        self.FAISS_CACHE.mkdir(parents=True, exist_ok=True)
        self.REFERENCE_FOLDER.mkdir(parents=True, exist_ok=True)

    def get_embedding(self, text):
        # 使用API来获取文本的嵌入向量
        response = self.client.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=text,
            dimensions=1024,
            encoding_format="float"
        )
        return response.data[0].embedding

    def get_llm_response(self, prompt):
        # 使用 API 获取 LLM 模型的响应
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},  # 系统消息，指示模型的角色
            {"role": "user", "content": prompt}  # 用户输入的消息
        ]
        response = self.client.chat.completions.create(
            model=self.LLM_MODEL_NAME,  # 使用配置中的 LLM 模型名称
            messages=messages  # 传递消息
        )
        # 修改：使用正确的属性来访问生成文本
        return response.choices[0].message.content

# 实例化配置对象
config = config()