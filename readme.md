# 🔍 基于LLMs API 的 RAG 知识问答系统 (DeepSeek乞丐版)

# 简介

根据https://blog.csdn.net/weixin_43925843/article/details/146398834改用调用百炼api而非本地llm模型方式，结合RAG对上传文档进行FAISS 索引，通过OpenAI的Python SDK来调用百炼平台上的模型如qwq、deepseek等。

```
前端：Gradio交互界面 / FastAPI服务。

核心模块：文档加载 → 向量索引 → 检索增强生成。

底层支持：FAISS向量库、SentenceTransformer嵌入模型、LLM api（如DeepSeek/ChatGLM）。
```

项目结构

```
Rag-System/
├─ knowledge_base/
│   ├─ some_text.txt          # 你本地知识库中的各种文件(txt/pdf/docx等)
│   ├─ identity.md            # 自我认知文件
│   └─ ...
├─ cache/
│   └─ faiss_index/           # FAISS 索引缓存目录（会自动生成）
├─ icon/
│   ├─ bot.png                # agent 头像
│   └─ user.png               # 用户 头像
├─ config.py                  # 配置文件
├─ loader.py                  # 索引构建
├─ main.py                    # 多线程加载文档
├─ rag.py                     # 文档 检索、回答生成
├─ app.py                     # Gradio交互界面
├─ api.py                     # FastAPI服务端，提供REST接口
└── ... 其他文件 ...
```

```
向量检索：FAISS（Facebook开源的相似性搜索库）。

嵌入模型：text-embedding-v2

生成模型：deepseek-r1-distill-qwen-1.5b

开发框架：Gradio（快速构建UI）、FastAPI（高性能API服务）。
```

# 准备

✅安装依赖。pip install -r requirements.txt

✅本地环境变量配置DASHSCOPE_API_KEY，或在config.py的API_KEY行替换sk-default

✅运行main.py 重建FAISS 索引

# 运行

✅启动 命令行交互界面   python rag.py

✅启动Gradio界面   python app.py   访问 http://localhost:7860 使用交互界面。

![](fig\\demo.png)











