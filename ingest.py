"""
本程序主要用于读取指定目录下所有研报文本,建立索引.
"""
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle

from dotenv import load_dotenv
load_dotenv()


# 下面开始处理该目录下所有研报文本,用于构建研报知识库
ps = list(Path("doc/").glob("**/*.txt"))

data = []
sources = []
for p in ps:
    with open(p, encoding="utf-8") as f:
        data.append(f.read())
    sources.append(p)

# 分割文档,防止单个文档过长.
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))

# 为文档计算向量,构建索引库
store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
