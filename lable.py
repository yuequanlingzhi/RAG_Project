import PyPDF2
import os
import random
import json
from sentence_transformers import SentenceTransformer, losses
import faiss
import numpy as np
import torch
import textwrap


#pdf读取文本
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_pdfs(pdf_dir):
    pdf_texts = {}
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            pdf_texts[filename] = extract_text_from_pdf(pdf_path)
    return pdf_texts

#文本截断为小chunk
def split_into_chunks(text, max_chunk_size=512):
    #以句号拆分
    for char in ['\n', '\t', '.']:
        text = text.replace(char, '')
    sentences = text.split('。')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence 
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

pdf_dir = r'Final_Project_Documents\documents' 
pdf_texts = extract_text_from_pdfs(pdf_dir)


len_min = 10 #随机片段的下限
len_max = 64 #随机片段的上限r
top_n = 10

labeled_data = []

# 加载模型
embedder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1') #或fine_tuned_simcse_model

# 创建FAISS索引
def create_embeddings_and_index(pdf_texts, max_chunk_size=512):
    all_chunks = []
    for filename, text in pdf_texts.items():
        chunks = split_into_chunks(text, max_chunk_size)
        all_chunks.extend(chunks)  # 将所有chunk放入一个列表
    
    embeddings = embedder.encode(all_chunks, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return all_chunks, index

all_chunks, index = create_embeddings_and_index(pdf_texts, max_chunk_size=128)

# 搜索函数，检索相似的文段
def search(query, all_chunks, index, k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, k)
    return [(all_chunks[i], D[0][idx]) for idx, i in enumerate(I[0])]

def save_labeled_data(data, filename='labeled_data.json'):
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    print(f"警告：{filename} 中的数据格式不正确，将覆盖为新的列表。")
                    existing_data = []
        except json.JSONDecodeError:
            print(f"警告：无法解码 {filename}，将覆盖为新的列表。")
            existing_data = []
    else:
        existing_data = []
    
    # 将标签数据的 similarity_score 转换为 float 类型
    for item in data:
        if isinstance(item.get("similarity_score"), (torch.Tensor, float, np.float32)):
            item["similarity_score"] = float(item["similarity_score"])
    existing_data.extend(data)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
    print(f"标注数据已追加到 {filename}")

#限制一行最长为50字符
def wrap_text(text, width=50):
    return textwrap.fill(text, width=width)

while True:
    chunk = random.choice(all_chunks)
    random_len = random.randint(0,min(len_max,len(chunk)))
    satrt_pos = random.randint(0,len(chunk)-random_len)
    query = chunk[satrt_pos:satrt_pos+random_len]
    retrieved_chunks = search(query, all_chunks, index, k=top_n)
    print(f"\n\n随机文段：{query}\n")
    for chunk, score in retrieved_chunks:
        print(f"Chunk:\n {wrap_text(chunk)}\nScore: {score}\n")
    
    user_input = input(f"请输入标签（例如 '1,0,1'），或输入 'q' 结束: ").strip()
    if user_input.lower() == 'q':
        break
    try:
        labels = [int(x) for x in user_input.split(',')]
        if len(labels) != top_n or any(label not in [0,1] for label in labels):
            raise ValueError
    except ValueError:
        print("输入格式错误，请输入如 '1,0,1' 的标签，且每个标签为0或1。")
        continue

    for i, (chunk, score) in enumerate(retrieved_chunks):
            label = labels[i]
            example = {
                "substring": query,
                "matched_chunk": chunk,
                "similarity_score": score,
                "label": label
            }
            labeled_data.append(example)
        
    print("标注已记录。")
# 保存标注数据
if labeled_data:
    save_labeled_data(labeled_data)
else:
    print("没有标注数据被记录。")