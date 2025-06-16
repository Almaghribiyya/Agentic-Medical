# fungsi untuk membuat index FAISS dari dataset CSV yang berisi informasi medis.
import os
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_community.embeddings import CohereEmbeddings

def create_faiss_index():
    load_dotenv()
    
    csv_path = 'data/Penyebab Kematian di Indonesia yang Dilaporkan - Clean.csv'
    index_path = 'data/faiss_index.idx'
    
    if os.path.exists(index_path):
        print(f"Index sudah ada di {index_path}. Hapus file ini jika ingin membuat ulang.")
        return
    
    print("Memuat data...")
    df = pd.read_csv(csv_path).fillna("")
    df['combined'] = df.astype(str).agg(' '.join, axis=1)
    texts = df['combined'].tolist()
    
    print("Inisialisasi model embedding...")
    cohere_api_key = os.getenv('COHERE_API_KEY')
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY harus diset di file .env")
    
    embedding_model = CohereEmbeddings(
        cohere_api_key=cohere_api_key,
        model="embed-multilingual-v3.0",
        user_agent="medical-chatbot-agent"  # <-- PENAMBAHAN BARIS INI UNTUK FIX ERROR
    )
    
    print("Membuat embeddings... Proses ini mungkin memakan waktu beberapa menit.")
    vectors = embedding_model.embed_documents(texts)
    
    print("Membuat FAISS index...")
    vector_dimension = len(vectors[0])
    index = faiss.IndexFlatL2(vector_dimension)
    index.add(np.array(vectors, dtype='float32'))
    
    faiss.write_index(index, index_path)
    print(f"Index berhasil dibuat dan disimpan di {index_path}")

if __name__ == '__main__':
    create_faiss_index()