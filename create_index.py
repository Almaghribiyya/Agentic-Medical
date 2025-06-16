# File ini digunakan untuk membuat indeks FAISS dari dataset lokal
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
        print(f"Index sudah ada. Hapus file {index_path} jika ingin membuat ulang.")
        return
    
    df = pd.read_csv(csv_path).fillna("")
    df['combined'] = df.astype(str).agg(' '.join, axis=1)
    texts = df['combined'].tolist()
    
    # Hanya gunakan os.getenv karena file ini dijalankan lokal
    cohere_api_key = os.getenv('COHERE_API_KEY')
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY harus diset di file .env")
    
    embedding_model = CohereEmbeddings(
        cohere_api_key=cohere_api_key,
        model="embed-multilingual-v3.0",
        user_agent="medical-chatbot-agent"
    )
    
    print("Membuat embeddings...")
    vectors = embedding_model.embed_documents(texts)
    
    print("Membuat FAISS index...")
    vector_dimension = len(vectors[0])
    index = faiss.IndexFlatL2(vector_dimension)
    index.add(np.array(vectors, dtype='float32'))
    
    faiss.write_index(index, index_path)
    print(f"Index berhasil dibuat di {index_path}")

if __name__ == '__main__':
    create_faiss_index()