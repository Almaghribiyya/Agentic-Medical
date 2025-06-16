# File ini bertanggung jawab untuk mengambil dokumen medis yang relevan dari dataset menggunakan FAISS.
#                 full_text += page_text + "\n"

import os
import pandas as pd
import numpy as np
import faiss
from langchain_community.embeddings import CohereEmbeddings
from dotenv import load_dotenv

class FaissRetriever:
    def __init__(self, csv_path: str, index_path: str):
        """
        Inisialisasi retriever dengan memuat dataset dan index FAISS.
        """
        load_dotenv()
        
        print("Memuat dataset ke memori...")
        self.df = pd.read_csv(csv_path).fillna("")

        print("Inisialisasi model embedding Cohere...")
        cohere_api_key = os.getenv('COHERE_API_KEY')
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY harus diset di .env")
            
        self.embedding_model = CohereEmbeddings(
            cohere_api_key=cohere_api_key,
            model="embed-multilingual-v3.0",
            user_agent="medical-chatbot-agent"
        )

        if os.path.exists(index_path):
            print(f"Memuat FAISS index dari {index_path}")
            self.index = faiss.read_index(index_path)
        else:
            raise FileNotFoundError(f"File index FAISS tidak ditemukan di {index_path}. Jalankan create_index.py terlebih dahulu.")

    def get_relevant(self, query: str, k: int = 5) -> list:
        """
        Mencari dokumen yang relevan dengan sebuah query.
        """
        q_vec = self.embedding_model.embed_query(query)
        
        
        # Mengganti placeholder '_' dengan nama variabel yang jelas 'distances' dan 'indices'.
        distances, indices = self.index.search(np.array([q_vec], dtype='float32'), k)
        
        results = []
        # Menggunakan variabel 'indices' yang baru dan lebih jelas
        for i in range(k):
            idx = indices[0][i]
            # Memastikan index valid sebelum mengakses dataframe
            if 0 <= idx < len(self.df):
                result_dict = self.df.iloc[idx].to_dict()
                # Secara opsional, kita bisa menyimpan jarak/skor relevansi
                result_dict['distance'] = distances[0][i]
                results.append(result_dict)
        return results