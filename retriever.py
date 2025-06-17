# File ini adalah bagian dari proyek chatbot medis yang menggunakan FAISS untuk pencarian berbasis vektor.
# File ini bertanggung jawab untuk mengambil data relevan dari indeks FAISS
# berdasarkan query yang diberikan, menggunakan model embedding dari Cohere.

import os
import pandas as pd
import numpy as np
import faiss
from langchain_community.embeddings import CohereEmbeddings
from dotenv import load_dotenv
import streamlit as st 

class FaissRetriever:
    def __init__(self, csv_path: str, index_path: str):
        load_dotenv()
        
        self.df = pd.read_csv(csv_path).fillna("")

        
        # Coba ambil dari st.secrets dulu, jika gagal (berarti jalan lokal), ambil dari .env
        cohere_api_key = st.secrets.get("COHERE_API_KEY") or os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY harus diset di .env (lokal) atau di Secrets (Streamlit Cloud)")
            
        self.embedding_model = CohereEmbeddings(
            cohere_api_key=cohere_api_key,
            model="embed-multilingual-v3.0",
            user_agent="medical-chatbot-agent"
        )

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            raise FileNotFoundError(f"File index FAISS tidak ditemukan di {index_path}.")

    def get_relevant(self, query: str, k: int = 5) -> list:
        
        q_vec = self.embedding_model.embed_query(query)
        distances, indices = self.index.search(np.array([q_vec], dtype='float32'), k)
        results = []
        for i in range(k):
            idx = indices[0][i]
            if 0 <= idx < len(self.df):
                result_dict = self.df.iloc[idx].to_dict()
                result_dict['distance'] = distances[0][i]
                results.append(result_dict)
        return results