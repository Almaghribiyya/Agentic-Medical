# File utama aplikasi Chatbot Kesehatan AI.


import os
from dotenv import load_dotenv
import streamlit as st

# Import library LangChain dan lainnya
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import GoogleAPICallError
from googlesearch import search

# Import komponen lokal
from retriever import FaissRetriever
from medical_document_processor import extract_medical_document
from callback_handler import GeminiCallbackHandler

# Import semua fungsi dari tools
from tools.statistics_tool import find_extremes_in_year, analyze_cause_trend
from tools.recommendation_tool import recommend_actions
from tools.medical_info_tool import get_medical_info
from tools.date_tool import get_current_date
from tools.translator_tool import translate_medical_terms

# Load environment variables di awal
load_dotenv()

def load_css():
    """Load custom CSS dari file style.css."""
    try:
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("File 'style.css' tidak ditemukan.")

def get_Google_Search_results(query: str) -> str:
    """
    Melakukan pencarian Google dan mengembalikan string jawaban yang sudah diformat lengkap dengan URL.
    """
    try:
        # Menentukan jumlah hasil pencarian
        num_results = 5
        print(f"Melakukan pencarian Google untuk '{query}'...")
        results = list(search(query, num_results=num_results, lang="id"))
        
        if not results:
            return f"Maaf, saya tidak dapat menemukan hasil yang relevan di Google untuk '{query}'."

        # Buat daftar URL yang diformat dengan nomor
        url_list = "\n".join([f"{i+1}. {url}" for i, url in enumerate(results)])
        
        # --- [PERBAIKAN UTAMA] ---
        # Buat string jawaban final yang akan langsung ditampilkan oleh agent.
        final_answer = (
            f"Tentu, berikut adalah {len(results)} hasil pencarian teratas untuk '{query}':\n"
            f"{url_list}\n\n"
            "**Penting**: Harap evaluasi sendiri kredibilitas dan keakuratan informasi dari situs-situs tersebut."
        )
        return final_answer
        
    except Exception as e:
        return f"Terjadi kesalahan saat melakukan pencarian Google: {str(e)}"

# --- Logika Agent ---
def run_agent(user_input: str, retriever: FaissRetriever, memory, pdf_content: str = None):
    """
    Menginisialisasi dan menjalankan agent.
    Agent ini sekarang sadar akan konteks PDF dan diinstruksikan untuk memberikan jawaban yang lebih kontekstual.
    """
    print("--- Menjalankan Unified Agent ---")
    
    # Buat prompt sistem yang dinamis berdasarkan ada atau tidaknya konteks PDF
    system_prompt = ""
    if pdf_content:
        # --- [PROMPT DISEMPURNAKAN DI SINI] ---
        system_prompt = f"""
        PERHATIAN: Pengguna telah mengunggah sebuah dokumen medis. Anda adalah asisten medis yang sangat teliti dan komunikatif.
        
        TUGAS UTAMA ANDA:
        1.  Jawab pertanyaan pengguna HANYA berdasarkan 'KONTEKS DOKUMEN' di bawah ini.
        2.  Saat menjawab pertanyaan tentang pasien, SELALU sebutkan nama pasien (jika ada di dalam dokumen) untuk memberikan konteks yang jelas. Contoh: "Menurut dokumen, diagnosis untuk pasien Budi Santoso adalah..."
        3.  Jika jawaban tidak ada di dokumen, katakan dengan jujur bahwa informasi tersebut tidak ditemukan di dalam dokumen.
        4.  Gunakan 'Tools' hanya jika pertanyaan jelas-jelas tidak berkaitan dengan isi dokumen (misalnya, menanyakan statistik umum, tanggal, atau pencarian internet).

        --- KONTEKS DOKUMEN ---
        {pdf_content}
        --- AKHIR KONTEKS DOKUMEN ---
        """

    # Gabungkan prompt sistem dengan pertanyaan pengguna untuk input agent
    final_input = f"{system_prompt}\n\nPertanyaan: {user_input}"

    try:
        # Definisi tools (tidak ada perubahan)
        tools = [
            Tool(name='cari_info_dari_database_kesehatan', func=lambda q: get_medical_info(q, retriever), description="Gunakan untuk menjawab pertanyaan spesifik tentang penyakit atau kondisi dari database statistik internal. Input harus nama penyakit."),
            Tool(name='analisis_tren_statistik_penyakit', func=lambda q: analyze_cause_trend(q, retriever), description="Gunakan untuk menganalisis statistik tren untuk SATU JENIS penyakit dari waktu ke waktu dari database. Input harus nama penyakitnya."),
            Tool(name='cari_penyebab_kematian_ekstrem_per_tahun', func=find_extremes_in_year, description="Gunakan untuk mencari penyebab kematian TERTINGGI atau TERENDAH pada SATU TAHUN spesifik dari database. Pertanyaan harus mengandung 'tertinggi' atau 'terendah' dan tahun."),
            Tool(name='beri_rekomendasi_kesehatan_umum', func=lambda q: recommend_actions(q, retriever), description="Gunakan untuk memberikan rekomendasi kesehatan umum berdasarkan topik dari database."),
            Tool(name='pencarian_internet_google', func=get_Google_Search_results, description="Gunakan HANYA untuk mencari berita kesehatan SANGAT BARU atau informasi medis umum yang TIDAK ADA di database maupun dokumen."),
            Tool(name='terjemah_istilah_medis', func=translate_medical_terms, description="Gunakan untuk menerjemahkan istilah medis. Format: 'teks to bahasa_tujuan'."),
            Tool(name='dapatkan_tanggal_sekarang', func=get_current_date, description="Gunakan untuk mengetahui tanggal dan waktu saat ini.")
        ]
        
        gemini_handler = GeminiCallbackHandler()
        llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.2, streaming=True, callbacks=[gemini_handler], convert_system_message_to_human=True)
        agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True, handle_parsing_errors="Maaf, terjadi sedikit kendala. Coba tanyakan dengan cara lain.")
        
        response = agent.run(final_input)
        return response

    except (GoogleAPICallError) as e:
        st.error(f"Error pada layanan Gemini: {e}. Pastikan API Key Anda valid.")
        raise e
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menjalankan agent: {e}")
        raise e

# --- Fungsi Utama Aplikasi ---
def main():
    st.set_page_config(page_title="Asisten Kesehatan AI", page_icon="🩺", layout="centered")
    load_css()
    st.markdown("<header><h1>🩺 Asisten Kesehatan Indonesia</h1><p>Didukung oleh AI, RAG, dan Google Search</p></header>", unsafe_allow_html=True)
    
    # ---Saran pertanyaan yang lebih jelas dan mengarahkan ---
    initial_greeting = """Halo! Saya adalah Asisten Kesehatan AI Anda.

**Apa yang bisa saya bantu?**

Anda bisa bertanya tentang data kesehatan umum dari database kami, contohnya:
* `Apa penyebab kematian tertinggi di Indonesia pada tahun 2015?`
* `Bagaimana tren statistik penyakit DBD?`
* `Berikan rekomendasi untuk penyakit menular.`

**Anda juga bisa mengunggah dokumen medis (PDF) di sidebar.** Setelah diunggah, Anda bisa langsung bertanya tentang isinya, misalnya:
* `Apa diagnosis utama pasien dalam dokumen ini?`
* `Sebutkan semua obat yang diresepkan dalam file tersebut.`
"""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": initial_greeting}]
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if "processed_file_name" not in st.session_state:
        st.session_state.processed_file_name = None
    if "pdf_content" not in st.session_state:
        st.session_state.pdf_content = None

    @st.cache_resource
    def init_retriever():
        return FaissRetriever(csv_path='data/Penyebab Kematian di Indonesia yang Dilaporkan - Clean.csv', index_path='data/faiss_index.idx')
    retriever = init_retriever()

    with st.sidebar:
        st.header("Analisis Dokumen")
        uploaded_file = st.file_uploader("Upload PDF Rekam Medis", type=['pdf'], key="pdf_uploader")
        if uploaded_file and uploaded_file.name != st.session_state.get('processed_file_name'):
            with st.spinner("Memproses dokumen..."):
                doc_info = extract_medical_document(uploaded_file)
                if 'error' in doc_info:
                    st.error(doc_info['error'])
                    st.session_state.processed_file_name = None
                else:
                    st.session_state.pdf_content = doc_info.get('full_text')
                    st.session_state.processed_file_name = uploaded_file.name
                    st.success("Dokumen berhasil diproses!")
                    st.session_state.messages.append({"role": "system", "content": f"Dokumen '{uploaded_file.name}' telah diunggah. Anda sekarang bisa bertanya mengenai isinya."})
        
        st.divider()
        with st.expander("📜 Riwayat Percakapan"):
            if not st.session_state.messages: st.write("Belum ada percakapan.")
            else:
                for msg in st.session_state.messages:
                    if msg["role"] != "system": st.markdown(f'**{msg["role"].replace("user", "Anda").replace("assistant", "AI")}:** *{msg["content"][:40]}...*')
        
        if st.button("Hapus Riwayat & Dokumen", type="secondary", key="delete_history"):
            st.session_state.messages = [{"role": "assistant", "content": "Riwayat chat dan dokumen telah dihapus. Silakan mulai percakapan baru."}]
            st.session_state.memory.clear()
            st.session_state.pdf_content = None
            st.session_state.processed_file_name = None
            st.rerun()

    for message in st.session_state.messages:
        if message.get("role") != "system":
            avatar = "🧑‍💻" if message["role"] == "user" else "🩺"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    if user_input := st.chat_input("Tanyakan sesuatu..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("Asisten sedang berpikir..."):
                try:
                    pdf_context = st.session_state.get("pdf_content")
                    response_text = run_agent(user_input, retriever, st.session_state.memory, pdf_content=pdf_context)
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    st.error(f"Maaf, terjadi kesalahan fatal: {e}")
                    st.session_state.messages.pop()

if __name__ == "__main__":
    main()