# main.py

import os
from dotenv import load_dotenv
import streamlit as st
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from googlesearch import search
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_core.messages import SystemMessage

# Import komponen lokal
from retriever import FaissRetriever
from medical_document_processor import extract_medical_document
from callback_handler import GeminiCallbackHandler
from tools.statistics_tool import find_extremes_in_year, analyze_cause_trend
from tools.recommendation_tool import recommend_actions
from tools.medical_info_tool import get_medical_info
from tools.date_tool import get_current_date
from tools.translator_tool import translate_medical_terms

load_dotenv()

# --- FUNGSI-FUNGSI UTAMA & TOOLS ---
@st.cache_resource
def init_retriever():
    """Menginisialisasi retriever dan menyimpannya di cache Streamlit untuk efisiensi."""
    return FaissRetriever(
        csv_path='data/Penyebab Kematian di Indonesia yang Dilaporkan - Clean.csv',
        index_path='data/faiss_index.idx'
    )

def search_and_summarize_web(query: str) -> str:
    """
    Melakukan pencarian Google, membuka link teratas, membaca isinya,
    dan merangkumnya untuk menjawab pertanyaan pengguna.
    """
    try:
        print(f"Melakukan pencarian Google untuk '{query}'...")
        search_results = list(search(query, num_results=1, lang="id"))
        
        if not search_results:
            return f"Maaf, tidak ada hasil relevan di Google untuk '{query}'.<<SOURCE:Google Search>>"

        top_url = search_results[0]
        print(f"Membaca konten dari: {top_url}")

        loader = WebBaseLoader(web_paths=(top_url,), requests_kwargs={"verify": False})
        docs = loader.load()

        llm_summarizer = ChatGoogleGenerativeAI(
            model='gemini-2.0-flash',
            google_api_key=st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )
        chain = load_summarize_chain(llm_summarizer, chain_type="stuff")
        summary = chain.run(docs)
        
        response = (f"Berdasarkan informasi dari {top_url}, berikut adalah rangkumannya:\n\n"
                    f"{summary}\n\n"
                    "**Penting**: Informasi ini adalah rangkuman dari sumber eksternal dan bukan pengganti saran medis profesional.")
        
        return f"{response}<<SOURCE:Google Search - {top_url}>>"
    except Exception as e:
        return f"Terjadi kesalahan saat mencari dan merangkum dari internet: {str(e)}<<SOURCE:Error>>"


def run_agent(user_input: str, retriever: FaissRetriever, memory, pdf_content: str = None):
    """Menginisialisasi dan menjalankan agent dengan instruksi yang disempurnakan."""
    system_prompt = "Anda adalah AMIA (Asisten Medis AI), asisten AI yang cerdas, teliti, dan komunikatif. Jawab pertanyaan pengguna menggunakan alat yang tersedia atau informasi yang diberikan."
    
    # Logika mode dokumen eksklusif
    if pdf_content:
        system_prompt += f"\n\nPERHATIAN: Anda sekarang dalam 'Mode Dokumen'. Fokus utama Anda adalah menjawab pertanyaan HANYA berdasarkan 'KONTEKS DOKUMEN' di bawah ini. Jangan gunakan tools lain kecuali diminta secara eksplisit oleh pengguna. Selalu sebutkan nama pasien jika relevan.\n--- KONTEKS DOKUMEN ---\n{pdf_content}\n--- AKHIR KONTEKS ---"
    
    system_prompt += "\nPENTING: Jika observasi dari tool mengandung penanda '<<SOURCE:xyz>>', Anda WAJIB menyertakan penanda tersebut persis apa adanya di akhir jawaban final Anda."

    final_input = f"{system_prompt}\n\nPertanyaan: {user_input}"
    
    tools = [
        Tool(name='pencarian_dan_rangkuman_internet', func=search_and_summarize_web, description="Gunakan untuk pertanyaan pengetahuan umum kesehatan, TIPS (seperti 'tips kesehatan kulit'), cara mengobati penyakit, atau berita kesehatan terbaru yang TIDAK ADA di database statistik."),
        Tool(name='cari_info_dari_database_kesehatan', func=lambda q: get_medical_info(q, retriever), description="Gunakan untuk mencari informasi spesifik tentang PENYEBAB KEMATIAN dari database statistik."),
        Tool(name='analisis_tren_statistik_penyakit', func=lambda q: analyze_cause_trend(q, retriever), description="Gunakan untuk menganalisis tren statistik dari satu jenis PENYEBAB KEMATIAN."),
        Tool(name='cari_penyebab_kematian_tertinggi_atau_terendah_per_tahun', func=find_extremes_in_year, description="Gunakan untuk mencari penyebab kematian TERTINGGI atau TERENDAH pada SATU TAHUN spesifik."),
        Tool(name='beri_rekomendasi_terkait_penyebab_kematian', func=lambda q: recommend_actions(q, retriever), description="Gunakan untuk memberikan rekomendasi kesehatan terkait JENIS PENYEBAB KEMATIAN atau BENCANA dari database."),
        Tool(name='terjemah_istilah_medis', func=translate_medical_terms, description="Gunakan untuk menerjemahkan istilah medis. Format: 'teks to bahasa_tujuan'."),
    ]
    
    try:
        google_api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not google_api_key: raise ValueError("GOOGLE_API_KEY tidak ditemukan.")
        llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', google_api_key=google_api_key, temperature=0.2)
        agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True, handle_parsing_errors="Maaf, terjadi sedikit kendala.")
        response = agent.run(final_input)
        return response
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menjalankan agent: {e}")
        return None

# --- APLIKASI UTAMA STREAMLIT ---
def main():
    st.set_page_config(page_title="AMIA - Asisten Medis AI", page_icon="🩺", layout="centered")
    try:
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError: st.warning("File 'style.css' tidak ditemukan.")

    st.markdown("<header><h1>🩺 AMIA</h1><h2>Asisten Medis AI</h2></header>", unsafe_allow_html=True)
    
    # Inisialisasi state sesi jika belum ada
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if "pdf_content" not in st.session_state:
        st.session_state.pdf_content = None
    if "processed_file_name" not in st.session_state:
        st.session_state.processed_file_name = None

    retriever = init_retriever()

    # --- Sidebar ---
    with st.sidebar:
        st.header("Analisis Dokumen")
        uploaded_file = st.file_uploader("Upload PDF Rekam Medis", type=['pdf'], key="pdf_uploader")
        
        # Logika untuk memproses file BARU
        if uploaded_file and uploaded_file.name != st.session_state.processed_file_name:
            with st.spinner("Memproses..."):
                doc_info = extract_medical_document(uploaded_file)
                st.session_state.pdf_content = doc_info.get('full_text')
                st.session_state.processed_file_name = uploaded_file.name
                st.toast(f"Dokumen '{uploaded_file.name}' berhasil dianalisis.", icon="✅")
                # Hapus riwayat lama untuk memulai sesi chat dokumen yang baru
                st.session_state.messages = []
                st.session_state.memory.clear()
                st.rerun()
        
        # Logika untuk mendeteksi file dihapus oleh pengguna via tombol 'x'
        if uploaded_file is None and st.session_state.processed_file_name is not None:
            st.toast("Dokumen dihapus. Mode kembali ke percakapan umum.", icon="📄")
            st.session_state.pdf_content = None
            st.session_state.processed_file_name = None
            st.rerun()

        st.divider()

        # Riwayat hanya menampilkan pertanyaan pengguna
        with st.expander("📜 Riwayat Pertanyaan Anda"):
            user_questions = [msg['content'] for msg in st.session_state.messages if msg.get("role") == "user"]
            if not user_questions:
                st.write("Belum ada pertanyaan.")
            else:
                for question in reversed(user_questions):
                    st.markdown(f"*{question[:50]}...*")
        
        # Tombol hanya menghapus riwayat chat
        if st.button("Hapus Riwayat Chat", type="secondary"):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.toast("Riwayat percakapan dihapus.", icon="🗑️")
            st.rerun()

    # --- Tampilan Chat Utama ---
    if not st.session_state.messages:
        initial_greeting = "Halo, saya **AMIA**. Silakan ajukan pertanyaan atau unggah dokumen di sidebar."
        with st.chat_message("assistant", avatar="🩺"):
            st.markdown(initial_greeting)
    
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🩺"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("source"):
                st.caption(f"Sumber Data: {message['source']}")

    # --- Logika Input dan Respons ---
    if user_input := st.chat_input("Tanyakan sesuatu pada AMIA..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("AMIA sedang berpikir..."):
                response_text = run_agent(user_input, retriever, st.session_state.memory, pdf_content=st.session_state.pdf_content)
                display_text, source = response_text, "AI Generative"
                if response_text and "<<SOURCE:" in response_text:
                    parts = response_text.split("<<SOURCE:")
                    display_text = parts[0].strip(); source = parts[1].replace(">>", "").strip()
                
                st.markdown(display_text)
                st.caption(f"Sumber Data: {source}")
                st.session_state.messages.append({"role": "assistant", "content": display_text, "source": source})
        st.rerun()

if __name__ == "__main__":
    main()