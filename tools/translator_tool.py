# File ini adalah bagian dari aplikasi Streamlit yang menerjemahkan istilah medis

from google.cloud import translate_v2 as translate
import os

# Inisialisasi klien terjemahan.
# Otentikasi akan berjalan otomatis di lingkungan Google Cloud seperti Streamlit Cloud
# jika API-nya sudah diaktifkan.
try:
    translate_client = translate.Client()
except Exception as e:
    print(f"Peringatan: Gagal menginisialisasi Google Translate Client: {e}")
    translate_client = None

def translate_medical_terms(query: str) -> str:
    """
    Menerjemahkan istilah medis menggunakan Google Cloud Translation API.
    Format input: 'teks yang akan diterjemahkan to bahasa tujuan'.
    Contoh: 'headache to indonesia' atau 'sakit kepala to en'.
    """
    if not translate_client:
        return "Error: Klien terjemahan tidak berhasil diinisialisasi. Pastikan Cloud Translation API sudah aktif di proyek Google Cloud Anda."

    try:
        parts = query.lower().split(' to ')
        if len(parts) != 2:
            return "Format tidak valid. Gunakan format: 'teks to bahasa_tujuan' (contoh: 'fever to indonesia')."
        
        text_to_translate = parts[0].strip()
        dest_lang = parts[1].strip()
        
        # Mapping bahasa umum ke kode bahasa ISO 639-1
        lang_map = {'indonesia': 'id', 'inggris': 'en', 'jawa': 'jw', 'sunda': 'su'}
        dest_code = lang_map.get(dest_lang, dest_lang)
        
        # Panggil API terjemahan
        result = translate_client.translate(text_to_translate, target_language=dest_code)
        
        return f"Hasil terjemahan '{text_to_translate}' adalah: {result['translatedText']}"

    except Exception as e:
        return f"Gagal menerjemahkan: {e}. Pastikan Cloud Translation API sudah diaktifkan."