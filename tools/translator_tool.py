# Berisi fungsi untuk menerjemahkan istilah penyakit dan sebainya.
from googletrans import Translator

def translate_medical_terms(query: str) -> str:
    """
    Menerjemahkan teks atau istilah medis.
    Args:
        query (str): Teks yang ingin diterjemahkan. Harus dalam format 'teks_sumber to bahasa_tujuan'.
                     Contoh: 'headache to indonesia' atau 'sakit kepala to en'.
    Returns:
        str: Teks yang sudah diterjemahkan.
    """
    try:
        parts = query.lower().split(' to ')
        if len(parts) != 2:
            return "Format tidak valid. Gunakan format: 'teks to bahasa_tujuan' (contoh: 'fever to indonesia')."
        
        text_to_translate = parts[0].strip()
        dest_lang = parts[1].strip()
        
        # Mapping bahasa umum ke kode bahasa
        lang_map = {'indonesia': 'id', 'inggris': 'en', 'jawa': 'jw'}
        dest_code = lang_map.get(dest_lang, dest_lang)
        
        translator = Translator()
        translated = translator.translate(text_to_translate, dest=dest_code)
        return f"Hasil terjemahan '{text_to_translate}' adalah: {translated.text}"
    except Exception as e:
        return f"Gagal menerjemahkan: {e}"