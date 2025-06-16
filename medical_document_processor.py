# File ini bertanggung jawab untuk memproses dokumen medis yang diunggah (PDF).

import re
from typing import Dict, Any
import PyPDF2
import io

def extract_medical_info_from_text(text: str) -> Dict[str, Any]:
    """
    Mengekstrak informasi medis terstruktur dari sebuah string teks mentah
    menggunakan serangkaian pola Regular Expression (regex).
    """
    medical_info = {
        'diagnosa': [], 'riwayat_penyakit': [], 'obat_obatan': [],
        'hasil_lab': [], 'vital_signs': [], 'rekomendasi': []
    }
    
    # Kumpulan pola regex untuk mencari informasi kunci.
    patterns = {
        'diagnosa': r'(?i)diagnos[ai]s?[:\s]+([\w\s,.-]+)',
        'riwayat_penyakit': r'(?i)riwayat penyakit[:\s]+([\w\s,.-]+)',
        'obat_obatan': r'(?i)obat(?:-obatan)?[:\s]+([\w\s,.()-]+)',
        'hasil_lab': r'(?i)hasil lab[:\s]+([\w\s,./<>-]+)',
        'vital_signs': r'(?i)tanda vital|vital signs[:\s]+([\w\s,./=°]+)',
        'rekomendasi': r'(?i)rekomendasi|saran[:\s]+([\w\s,.-]+)'
    }

    for key, pattern in patterns.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            # 1. Ambil grup yang cocok
            found_group = match.group(1)
            
            # 2. Periksa apakah grup tersebut tidak kosong (bukan None) sebelum diproses
            if found_group:
                clean_text = found_group.strip().replace('\n', ' ').strip()
                if clean_text: # Pastikan teks tidak hanya spasi kosong setelah dibersihkan
                    medical_info[key].append(clean_text)
    
    return medical_info


def extract_medical_document(uploaded_file) -> Dict[str, Any]:
    """
    Membaca file PDF yang diunggah, mengekstrak metadata dan teks lengkap,
    lalu memproses teks untuk mendapatkan informasi terstruktur.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        
        metadata = {
            'total_pages': len(pdf_reader.pages),
            'document_info': pdf_reader.metadata or {},
        }
        
        full_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
        
        doc_type = "Dokumen Medis Umum"
        if re.search(r'(?i)hasil\s+lab|laboratory\s+result', full_text):
            doc_type = "Hasil Laboratorium"
        elif re.search(r'(?i)resep|prescription', full_text):
            doc_type = "Resep Obat"
        elif re.search(r'(?i)rekam\s+medis|medical\s+record', full_text):
            doc_type = "Rekam Medis"
        elif re.search(r'(?i)rujukan|referral', full_text):
            doc_type = "Surat Rujukan"
            
        medical_info = extract_medical_info_from_text(full_text)
        
        return {
            'document_type': doc_type,
            'metadata': metadata,
            'medical_info': medical_info,
            'full_text': full_text
        }

    except Exception as e:
        return {
            'error': f"Gagal memproses file PDF: {e}",
            'document_type': "Error",
            'metadata': {},
            'medical_info': {},
            'full_text': ""
        }