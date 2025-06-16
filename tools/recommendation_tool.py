# tools/recommendation_tool.py
# Berisi fungsi untuk memberikan rekomendasi kesehatan umum berdasarkan jenis penyebab.

def recommend_actions(query: str, retriever) -> str:
    """
    Memberikan rekomendasi tindakan kesehatan umum berdasarkan jenis penyebab.
    Args:
        query (str): Pertanyaan atau topik kesehatan.
        retriever: Objek FaissRetriever yang dikirim dari main.py.
    Returns:
        str: String berisi daftar rekomendasi.
    """
    relevant_data = retriever.get_relevant(query, k=5)
    
    if not relevant_data:
        return "Informasi tidak ditemukan, rekomendasi umum: selalu jaga kesehatan dan konsultasi dengan tenaga medis profesional."

    types = set(item.get('Type', '') for item in relevant_data)
    
    recommendations = []
    for type_ in types:
        if 'Bencana Alam' in type_:
            recommendations.append("Untuk Bencana Alam: Perhatikan peringatan dini dari BMKG, siapkan tas siaga bencana, dan kenali jalur evakuasi.")
        elif 'Penyakit' in type_ or 'Non Alam' in type_:
            recommendations.append("Untuk Penyakit: Terapkan pola hidup sehat, lakukan pemeriksaan rutin, dan ikuti program vaksinasi pemerintah.")
        elif 'Bencana Sosial' in type_:
            recommendations.append("Untuk Bencana Sosial: Tingkatkan kewaspadaan lingkungan dan ikuti protokol keamanan yang berlaku.")
    
    if not recommendations:
        return "Tidak ada rekomendasi spesifik yang bisa diberikan berdasarkan data yang ada. Mohon jaga kesehatan."

    return "Berdasarkan jenis penyebab yang teridentifikasi, berikut beberapa rekomendasi umum:\n- " + "\n- ".join(list(set(recommendations)))