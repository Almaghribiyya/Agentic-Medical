# Berisi fungsi untuk mengambil dan merangkum informasi medis dari database.

def get_medical_info(query: str, retriever) -> str:
    """
    Mencari dan merangkum informasi medis tentang suatu topik dari database.
    Args:
        query (str): Pertanyaan atau topik medis yang ingin dicari.
        retriever: Objek FaissRetriever yang dikirim dari main.py.
    Returns:
        str: String berisi rangkuman informasi yang ditemukan.
    """
    relevant_data = retriever.get_relevant(query, k=5)
    
    if not relevant_data:
        return f"Informasi tentang '{query}' tidak tersedia dalam database internal."
    
    info_parts = []
    seen_causes = set()
    for item in relevant_data:
        cause = item.get('Cause', '')
        if cause and cause not in seen_causes:
            info = (
                f"- Untuk penyebab '{cause}' (Tipe: {item.get('Type')}): "
                f"Tercatat {int(item.get('Total Deaths', 0)):,} kematian pada tahun {item.get('Year')}. "
                f"(Sumber: {item.get('Source')})."
            )
            info_parts.append(info)
            seen_causes.add(cause)
    
    if not info_parts:
        return f"Data untuk '{query}' ditemukan, namun tidak ada informasi detail yang bisa dirangkum."
        
    header = f"Berikut rangkuman informasi yang ditemukan terkait '{query}':\n"
    return header + "\n".join(info_parts)