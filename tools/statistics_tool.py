# Berisi fungsi untuk menganalisis informasi medis dari database.

import pandas as pd
import re
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Penyebab Kematian di Indonesia yang Dilaporkan - Clean.csv')

def find_extremes_in_year(query: str) -> str:
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return f"Error: File data tidak ditemukan."
    year_match = re.search(r'\b(\d{4})\b', query)
    if not year_match: return "Gagal menemukan tahun dalam pertanyaan."
    year = int(year_match.group(1))
    df_year = df[df['Year'] == year]
    if df_year.empty: return f"Tidak ada data untuk tahun {year}."
    if 'tertinggi' in query.lower():
        record = df_year.loc[df_year['Total Deaths'].idxmax()]
        analysis_type = "tertinggi"
    elif 'terendah' in query.lower():
        record = df_year.loc[df_year['Total Deaths'].idxmin()]
        analysis_type = "terendah"
    else:
        return "Query tidak valid. Gunakan tool ini untuk mencari data 'tertinggi' atau 'terendah'."
    return (f"Analisis untuk tahun {year}:\n- Penyebab Kematian {analysis_type.capitalize()}: {record['Cause']}\n- Tipe: {record['Type']}\n- Jumlah Kematian: {int(record['Total Deaths']):,} jiwa.")

def analyze_cause_trend(query: str, retriever) -> str:
    search_term = query.replace("penyakit", "").replace("analisis", "").replace("tren", "").strip()
    relevant_data = retriever.get_relevant(search_term, k=20)
    if not relevant_data: return f"Tidak ditemukan data historis untuk '{search_term}'."
    keyword = search_term.lower().split()[0]
    records = [item for item in relevant_data if keyword in item.get('Cause', '').lower()]
    if not records: return f"Data ditemukan, namun tidak ada yang cocok spesifik dengan '{search_term}'."
    df_cause = pd.DataFrame(records).drop_duplicates(subset=['Year']).sort_values('Year')
    if len(df_cause) < 2: return f"Data untuk '{search_term}' ditemukan, tetapi tidak cukup untuk analisis tren."
    stats = {'cause': df_cause['Cause'].mode()[0],'period': f"{df_cause['Year'].min()} - {df_cause['Year'].max()}",'reports': len(df_cause),'mean': df_cause['Total Deaths'].mean(),'max': df_cause['Total Deaths'].max(),'max_year': df_cause.loc[df_cause['Total Deaths'].idxmax(), 'Year'],'min': df_cause['Total Deaths'].min(),'min_year': df_cause.loc[df_cause['Total Deaths'].idxmin(), 'Year']}
    return (f"Berikut analisis tren untuk '{stats['cause']}' periode {stats['period']}:\n- Laporan Tahunan: {stats['reports']} data.\n- Rata-rata Kematian: {stats['mean']:,.0f} jiwa/tahun.\n- Puncak Kematian: {stats['max']:,.0f} jiwa ({stats['max_year']}).\n- Kematian Terendah: {stats['min']:,.0f} jiwa ({stats['min_year']}).")