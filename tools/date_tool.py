# tools/date_tool.py
from datetime import datetime

def get_current_date(query: str = "") -> str:
    """Mengembalikan tanggal dan waktu saat ini. Abaikan argumen query."""
    now = datetime.now()
    return f"Tanggal dan waktu saat ini adalah: {now.strftime('%A, %d %B %Y, %H:%M:%S')} WIB."