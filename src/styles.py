# styles.py
def css() -> str:
    return """
    :root{
      --bg:#0b1220;
      --panel:#0f172a;
      --panel-2:#111827;
      --text:#e5e7eb;
      --muted:#9ca3af;
      --primary:#22d3ee;
      --ring:rgba(34, 211, 238, .25);
      --radius:16px;
    }
    .stApp { background: radial-gradient(1200px 800px at 10% -10%, #0b1730 0%, var(--bg) 60%) }
    header, footer {visibility: hidden;}
    .app-header{
      padding: 18px 22px; border-radius: var(--radius);
      background: linear-gradient(180deg, rgba(34,211,238,.12), rgba(34,211,238,0));
      border: 1px solid rgba(255,255,255,.06);
      box-shadow: 0 8px 30px rgba(0,0,0,.35);
      color: var(--text);
    }
    .app-title{ font-size: 28px; font-weight: 800; letter-spacing: .2px; margin: 0; }
    .app-sub{ margin: 6px 0 0 0; color: var(--muted); }
    .card{
      background: var(--panel);
      border: 1px solid rgba(255,255,255,.06);
      border-radius: var(--radius);
      padding: 18px;
      box-shadow: 0 10px 30px rgba(0,0,0,.35);
    }
    .soft{ background: var(--panel-2) }
    .section-title{ font-weight: 700; font-size: 16px; margin: 0 0 8px 0; color: var(--text); }
    .stDataFrame, .stTable { border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,.06); }
    [data-testid="stMetric"] {
      background: var(--panel); padding: 16px; border-radius: 14px;
      border: 1px solid rgba(255,255,255,.06); box-shadow: 0 8px 20px rgba(0,0,0,.25);
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] { font-weight: 600; }
    [data-testid="stFileUploader"] > div {
      background: var(--panel); border-radius: 14px; border: 1px dashed rgba(255,255,255,.18); padding: 14px;
    }
    .stTextInput input:focus, .stSelectbox:focus, .stFileUploader:focus-within { box-shadow: 0 0 0 3px var(--ring); }
    .stMarkdown h3, .stMarkdown h2 { color: var(--text) !important; }
    """
