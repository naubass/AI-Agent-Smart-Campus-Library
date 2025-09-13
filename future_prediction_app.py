import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from prophet import Prophet
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# streamlit setup
st.set_page_config(layout="centered", page_title="📚 AI Agent Smart Library")
st.title("📚 AI Agent Smart Library - Prediksi Trending Book")

# load data
@st.cache_data
def load_data():
    df = pd.read_csv('library_borrowing_book_data.csv')
    df["borrow_date"] = pd.to_datetime(df["borrow_date"])
    return df

df_borrow = load_data()

# function for AI Agent
def _trending_predict_book(category: str = None, days_ahead: int = 30) -> dict:
    df = df_borrow.copy()
    if category and category.lower() != "semua":
        df = df[df["category"].str.lower() == category.lower()]

    df_borrow_day = df.groupby("borrow_date").size().reset_index(name="y")
    df_borrow_day.rename(columns={"borrow_date": "ds"}, inplace=True)

    holiday_df = pd.read_csv('library_holidays_exams.csv')
    holiday_df["date"] = pd.to_datetime(holiday_df["date"])
    holiday_df = holiday_df.rename(columns={"date": "ds", "event": "holiday"})
    holiday_df["lower_window"] = -1
    holiday_df["upper_window"] = 1

    m = Prophet(
        holidays=holiday_df,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False
    )
    m.fit(df_borrow_day)

    future = m.make_future_dataframe(periods=days_ahead)
    forecast = m.predict(future)

    pred_mean = forecast.tail(days_ahead)["yhat"].mean()
    pred_total = forecast.tail(days_ahead)["yhat"].sum()

    top_books = (
        df.groupby("title")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(5)
    )

    text_result = (
        f"📊 Prediksi rata-rata peminjaman harian dalam {days_ahead} hari ke depan "
        f"sekitar **{pred_mean:.0f} buku/hari**.\n\n"
        f"📈 Total prediksi peminjaman periode tersebut: **{pred_total:.0f} buku**.\n\n"
        f"📚 Buku yang kemungkinan trending: {', '.join(top_books['title'].tolist())}."
    )

    return {
        "text": text_result,
        "forecast": forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        "top_books": top_books
    }

# tool prediksi
@tool
def trending_predict_book(category: str = None, days_ahead: int = 30) -> str:
    """
    Prediksi tren peminjaman buku berdasarkan kategori untuk periode tertentu.

    Fungsi ini memanfaatkan model Prophet untuk memproyeksikan jumlah peminjaman harian
    buku dalam kategori tertentu, termasuk estimasi buku yang kemungkinan akan trending.
    Cocok digunakan untuk analisis popularitas buku dan perencanaan inventaris perpustakaan.

    Parameters
    ----------
    category : str, optional
        Nama kategori buku yang ingin diprediksi (default None, artinya semua kategori).
        Case-insensitive. Contoh: "Teknologi", "Sains", "Fiksi".
    days_ahead : int, optional
        Jumlah hari ke depan untuk melakukan prediksi (default 30). Harus > 0.

    Returns
    -------
    dict
        Dictionary dengan kunci:
        - "text" : str
            Ringkasan prediksi berupa rata-rata harian, total peminjaman, dan daftar 5 buku teratas.
        - "forecast" : pd.DataFrame
            Data prediksi harian dengan kolom ['ds', 'yhat', 'yhat_lower', 'yhat_upper'].
        - "top_books" : pd.DataFrame
            Daftar 1 sampai 5 buku teratas berdasarkan frekuensi peminjaman historis.

    Example
    -------
    >>> result = trending_predict_book(category="Teknologi", days_ahead=14)
    >>> print(result["text"])
    "📊 Prediksi rata-rata peminjaman harian dalam 14 hari ke depan sekitar 7 buku/hari..."
    
    Notes
    -----
    - Fungsi ini hanya memprediksi tren peminjaman, bukan rekomendasi personal mahasiswa.
    - Pastikan dataset 'library_borrowing_book_data.csv' sudah dimuat dan mengandung kolom
      ['title', 'category', 'borrow_date'].
    - Untuk akurasi lebih baik, disarankan menyediakan data holiday/exam untuk Prophet.
    """
    result = _trending_predict_book(category=category, days_ahead=days_ahead)
    return result["text"]

#gemini LLM setup
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    llm = ChatGoogleGenerativeAI(
        api_key=api_key,
        model="gemini-2.5-flash",
        temperature=0.3
    )
    agent = create_react_agent(model=llm, tools=[trending_predict_book])
else:
    st.error("GOOGLE_API_KEY belum di-set di environment variable.")

# AI AGENT
st.header("🤖 Prediksi Menggunakan AI Agent")
agent_input = st.text_input("Masukkan prompt AI Agent (misal: 'Prediksi kategori Teknologi 7 hari ke depan'):")

if agent_input:
    with st.spinner("AI Agent sedang memproses..."):
        response = agent.invoke({"messages": [HumanMessage(content=agent_input)]})
        st.success(response["messages"][-1].content)

# generate prediksi manual
st.header("📊 Prediksi Manual (Filter Sidebar)")
st.sidebar.header("🔎 Filter Prediksi Manual")
selected_cat = st.sidebar.selectbox(
    "Pilih kategori:", options=["Semua"] + list(df_borrow["category"].unique())
)
days_ahead = st.sidebar.slider(
    "Prediksi untuk (hari ke depan):", min_value=7, max_value=90, value=30
)

if st.sidebar.button("Generate Prediksi"):
    with st.spinner("Sedang memproses prediksi..."):
        if selected_cat == "Semua":
            result = _trending_predict_book(days_ahead=days_ahead)
        else:
            result = _trending_predict_book(category=selected_cat, days_ahead=days_ahead)

        # tampilkan teks hasil
        st.success(result["text"])

        # matpotlib plot
        import matplotlib.pyplot as plt

        df_hist = df_borrow.copy()
        df_hist_daily = df_hist.groupby("borrow_date").size().reset_index(name="y")
        forecast_df = result["forecast"].copy()
        df_hist_daily["y_roll"] = df_hist_daily["y"].rolling(7, min_periods=1).mean()

        plt.figure(figsize=(12,6))
        plt.plot(df_hist_daily["borrow_date"], df_hist_daily["y_roll"], 
                 label="Data Historis (7-hari avg)", color="blue", linewidth=2)
        plt.plot(forecast_df["ds"], forecast_df["yhat"], label="Prediksi", color="orange", linewidth=2)
        if "yhat_lower" in forecast_df.columns and "yhat_upper" in forecast_df.columns:
            plt.fill_between(forecast_df["ds"], forecast_df["yhat_lower"], forecast_df["yhat_upper"], 
                             color="orange", alpha=0.2, label="Confidence Interval")
        plt.xlabel("Tanggal")
        plt.ylabel("Jumlah Peminjaman")
        plt.title(f"Prediksi Peminjaman Buku - Kategori: {selected_cat}")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

        # top books
        st.subheader("🔥 Top 5 Buku Trending")
        st.bar_chart(result["top_books"].set_index("title"))


