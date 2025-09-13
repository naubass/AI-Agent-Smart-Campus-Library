import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import warnings
import traceback
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

load_dotenv()

# load data
books_df = pd.read_csv("book_metadata.csv")
borrows_df = pd.read_csv("library_borrowing_book_data.csv")

# normalisasi tipe ID (paling aman: string)
books_df['book_id'] = books_df['book_id'].astype(str)
borrows_df['book_id'] = borrows_df['book_id'].astype(str)
borrows_df['borrower_id'] = borrows_df['borrower_id'].astype(str)

# optional: remove exact duplicates in books_df (keep first)
books_df = books_df.drop_duplicates(subset='book_id').reset_index(drop=True)

# collaborative Filtering Matrix (item-based)
user_book_matrix = borrows_df.groupby(['borrower_id','book_id']).size().unstack(fill_value=0)
user_book_matrix[user_book_matrix > 0] = 1

# handle NaN and build tokens
books_df['category'] = books_df['category'].fillna('')
books_df['subcategory'] = books_df['subcategory'].fillna('')
books_df['title'] = books_df['title'].fillna('')

books_df['tags'] = (books_df['category'] + ' ' + books_df['subcategory'] + ' ' + books_df['title']).str.lower()
books_df['tokens'] = books_df['tags'].str.split()  # simple tokenization; replace if you want advanced tokenization

mlb = MultiLabelBinarizer()
# fit on list-of-tokens; ensure order aligns with books_df
book_tags_matrix = mlb.fit_transform(books_df['tokens'])
book_tags_df = pd.DataFrame(book_tags_matrix, index=books_df['book_id'], columns=mlb.classes_)

# ensure user_book_matrix.columns are strings too
user_book_matrix.columns = user_book_matrix.columns.astype(str)

item_cf_sim_df = pd.DataFrame(
    cosine_similarity(user_book_matrix.T),
    index=user_book_matrix.columns,
    columns=user_book_matrix.columns
)

# item-based CBF similarity (books by tags)
item_cbf_sim_df = pd.DataFrame(
    cosine_similarity(book_tags_df),
    index=book_tags_df.index,
    columns=book_tags_df.index
)

# hybrid recommendation function
def recomended_books_hybird(user_id, top_n=5, alpha=0.5, category=None):
    try:
        user_id = str(user_id)
        # cek user
        if user_id not in user_book_matrix.index:
            return pd.DataFrame([{"title": "User tidak ditemukan"}])

        # buku yang pernah dipinjam user
        interacted = user_book_matrix.loc[user_id]
        interacted_books = interacted[interacted > 0].index.astype(str).tolist()

        if not interacted_books:
            return pd.DataFrame([{"title": "User belum pernah meminjam buku"}])

        # pastikan hanya gunakan interacted_books yang ada di matrix similarity
        cf_books = [b for b in interacted_books if b in item_cf_sim_df.columns]
        cbf_books = [b for b in interacted_books if b in item_cbf_sim_df.columns]

        # Hitung skor — kontrol ketika list kosong
        if cf_books:
            cf_scores = item_cf_sim_df[cf_books].mean(axis=1)
        else:
            cf_scores = pd.Series(0, index=item_cf_sim_df.index)

        if cbf_books:
            cbf_scores = item_cbf_sim_df[cbf_books].mean(axis=1)
        else:
            cbf_scores = pd.Series(0, index=item_cbf_sim_df.index)

        # align indices (union) dan isi 0 jika tidak ada
        all_idx = cf_scores.index.union(cbf_scores.index)
        cf_scores = cf_scores.reindex(all_idx, fill_value=0)
        cbf_scores = cbf_scores.reindex(all_idx, fill_value=0)

        hybrid_scores = alpha * cf_scores + (1 - alpha) * cbf_scores

        # jangan rekomendasikan buku yang sudah dipinjam user (hanya jika index cocok)
        hybrid_scores = hybrid_scores.drop([b for b in interacted_books if b in hybrid_scores.index], errors='ignore')

        # urutkan kandidat berdasarkan hybrid score
        ranked_ids = hybrid_scores.sort_values(ascending=False).index.tolist()

        # pastikan kandidat juga ada di books_df
        ranked_ids = [rid for rid in ranked_ids if rid in books_df['book_id'].values]

        if not ranked_ids:
            return pd.DataFrame([{"title": "Tidak ada rekomendasi yang cocok"}])

        rec_books = books_df.set_index('book_id').loc[ranked_ids]

        # Jika ada filter kategori: ambil top_n dari kategori itu, kalau kurang -> fill dengan kandidat lainnya
        if category:
            cat = category.strip().lower()
            rec_books_cat = rec_books[rec_books['category'].str.lower().str.contains(cat)]

            # fallback: langsung ambil semua buku kategori tsb kalau rec kosong
            if rec_books_cat.empty:
                rec_books_cat = books_df[books_df['category'].str.lower().str.contains(cat)]

            results = rec_books_cat.head(top_n)
            if len(results) < top_n:
                needed = top_n - len(results)
                fallback = rec_books[~(rec_books.index.isin(results.index))].head(needed)
                results = pd.concat([results, fallback])

            return results.head(top_n)

        # Jika tanpa filter kategori -> diversifikasi kategori (1 per kategori) lalu fill jika kurang
        top_books = []
        seen_categories = set()
        seen_titles = set()

        for idx, row in rec_books.iterrows():
            if row['title'] not in seen_titles and row['category'] not in seen_categories:
                top_books.append(idx)
                seen_categories.add(row['category'])
                seen_titles.add(row['title'])
            if len(top_books) >= top_n:
                break

        # fill sisa dengan buku terbaik (jika kurang dari top_n)
        if len(top_books) < top_n:
            for idx in rec_books.index:
                if idx not in top_books:
                    top_books.append(idx)
                    if len(top_books) >= top_n:
                        break

        return rec_books.loc[top_books]

    except Exception as e:
        # beri trace untuk debugging
        tb = traceback.format_exc()
        return pd.DataFrame([{"error": str(e), "traceback": tb}])

# Streamlit part (tidy output)
import streamlit as st

st.set_page_config(page_title="Smart Campus Library", page_icon="📚", layout="wide")
st.header("Smart Campus Library")
st.subheader("Rekomendasi Buku")

# Sidebar Borrower Filter
st.sidebar.header("Filter User")

# pastikan borrower_id dibaca sebagai int -> sort -> balik ke string
user_list = sorted(borrows_df['borrower_id'].astype(int).unique().tolist())
user_list = [str(uid) for uid in user_list]  # konsisten ke string

# selectbox dengan default ke 5001 kalau ada
default_index = user_list.index("5001") if "5001" in user_list else 0
user_select = st.sidebar.selectbox("Pilih Borrower ID", user_list, index=default_index)

# slider dan filter lain
top_n = st.sidebar.slider("Jumlah buku rekomendasi", 1, 10, 5)
alpha = st.sidebar.slider("Bobot CF vs CBF (alpha)", 0.0, 1.0, 0.5)
category_filter = st.sidebar.text_input("Filter kategori (opsional, contoh: Sains)")

if st.sidebar.button("Generate Rekomendasi"):
    recs = recomended_books_hybird(user_select, top_n=top_n, alpha=alpha, category=category_filter or None)
    st.subheader(f"Rekomendasi untuk Borrower ID {user_select}:")
    st.dataframe(recs)

# AI Agent Integration
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    llm = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-2.5-flash", temperature=0.3)

    @tool()
    def hybird_recommendation(user_id: str, top_n: int = 5, alpha: float = 0.5, category: str = None):
        """
        Hybrid recommendation with diversity for a borrower.
        Supports optional category filtering.
        """
        try:
            recs = recomended_books_hybird(user_id, top_n=top_n, alpha=alpha, category=category)
            if isinstance(recs, pd.DataFrame):
                if 'error' in recs.columns:
                    return f"Error: {recs.iloc[0].get('error')}\nTrace:\n{recs.iloc[0].get('traceback')}"
                lines = []
                for i, (idx, row) in enumerate(recs.reset_index().iterrows(), start=1):
                    author = row.get('author', '-') if 'author' in row else '-'
                    year = row.get('year', '-') if 'year' in row else '-'
                    lines.append(f"{i}. {row['title']} oleh {author} ({row.get('category','-')}, {year})")
                return "\n".join(lines)
            else:
                return str(recs)
        except Exception as e:
            return "Tool error: " + str(e) + "\n" + traceback.format_exc()

    agent = create_react_agent(model=llm, tools=[hybird_recommendation])

    st.header("🤖 AI Agent Recommendation")
    agent_input = st.text_input("Prompt AI Agent (contoh: 'Rekomendasikan buku untuk borrower 5001 top 5 alpha 0.5'):")

    if agent_input:
        with st.spinner("AI Agent sedang memproses..."):
            try:
                response = agent.invoke({"messages":[HumanMessage(content=agent_input)]})
                st.success(response["messages"][-1].content)
            except Exception as e:
                st.error("Agent error: " + str(e))
                st.code(traceback.format_exc())
else:
    st.warning("GOOGLE_API_KEY belum di-set di Streamlit secrets")
