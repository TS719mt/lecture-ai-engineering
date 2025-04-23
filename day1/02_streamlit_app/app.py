# app.py
import streamlit as st
import ui                   # UIモジュール
import llm                  # LLMモジュール
import database             # データベースモジュール
import metrics              # 評価指標モジュール
import data                 # データモジュール
import torch
from transformers import pipeline
from config import MODEL_NAME
from huggingface_hub import HfFolder

# --- カスタムCSS ---
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .stTitle {
        font-family: 'Helvetica Neue', sans-serif;
        color: #2E4057;
    }
    .stSidebar {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# --- アプリケーション設定 ---
st.set_page_config(
    page_title="Gemma Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/your-repo',
        'Report a bug': "https://github.com/yourusername/your-repo/issues",
        'About': "# Gemma Chatbot\nAIとの対話を通じて新しい発見を。"
    }
)

# --- 初期化処理 ---
# NLTKデータのダウンロード（初回起動時など）
metrics.initialize_nltk()

# データベースの初期化（テーブルが存在しない場合、作成）
database.init_db()

# データベースが空ならサンプルデータを投入
data.ensure_initial_data()

# LLMモデルのロード（キャッシュを利用）
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with st.spinner("🔄 モデルを読み込んでいます..."):
            pipe = pipeline(
                "text-generation",
                model=MODEL_NAME,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device=device
            )
            st.success(f"✨ モデル '{MODEL_NAME}' の読み込みに成功しました。")
            return pipe
    except Exception as e:
        st.error(f"❌ モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.warning("💡 GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None

pipe = llm.load_model()

# --- メインアプリケーション ---
col1, col2, col3 = st.columns([1,6,1])
with col2:
    st.title("🤖 Gemma 2 Chatbot")
    st.markdown("""
    <div style='text-align: center; padding: 1rem; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            Gemmaモデルを活用した次世代のチャットボット。<br>
            あなたの質問に対して、的確な回答とインサイトを提供します。
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- サイドバー ---
st.sidebar.markdown("""
<div style='padding: 1rem 0; text-align: center;'>
    <h1 style='color: #2E4057;'>🎯 ナビゲーション</h1>
</div>
""", unsafe_allow_html=True)

# セッション状態を使用して選択ページを保持
if 'page' not in st.session_state:
    st.session_state.page = "チャット"

page = st.sidebar.radio(
    "",  # ラベルを空にして、上のマークダウンタイトルを使用
    ["💬 チャット", "📚 履歴閲覧", "⚙️ サンプルデータ管理"],
    key="page_selector",
    index=["💬 チャット", "📚 履歴閲覧", "⚙️ サンプルデータ管理"].index(f"💬 {st.session_state.page}" if st.session_state.page == "チャット" else f"📚 {st.session_state.page}" if st.session_state.page == "履歴閲覧" else f"⚙️ {st.session_state.page}"),
    on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_selector.split(" ")[1])
)

# --- メインコンテンツ ---
if st.session_state.page == "チャット":
    if pipe:
        ui.display_chat_page(pipe)
    else:
        st.error("⚠️ チャット機能を利用できません。モデルの読み込みに失敗しました。")
elif st.session_state.page == "履歴閲覧":
    ui.display_history_page()
elif st.session_state.page == "サンプルデータ管理":
    ui.display_data_page()

# --- フッター ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem;'>
    <p style='color: #666; font-size: 0.9rem;'>
        Made with ❤️ by Your Team<br>
        <a href='https://github.com/yourusername/your-repo' target='_blank' style='color: #4CAF50;'>GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)