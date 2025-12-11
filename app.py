# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from pipeline import (
    load_character_data, load_embeddings, extract_traits, 
    embed_text_ollama, match_user, generate_final_report,
    CHAR_DATA_PATH, EMBEDDINGS_PATH, TOP_K
)

# --- Data Loading (Cached) ---

@st.cache_resource
def cached_load_data():
    """Load large data files once and cache them."""
    try:
        char_data = load_character_data(CHAR_DATA_PATH)
        char_embeddings = load_embeddings(EMBEDDINGS_PATH)
        return char_data, char_embeddings
    except FileNotFoundError as e:
        st.error(f"檔案載入錯誤：{e}")
        return None, None
    except Exception as e:
        st.error(f"載入資料時發生未知錯誤: {e}")
        return None, None

# Load data at the start
char_data, char_embeddings = cached_load_data()


# --- Streamlit UI Components ---

st.set_page_config(
    page_title="ACG Ideal Matchmaker",
    layout="wide"
)

st.title("💖 ACG 理想型匹配系統 (Waifu/Husbando Finder)")
st.subheader("請描述您心目中的二次元理想型，讓我們找到與之最相似的角色！")

# Display system status
if char_data is None or char_embeddings is None:
    st.warning("系統無法載入角色資料，請檢查檔案路徑。")
    st.stop()
else:
    st.info(f"系統準備就緒：已載入 {len(char_data)} 個角色資料。")


# Sidebar for user input (Ideal Type)
with st.sidebar:
    st.header("💖 你的理想型（核心萌點）")
    user_text = st.text_area(
        "請描述你心目中**最喜歡的萌點、性格、行為傾向**：",
        height=200,
        placeholder="例如：我喜歡有點傲嬌，但內心非常溫柔體貼，會默默照顧人的類型。外表看起來冷靜，但實際上很容易害羞，偶爾會展現出意外的反差萌。",
        key="user_input"
    )

    run_button = st.button("🚀 開始匹配！ (尋找你的 TA)", type="primary", use_container_width=True)

st.markdown("---")

# Main execution logic
if run_button and user_text:
    
    # 1. Run Pipeline (Steps 2-5)
    report, traits, matches = None, None, None
    try:
        # Step 2: Trait extraction
        with st.spinner("正在呼叫 Gemini LLM 提取理想型萌點..."):
            traits = extract_traits(user_text)
        st.success(f"核心萌點抽取完成: {traits}")

        # Step 3 & 4: Embedding and Matching
        with st.spinner("正在呼叫 Ollama 嵌入向量與計算相似度..."):
            trait_text = "; ".join(traits)
            user_vec = embed_text_ollama(trait_text)
            match_scores = match_user(user_vec, char_embeddings, TOP_K)
            matches = [(idx, score, char_data[idx]) for idx, score in match_scores]
        st.success("相似度計算完成！(已套用 Min-Max 縮放)")

        # Step 5: Report Generation
        with st.spinner("正在呼叫 Gemini LLM 生成最終匹配報告..."):
            report = generate_final_report(user_text, traits, matches)
        st.success("報告生成完成！")

    except RuntimeError as e:
        st.error(f"匹配過程中發生錯誤：{e}")
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()
    
    # --- Display Match Report ---
    st.header("📜 最終匹配報告")
    st.markdown(report)
    
    # --- Display Visualizations ---
    st.header("📊 相似度分數一覽 (Min-Max 縮放)")
    
    # Prepare data for charts
    match_df = pd.DataFrame([
        {"Name": char["name"], "Score": score, "Rank": i + 1, "Top_Traits": ', '.join(char.get('moe_traits', [])[:3])}
        for i, (_, score, char) in enumerate(matches)
    ])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Top 5 理想型匹配分數")
        # Bar Chart for clear comparison of scores
        fig_bar = px.bar(
            match_df, 
            x='Name', 
            y='Score', 
            color='Name',
            title='相似度得分 (Min-Max 縮放，範圍 0-100)',
            text='Score',
            height=500
        )
        fig_bar.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        
        # Y-axis starts near the lowest score but never below 0
        min_score_actual = match_df['Score'].min() if not match_df.empty else 0
        y_axis_start = max(0, min_score_actual - 5)
        fig_bar.update_yaxes(range=[y_axis_start, 100], ticksuffix="%")
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        # 
    
    with col2:
        st.subheader("原始結果")
        
        # Display table of raw results
        st.dataframe(
            match_df[['Rank', 'Name', 'Score', 'Top_Traits']].set_index('Rank'),
            column_order=('Name', 'Score', 'Top_Traits'),
            column_config={
                "Score": st.column_config.ProgressColumn("得分 (Max 100)", format="%.1f", max_value=100),
                "Name": "角色名稱",
                "Top_Traits": "主要特質"
            },
            use_container_width=True
        )

        st.subheader("你的理想型萌點")
        st.markdown(f"**提取萌點:** `{', '.join(traits)}`")
        
# --- Initial state or when input is empty ---
else:
    st.info("請在左側欄位輸入您理想型的描述，然後點擊「開始匹配！」")