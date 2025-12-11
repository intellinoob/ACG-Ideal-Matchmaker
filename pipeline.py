#!/usr/bin/env python3
"""
Core pipeline logic: Ideal Type Matching System
1. Gemini LLM: extract ideal type traits
2. Ollama: embed trait tokens
3. Cosine similarity with DYNAMIC MIN-MAX SCALING (scores range from 0 to 100)
4. Gemini LLM: generate final match report
"""

import os
import json
import time
import numpy as np
import re
from typing import List, Tuple

from sklearn.metrics.pairwise import cosine_similarity
from google import genai
import ollama 

# ============================================================
# Global Configuration & Constants
# ============================================================
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Please set the environment variable.")

MODEL_NAME = "gemini-2.5-flash"
GENAI_DELAY = 1.0
GENAI_RETRY = 3

OLLAMA_EMBED_MODEL = "bge-m3"

CHAR_DATA_PATH = "character_data_with_id.json"
EMBEDDINGS_PATH = "character_embeddings_ollama.npy"
TOP_K = 5

# ============================================================
# Gemini Client Setup
# ============================================================
try:
    client = genai.Client(api_key=API_KEY) 
except Exception as e:
    raise RuntimeError(f"Failed to initialize Gemini Client: {e}")


# ============================================================
# Data Loading & Utilities
#============================================================

def load_character_data(path: str):
    """Loads character metadata from JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Character data file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i, item in enumerate(data):
        item.setdefault("id", i)
    return data

def load_embeddings(path: str):
    """Loads character embeddings from NumPy file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found at: {path}")
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def gemini_generate(prompt: str) -> str:
    """Handles content generation with retries using the correct client call."""
    for i in range(GENAI_RETRY):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,  
                contents=prompt
            )
            text = response.text
            if text:
                return text
        except Exception as e:
            print(f"[Gemini] Retry {i+1}/{GENAI_RETRY}: {e}")
            time.sleep(GENAI_DELAY)
    raise RuntimeError("Gemini API failed after retries.")


# ============================================================
# Pipeline Steps
# ============================================================

def extract_traits(user_text: str) -> List[str]:
    """Step 2: Extracts ideal type traits via Gemini (Optimized prompt)."""
    prompt = f"""
你是一個嚴謹的二次元理想型特質分析師。
請從以下**理想型描述**中，提取 **8 到 12 個** 最能代表其**核心萌點、性格或行為傾向**的關鍵詞彙。
請注意：
- **只**以 JSON 陣列格式輸出，例如：["傲嬌", "溫柔體貼", "反差萌", "冷靜", "害羞", "長髮", "可靠"]
- 每個詞語請保持簡潔（1 到 3 個詞）。
- **絕對不要**輸出任何額外的文字敘述、解釋或程式碼標記。

理想型描述：
{user_text}
"""
    raw = gemini_generate(prompt)
    time.sleep(GENAI_DELAY)

    # Robust JSON extraction
    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not match:
        raise RuntimeError(f"Gemini 回傳中找不到 JSON 陣列: {raw}")

    try:
        arr = json.loads(match.group(0))
    except:
        raise RuntimeError("Gemini 回傳 JSON 解析失敗")

    traits = [t.strip() for t in arr if isinstance(t, str) and t.strip()]
    if not traits:
        raise RuntimeError("提取不到任何有效特質。")
        
    return traits

def embed_text_ollama(text: str) -> np.ndarray:
    """Step 3: Creates embeddings via Ollama."""
    try:
        resp = ollama.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
        vec = np.array(resp["embedding"], dtype=float)
        return vec
    except Exception as e:
        raise RuntimeError(f"Ollama embedding error. Is Ollama server running? Error: {e}")

def match_user(user_vec: np.ndarray, char_embeddings: np.ndarray, k=5):
    """
    Step 4: Calculates similarity and applies Dynamic Min-Max Scaling 
    to map the entire score distribution to the [0, 100] range.
    """
    # 1. Calculate raw cosine similarity (range 0 to 1)
    # Using raw_sim for demonstration of dynamic scaling
    raw_sim = cosine_similarity(user_vec.reshape(1, -1), char_embeddings)[0]
    
    # 2. Dynamic Scaling Calculation
    min_sim = raw_sim.min()
    max_sim = raw_sim.max()
    
    # Check if all scores are identical (division by zero risk)
    if max_sim == min_sim:
        # If all are identical, everyone gets the perfect score of 100
        curved_scores = np.full_like(raw_sim, 100.0)
    else:
        # Apply Min-Max Scaling: maps [min_sim, max_sim] to [0, 100].
        # (Raw Score - Min) / (Max - Min) gives a 0-1 range. Multiply by 100.
        normalized_sim = (raw_sim - min_sim) / (max_sim - min_sim)
        curved_scores = normalized_sim * 100.0
    
    # 3. Find the top indices based on the new, stretched scores
    top_idx = np.argsort(-curved_scores)[:k]
    
    # 4. Return index and the curved score
    return [(i, float(curved_scores[i])) for i in top_idx]

def generate_final_report(user_text: str, traits: List[str], matches: List[Tuple[int, float, dict]]) -> str:
    """Step 5: Generates the final human-readable report via Gemini (Optimized for ideal match)."""
    match_display = [(c["name"], f"{s:.1f}") for (_, s, c) in matches]
    
    prompt = f"""
你是一個專業的二次元理想型匹配助理。
根據以下資訊，產生一份**自然語言的最終匹配報告**，幫助使用者找到他們理想型的**二次元化身**。

需要包含：
1. **最符合理想型**的角色（第 1 名），並解釋特質吻合的程度（引用 2–3 個 traits）
2. 再列出 2 個 runner-ups（第 2、3 名），簡短說明他們也接近理想型的原因
3. 風格自然、有趣、充滿 ACG 氛圍。

使用者理想型描述：
{user_text}

提取的理想型核心萌點：
{traits}

匹配結果（依序，已套用 Min-Max 縮放分數）：
{match_display}

請直接輸出自然語言報告。
"""
    text = gemini_generate(prompt)
    time.sleep(GENAI_DELAY)
    return text.strip()