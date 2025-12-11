import json
import numpy as np
import ollama
import os

# --- 1. 配置 (保持不變) ---
OLLAMA_EMBEDDING_MODEL = 'bge-m3' 
INPUT_DATA_FILENAME = 'character_database.json'
OUTPUT_VECTORS_FILENAME = 'character_embeddings_ollama.npy'
OUTPUT_DATA_WITH_ID_FILENAME = 'character_data_with_id.json'

def prepare_text_for_embedding(character_data):
    """
    將 JSON 數據轉換為模型可以理解的單一文本描述。
    """
    input_texts = []
    for item in character_data:
        name = item.get("name", "Unknown Character")
        # 處理 name 可能包含的冗餘信息
        cleaned_name = name.split('(')[0].split('#')[0].strip()
        
        traits = ", ".join(item.get("moe_traits", []))
        
        # 創建一個標準化的描述
        text = f"角色: {cleaned_name}。萌點描述: {traits}。"
        input_texts.append(text)
        
    return input_texts

def generate_embeddings_with_ollama(input_texts):
    """
    連接到本地 Ollama 服務並生成 embeddings。
    🚨 解決方案：將批量調用更改為循環單次調用。
    """
    print(f"1. 正在使用 Ollama 模型: {OLLAMA_EMBEDDING_MODEL} 生成向量...")
    
    # 初始化 Ollama 客戶端
    client = ollama.Client() 
    vectors = []
    
    try:
        total = len(input_texts)
        print(f"   總共需要生成 {total} 個向量...")
        
        # 循環處理每個文本
        for i, text in enumerate(input_texts):
            # 這是關鍵的修復：每次只傳遞一個 'string' 提示
            response = client.embeddings(
                model=OLLAMA_EMBEDDING_MODEL,
                prompt=text 
            )
            vectors.append(response['embedding'])
            
            # 打印進度以顯示程式正在運行
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"   進度: {i + 1}/{total} 完成。")

        # 將列表轉換為 NumPy 陣列
        return np.array(vectors)

    except Exception as e:
        print(f"   [!!!] 錯誤: 連接到或使用 Ollama 時發生問題: {e}")
        print("   請確保 Ollama 服務 (ollama serve) 正在運行，且模型已拉取 (ollama pull bge-m3)。")
        return None

def generate_and_save_embeddings():
    """主函數：加載數據、生成向量並保存結果。"""
    
    # --- 1. 加載數據 ---
    if not os.path.exists(INPUT_DATA_FILENAME):
        print(f"   [致命錯誤] 找不到輸入文件: {INPUT_DATA_FILENAME}")
        print("   請確保您的 JSON 文件與腳本在同一目錄下。")
        return
        
    # ... (加載和驗證數據的代碼保持不變)
    with open(INPUT_DATA_FILENAME, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"   [致命錯誤] 無法解析文件: {INPUT_DATA_FILENAME}。請檢查 JSON 格式是否正確。")
            return
            
    if not isinstance(data, list) or not data:
        print(f"   [致命錯誤] {INPUT_DATA_FILENAME} 文件為空或格式不正確 (應為列表)。")
        return

    print(f"   成功加載 {len(data)} 個角色數據。")
    
    # 2. 準備文本輸入
    input_texts = prepare_text_for_embedding(data)
    
    # 3. 生成 embeddings
    embeddings = generate_embeddings_with_ollama(input_texts)
    
    if embeddings is not None:
        print(f"2. 向量生成完成。總向量數: {embeddings.shape[0]}, 維度: {embeddings.shape[1]}")
        
        # 4. 保存 Embeddings 陣列 (.npy 文件)
        np.save(OUTPUT_VECTORS_FILENAME, embeddings)
        print(f"3. Embeddings 已成功保存到 {OUTPUT_VECTORS_FILENAME}")
        
        # 5. 保存帶有 ID 的原始數據 (.json 文件)
        data_with_id = []
        for i, item in enumerate(data):
            item['id'] = i 
            data_with_id.append(item)

        with open(OUTPUT_DATA_WITH_ID_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(data_with_id, f, ensure_ascii=False, indent=4)
            
        print(f"4. 帶有 ID 的數據已保存到 {OUTPUT_DATA_WITH_ID_FILENAME}")
        
        print("\n🎉 向量化步驟成功完成！")
        return embeddings

if __name__ == "__main__":
    generate_and_save_embeddings()