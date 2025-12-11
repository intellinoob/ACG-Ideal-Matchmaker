import json
import time
import random
import urllib.parse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# --- 導入 WebDriver Manager ---
try:
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service as ChromeService
    # 設置 ChromeService 以便在後續使用
    SERVICE = ChromeService(ChromeDriverManager().install())
except ImportError:
    print("Warning: 'webdriver-manager' not installed. Falling back to PATH.")
    SERVICE = None 


# --- 配置與數據 ---
BASE_WIKI_URL = "https://zh.moegirl.org.cn"
# 確保字符列表已是繁體
CHARACTER_LIST = [
    "五條悟", "竈門炭治郎", "雷姆(Re:从零开始的异世界生活)", "黑川茜", "芙莉蓮", "阿尼亚·福杰", "猫猫(药师少女的独语)#", 
    "有马加奈", "艾莉莎·米哈伊罗芙娜·九条", "电次(电锯人)#", "早川秋", "魯迪烏斯", "菜月昴", "蒙奇·D·路飞", 
    "漩渦鳴人", "孙悟空(龙珠)#", "江戶川柯南", "坂田銀時", "利威尔·阿克曼", "魯路修·蘭佩洛基", 
    "阿尔托莉雅·潘德拉贡", "綾波麗", "赤井秀一", "毛利蘭", "赫蘿", "白銀御行", "四宮輝夜", 
    "雪之下雪乃", "加藤惠(路人女主的养成方法)#", "鹿野千夏", "司波深雪", "千石撫子", "椎名真白", "夏目貴志", 
    "月野兔", "澤村·史賓瑟·英梨梨", "湊阿庫婭", "金木研", "木之本櫻", "宇智波佐助", 
    "兵藤一誠", "阿良良木曆", "三笠·阿克曼", "約兒·佛傑", "千反田愛瑠", 
    "和泉紗霧", "桐谷和人", "亞絲娜", "立華奏", "菲倫"
]

# --- 穩健的 XPath 選擇器列表 ---
MOE_POINTS_XPATHS = [
    # 1. Flexbox/新模板結構 (您提供的結構)
    # 查找包含 '萌點' 或 '萌点' 的 span 標籤，然後選取其兄弟 div 
    "//span[contains(., '萌點') or contains(., '萌点')]/parent::div/following-sibling::div[1]",
    # 2. 舊表格結構 (作為備用 Fallback)
    "//td[contains(., '萌點') or contains(., '萌点')]/../td[2]",
    # 3. 備用 Flexbox 結構 (更通用，查找緊隨其後的第一個兄弟元素)
    "//span[contains(., '萌點') or contains(., '萌点')]/following-sibling::*[1]",
]

def clean_moe_points(raw_text):
    """
    對提取的文本進行後處理和清洗，以生成乾淨的萌點列表。
    """
    if not raw_text:
        return []
        
    # 移除腳註標記，例如 "[3]" 或 " [註]"
    import re
    cleaned_text = re.sub(r'\[.*?\]|\s*\[.*?\]|\s*\(.*?\)|(\(\w+\))|\s*(\.\.\.|\(|\))\s*|\s*(\d+)\s*$', '', raw_text)
    
    # 萌點通常以中文逗號(、)、英文逗號(,) 或換行符(\n) 分隔
    # 將所有分隔符統一替換為一個標準分隔符 (例如: |)，然後分割
    cleaned_text = cleaned_text.replace('、', '|').replace(',', '|').replace('\n', '|')
    
    # 分割並過濾空字符串
    traits = [t.strip() for t in cleaned_text.split('|') if t.strip()]
    
    return traits

def scrape_moe_points(character_name):
    """構造 URL 並嘗試多個 XPath 選擇器提取 '萌點'。"""
    url = f"{BASE_WIKI_URL}/zh-hk/{urllib.parse.quote(character_name)}"
    print(f"\n-> 請求中: {character_name} ({url})")
    
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")

    driver = None
    try:
        driver = webdriver.Chrome(service=SERVICE, options=options) if SERVICE else webdriver.Chrome(options=options)
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        
        for i, xpath in enumerate(MOE_POINTS_XPATHS):
            try:
                text = wait.until(EC.presence_of_element_located((By.XPATH, xpath))).text
                print(f"   [+] 成功! 使用 XPath #{i+1} 提取萌點。")
                return text
            except TimeoutException:
                continue
        
        print(f"   [!] 錯誤: {character_name} 超時，所有 {len(MOE_POINTS_XPATHS)} 個 XPath 均未匹配成功。")
        return ""
        
    except WebDriverException as e:
        print(f"   [!] WebDriver 運行錯誤: {e}")
        raise
    except Exception as e:
        print(f"   [!] 發生未知錯誤: {e}")
        return ""
    finally:
        if driver:
            driver.quit()

def run_integrated_crawler():
    """主執行函數：迭代列表，執行爬取，並輸出 JSON"""
    database = []
    output_filename = 'character_database.json'
    
    print("--- Selenium 萌娘百科爬蟲開始 (多 XPath 模式) ---")
    
    for character_name in CHARACTER_LIST:
        try:
            moe_points_raw = scrape_moe_points(character_name)
            
            # --- 穩健性：後處理清洗 ---
            traits_list = clean_moe_points(moe_points_raw)
            
            data_entry = {
                "name": character_name,
                "moe_traits": traits_list,
                "trait_count": len(traits_list)
            }
            database.append(data_entry)
            
        except Exception as e:
            # 捕獲所有異常，確保進度保存
            print(f"\n[致命錯誤] 程式停止。已保存數據。")
            break 
            
        # 設置延遲 (Rate Limiting)
        sleep_time = random.uniform(2.0, 4.0)
        time.sleep(sleep_time) 

    # 寫入 JSON 文件
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(database, f, ensure_ascii=False, indent=4)
        
    print("\n==============================")
    print(f"🎉 爬取結束。共保存 {len(database)} 個角色的數據。")
    print(f"數據已儲存至 {output_filename}")
    print("==============================")


if __name__ == "__main__":
    run_integrated_crawler()