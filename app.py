import gradio as gr
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 本地輕量資料庫初始化 (SQLite)
# ==========================================
DB_NAME = "rnd_experiment_core.db"

def init_db():
    """建立專案專用的資料庫表單"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # 儲存使用者自定義的實驗範本 (參數欄位定義)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            template_name TEXT UNIQUE,
            factors TEXT,
            responses TEXT
        )
    ''')
    # 儲存每次實驗填入的實際數據
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiment_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            template_name TEXT,
            data_json TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 2. 後端核心邏輯運算
# ==========================================
def save_new_template(template_name, factors_str, responses_str):
    """建立新的實驗種類範本"""
    if not template_name.strip():
        return "❌ 錯誤：請輸入實驗範本名稱！", gr.update()
    
    factors = [f.strip() for f in factors_str.split(",") if f.strip()]
    responses = [r.strip() for r in responses_str.split(",") if r.strip()]
    
    if not factors or not responses:
        return "❌ 錯誤：控制參數與結果指標皆至少需要填寫一項！", gr.update()
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO templates (template_name, factors, responses) VALUES (?, ?, ?)",
            (template_name, ",".join(factors), ",".join(responses))
        )
        conn.commit()
        status = f"✅ 實驗範本【{template_name}】建立成功！\n🔹 控制參數：{factors}\n🔹 結果指標：{responses}"
    except Exception as e:
        status = f"❌ 儲存失敗：{str(e)}"
    finally:
        conn.close()
        
    return status, gr.update(choices=get_all_templates())

def get_all_templates():
    """從小資料庫撈出所有已建立的範本清單"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT template_name FROM templates")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]

def generate_dynamic_table(template_name):
    """當選擇某個實驗種類時，動態渲染對應的 Dataframe 輸入表格"""
    if not template_name:
        return gr.update(value=pd.DataFrame(), visible=False), "請選擇實驗範本"
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT factors, responses FROM templates WHERE template_name=?", (template_name,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return gr.update(value=pd.DataFrame()), "找不到該範本"
        
    factors = row[0].split(",")
    responses = row[1].split(",")
    
    headers = ["實驗批次編號"] + factors + responses
    default_row = ["EXP-001"] + [0.0]*len(factors) + [0.0]*len(responses)
    df = pd.DataFrame([default_row], columns=headers)
    
    return gr.update(value=df, visible=True), f"📊 已載入【{template_name}】專用工作表。請直接在下方表格修改數據、或按 Tab 鍵新增列數。"

def commit_records_to_db(template_name, df):
    """將 UI 表格上的數據序列化存入資料庫"""
    if not template_name or df is None or df.empty:
        return "❌ 儲存失敗：請先確認已選擇範本且表格內有數據。"
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        data_json = df.to_json(orient='records')
        cursor.execute(
            "INSERT INTO experiment_records (template_name, data_json) VALUES (?, ?)",
            (template_name, data_json)
        )
        conn.commit()
        return f"✅ 成功！目前表格內的 {len(df)} 筆數據已同步存入本地 SQLite 資料庫。"
    except Exception as e:
        return f"❌ 資料庫寫入錯誤：{str(e)}"
    finally:
        conn.close()

def execute_similarity_check(template_name, current_df):
    """核心智慧功能：計算當前輸入參數與歷史數據的標準化歐氏距離，進行防錯比對與圖像化"""
    if not template_name or current_df is None or current_df.empty:
        return "⚠️ 請選擇範本並確保工作表中有數據。", None
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT factors FROM templates WHERE template_name=?", (template_name,))
    template_row = cursor.fetchone()
    
    if not template_row:
        conn.close()
        return "找不到範本設定", None
        
    factors = template_row[0].split(",")
    
    cursor.execute("SELECT data_json FROM experiment_records WHERE template_name=?", (template_name,))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "💡 提示：資料庫中尚無此實驗的歷史紀錄，無法進行相似度防錯比對。", None
        
    history_list = [pd.read_json(r[0]) for r in rows]
    history_master_df = pd.concat(history_list, ignore_index=True)
    
    for col in history_master_df.columns:
        if col != "實驗批次編號":
            history_master_df[col] = pd.to_numeric(history_master_df[col], errors='coerce').fillna(0.0)
            current_df[col] = pd.to_numeric(current_df[col], errors='coerce').fillna(0.0)
            
    latest_input_factors = current_df[factors].iloc[-1].values
    history_factors_matrix = history_master_df[factors].values
    
    stds = history_master_df[factors].std().values
    stds = np.where(stds == 0, 1.0, stds) 
    
    report = "### 🔍 歷史數據智慧防錯比對報告\n---\n"
    trigger_alert = False
    alert_details = ""
    
    for idx, hist_row in history_master_df.iterrows():
        hist_input = hist_row[factors].values
        distance = np.sqrt(np.sum(((latest_input_factors - hist_input) / stds) ** 2))
        
        if distance == 0:
            trigger_alert = True
            alert_details += f"🚨 **【完全重複】** 與歷史批次 `{hist_row['實驗批次編號']}` 的參數完全一致！\n"
        elif distance < 0.3:  
            trigger_alert = True
            alert_details += f"⚠️ **【高度相似】** 與歷史批次 `{hist_row['實驗批次編號']}` 極度接近 (標準距離: `{distance:.3f}`)\n"
            
    if trigger_alert:
        report += alert_details + "\n> 💡 **防錯建議**：新輸入的參數與上述歷史實驗雷同，請確認是否重複實驗，或可直接參考歷史結果。"
    else:
        report += "✨ **未發現相似紀錄**：這是一組全新的實驗參數組合，請安心進行實驗！\n"
        
    # ==========================================
    # 3. 數據圖像化 (Matplotlib)
    # ==========================================
    fig, ax = plt.subplots(figsize=(6, 4))
    x_param = factors[0] 
    y_result = history_master_df.columns[-1] 
    
    ax.scatter(history_master_df[x_param], history_master_df[y_result], color='#94a3b8', s=120, label='History Records', alpha=0.7)
    ax.scatter(latest_input_factors[0], current_df[y_result].iloc[-1], color='#ef4444', s=200, marker='*', label='Current Input Target')
    
    ax.set_title(f"Experiment Space Map: {x_param} vs {y_result}", fontweight='bold')
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_result)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    return report, fig

# ==========================================
# 4. Gradio 介面佈局
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald"), title="研發實驗數據智慧管理系統") as demo:
    gr.Markdown("# 💾 研發實驗數據智慧管理與相似度比對系統 (Codespaces 版)")
    
    with gr.Tabs():
        with gr.TabItem("📋 步驟一：定義實驗範本欄位"):
            with gr.Row():
                with gr.Column():
                    tpl_name = gr.Textbox(label="1. 實驗範本名稱", placeholder="例如：鍍膜製程實驗、陽極氧化測試")
                    tpl_factors = gr.Textbox(label="2. 控制參數種類 (請用英文逗號隔開)", value="溫度(℃), 壓力(MPa), 時間(min)")
                    tpl_responses = gr.Textbox(label="3. 實驗結果指標 (請用英文逗號隔開)", value="膜厚(nm), 硬度(HV)")
                    btn_create = gr.Button("💾 建立並儲存此範本欄位", variant="primary")
                with gr.Column():
                    log_output = gr.Textbox(label="系統運行狀態日誌", interactive=False)
                    
        with gr.TabItem("🧪 步驟二：實驗記錄與智慧防錯比對"):
            with gr.Row():
                tpl_selector = gr.Dropdown(choices=get_all_templates(), label="🔍 請選擇已建立的實驗範本")
            
            hint_markdown = gr.Markdown("💡 請先選擇上方選單以載入對應的數據工作表。")
            data_table = gr.Dataframe(interactive=True, label="動態實驗工作表 (可雙擊方格修改數據)")
            
            with gr.Row():
                btn_save = gr.Button("💾 同步儲存當前表格至資料庫", variant="primary")
                btn_compare = gr.Button("🔍 啟動歷史相似度防錯比對與圖像化", variant="secondary")
                
            with gr.Row():
                with gr.Column(scale=1):
                    md_analysis_report = gr.Markdown("💡 防錯比對報告將即時顯示於此...")
                with gr.Column(scale=1):
                    chart_output = gr.Plot(label="實驗數據分布圖")

    # ==========================================
    # 5. UI 事件驅動綁定
    # ==========================================
    btn_create.click(save_new_template, inputs=[tpl_name, tpl_factors, tpl_responses], outputs=[log_output, tpl_selector])
    tpl_selector.change(generate_dynamic_table, inputs=[tpl_selector], outputs=[data_table, hint_markdown])
    btn_save.click(commit_records_to_db, inputs=[tpl_selector, data_table], outputs=[hint_markdown])
    btn_compare.click(execute_similarity_check, inputs=[tpl_selector, data_table], outputs=[md_analysis_report, chart_output])

if __name__ == "__main__":
    # 🌟 設定 share=True，啟動時會自動生成外部網頁連結
    demo.launch(share=True)
