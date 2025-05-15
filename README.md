# Travel Agent Project

## 🧭 專案簡介 
[專案連結](https://github.com/SogoChang/Travel_Agent_Project.git)

本專案開發了一個旅遊代理人 AI，能夠根據使用者的偏好與需求，推薦適合的景點。透過整合景點資料與使用者輸入，系統能夠提供個性化的旅遊建議，提升旅遊體驗。

## 📂 專案結構

* `main.py`：主程式，負責整合各模組並執行旅遊規劃流程。
* `scenic_spots/`：包含景點資料的目錄，供系統進行推薦分析。
* `scenic_spot_index/`：景點索引資料，用於快速檢索與匹配。
* `scenic_spot_selection_index/`：景點選擇索引，協助篩選適合的景點。
* `scenic_spot_selection_rag.py`：景點選擇模組，實現推薦演算法。
* `api_manual_index/`：API 使用手冊索引，提供系統操作說明。
* `instruction_file/`：操作指引文件，協助使用者了解系統功能。
* `travel_spot_selection_guide.txt`：旅遊景點選擇指南，提供使用者參考。
* `.gitignore`：Git 忽略設定檔。

## 🧠 功能特色

* **個性化推薦**：根據使用者輸入的偏好與需求，推薦適合的旅遊景點。
* **景點資料整合**：整合多個景點資料來源，提供豐富的旅遊資訊。

## 📦 環境需求

### 套件依賴

本專案需要安裝以下套件：

```bash
pip install langchain-community langchain-huggingface langchain google-generativeai requests faiss-cpu sentence-transformers
```

### 必要套件說明
* langchain-community：用於向量存儲和檢索
* langchain-huggingface：用於使用HuggingFace嵌入模型
* langchain：核心LangChain函式庫
* google.generativeai：Google的Gemini API
* requests：用於HTTP請求
* faiss-cpu (或 faiss-gpu)：FAISS向量檢索庫
* sentence-transformers：用於文本嵌入

### 環境變數設定

使用前需設置以下環境變數：
* `GOOGLE_API_KEY`：用於訪問Gemini模型
* `TRIPADVISOR_API_KEY`：用於訪問TripAdvisor API
* `GOOGLE_MAPS_API_KEY`：用於訪問Google Maps Places API

## 🚀 使用方式

1. 確保已安裝 Python 3.x 環境。
2. 克隆本專案至本地：

   ```bash
   git clone https://github.com/SogoChang/Travel_Agent_Project.git
   ```
3. 安裝所需套件：

   ```bash
   pip install -r requirements.txt
   ```
   或手動安裝所有必要套件。

4. 設置必要的環境變數：

   ```bash
   # Windows
   set GOOGLE_API_KEY=your_key_here
   set TRIPADVISOR_API_KEY=your_key_here
   set GOOGLE_MAPS_API_KEY=your_key_here
   
   # Linux/macOS
   export GOOGLE_API_KEY=your_key_here
   export TRIPADVISOR_API_KEY=your_key_here
   export GOOGLE_MAPS_API_KEY=your_key_here
   ```

5. 確保創建了必要的資料夾和文件：
   - `scenic_spots/` 資料夾用於存儲景點資料
   - `travel_spot_selection_guide.txt` 包含旅行景點選擇指南

6. 進入專案目錄並執行主程式：

   ```bash
   cd Travel_Agent_Project
   python main.py
   ```
7. 依照提示輸入您的旅遊偏好與需求，系統將提供相應的景點推薦與行程安排。

## 📌 注意事項

* 如需更新景點資料，請同步更新相關索引檔案。
* 首次運行時，系統將創建必要的索引文件，這可能需要一些時間。


如需進一步協助或有任何建議，歡迎提出 Issue 或聯絡專案維護者。