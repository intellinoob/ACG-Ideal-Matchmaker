# ACG Ideal Matchmaker

## Overview

The ACG Ideal Matchmaker is a semantic recommendation system designed to match users with ideal fictional characters from Anime, Comics, and Games (ACG) culture. Leveraging large language models (LLMs) for trait extraction, vector embeddings for similarity computation, and generative AI for personalized reports, this application provides nuanced, explainable matches based on user-described preferences (e.g., "Moe" attributes like *tsundere* or *yandere*).

The system features:
- An offline ETL pipeline to crawl and vectorize character data from Moegirl Wiki.
- A retrieval-augmented generation (RAG) engine using BGE-M3 embeddings and Google Gemini.
- An interactive Streamlit interface for query submission and result visualization.

This README provides instructions for setup, preprocessing, and usage. The project is implemented in Python 3.10+ and assumes a local development environment.

## Prerequisites

Before proceeding, ensure the following are installed:
- **Python 3.10+**: Download from [python.org](https://www.python.org/downloads/).
- **Ollama**: For local embedding generation. Install from [ollama.com](https://ollama.com/download) and pull the BGE-M3 model via `ollama pull bge-m3`.
- **Google Gemini API Key**: Obtain a free key from [Google AI Studio](https://aistudio.google.com/app/apikey). Set it as an environment variable: `export GEMINI_API_KEY="your-api-key-here"`.
- **Chrome Browser**: Required for Selenium crawling (headless mode optional).
- **Virtual Environment Tools**: Use `venv` (built-in) or `conda` for dependency isolation.

## Installation

1. **Clone or Download the Repository**:
   ```
   git clone <your-repo-url>  # Or download the ZIP
   cd acg-ideal-matchmaker
   ```

2. **Create a Virtual Environment**:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Create a `requirements.txt` file with the following content (if not provided):
   ```
   streamlit>=1.28.0
   selenium>=4.0.0
   webdriver-manager>=4.0.0
   google-generativeai>=0.3.0
   ollama>=0.1.0
   scikit-learn>=1.3.0
   numpy>=1.24.0
   pandas>=2.0.0
   plotly>=5.15.0
   ```
   Then install:
   ```
   pip install -r requirements.txt
   ```

## Setup and Preprocessing

The application requires an initial offline preprocessing step to build the character dataset and embeddings. This must be run once (or after updating the dataset).

### Step 1: Crawl Character Data
Run the crawler to extract "Moe Points" from Moegirl Wiki:
```
python crawl.py
```
- **Expected Output**: A JSON file `character_database.json` with ~50 characters (e.g., {"name": "江戶川柯南", "moe_traits": ["正太", "偵探"], "trait_count": 3}).
- **Duration**: 15–20 minutes (includes rate limiting).
- **Notes**: Ensure ChromeDriver is installed via webdriver-manager. If issues arise (e.g., timeouts), check network connectivity or reduce the `CHARACTER_LIST` in `crawl.py`.

### Step 2: Generate Embeddings
Vectorize the dataset using Ollama:
```
python generate_embeddings.py
```
- **Expected Output**: Files `character_embeddings_ollama.npy` (vectors) and `character_data_with_id.json` (metadata).
- **Duration**: <5 minutes for 50 characters.
- **Notes**: Verify Ollama is running (`ollama serve`). The script uses single-prompt embedding to avoid batch errors.

## Usage

### Running the Application
1. **Start the Streamlit Server**:
   ```
   streamlit run app.py
   ```
   - The app opens in your default browser at `http://localhost:8501`.

2. **Interact with the Interface**:
   - **Sidebar Input**: Enter a free-form description of your ideal character (e.g., "A child-like detective with a strong sense of justice and sharp intellect").
   - **Submit Query**: Click "🚀 開始匹配！ (尋找你的 TA)".
   - **View Results**:
     - **Matching Report**: A Gemini-generated narrative explaining the top match and runners-up.
     - **Bar Chart**: Plotly visualization of top-5 scaled similarity scores (0–100%).
     - **Results Table**: Pandas dataframe with ranks, names, scores (as progress bars), and top traits.
     - **Extracted Traits**: List of 8–12 core "Moe points" derived from your query.

3. **Example Queries**:
   - Case 1 (Intellectual Detective): "A clever kid who looks innocent but solves mysteries like a pro."
   - Case 2 (Compassionate Hard Worker): "A dedicated big sister type who's kind, strong, and always there to help."

- **Response Time**: 3–5 seconds per query (due to LLM calls).
- **Customization**: Adjust `TOP_K` in `pipeline.py` for more/fewer results.

### Sample Output Structure
For a query, expect:
- Top Match: 江戶川柯南 (100%) – Traits: 正太, 正義感, 偵探.
- Visualization: Horizontal bar chart with hover details.
- Report: "Your ideal detective vibe perfectly aligns with Edogawa Conan's child-like genius..."

## Troubleshooting

- **Ollama Errors**: Ensure `ollama serve` is active and BGE-M3 is pulled. Restart if embeddings fail.
- **Gemini API Issues**: Verify `GEMINI_API_KEY` is set; check quotas at [Google AI Studio](https://aistudio.google.com).
- **Selenium Failures**: Update ChromeDriver via `pip install --upgrade webdriver-manager`. For headless mode, uncomment in `crawl.py`.
- **No Data Loaded**: Rerun preprocessing; check file paths in `pipeline.py`.
- **High Latency**: Use a faster machine for Ollama; consider cloud deployment via Streamlit Cloud.
- **Empty Traits**: Wiki pages may vary—manually inspect `character_database.json` and add fallbacks.

For bugs, review console logs or raise an issue on the repository.

## Contributing

This is a student project for CISC3014. Contributions are welcome via pull requests—focus on robustness (e.g., more characters, multilingual support). Please adhere to fair-use scraping and cite sources.

## License

MIT License. See [LICENSE](LICENSE) for details. Data from Moegirl Wiki used under fair use for educational purposes.

---

*Project developed by LAM IOK HOI (DC326834), University of Macau, December 2025.*