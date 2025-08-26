🌱 Cotton Leaf Disease Detection & Advisory System
This project is an AI-powered decision support system for cotton farmers.
It detects cotton leaf diseases using a custom-trained YOLOv8 model, calculates disease severity, and generates actionable advisories with LLM integration (GPT-4o-mini).

It also integrates real-time weather data to give context-aware recommendations (e.g., avoiding spraying before rain). A Streamlit interface provides an easy-to-use interactive UI for farmers to upload images, visualize results, and chat with the advisory assistant.

🚀 Installation & Usage

1️⃣ Clone Repo or download the zip file
git clone https://github.com/<your-username>/cotton-leaf-disease.git
cd cotton-leaf-disease

2️⃣ Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate    # Windows
source .venv/bin/activate # Linux/Mac

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Streamlit App
streamlit run ui.py
