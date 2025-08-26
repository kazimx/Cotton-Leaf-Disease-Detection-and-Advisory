import os
import json
import tempfile
from pathlib import Path
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from detector import load_model, run_detection
from advisory import generate_advisory
from weather import get_weather, format_weather_advice
# ======================
# ⚙️ Config & Model Load
# ======================
st.set_page_config(page_title="🌱 Cotton Leaf Disease Advisor", layout="wide")
MODEL_PATH = r"C:\\Users\\pc_37\\Desktop\\Final_Project\\runs\\detect\\cotton_leaf_disease_yolo11n3\\weights\\best.pt"
yolo_model = load_model(MODEL_PATH)

# ======================
# 🎨 Custom CSS Styling
# ======================
st.markdown("""
    <style>
    /* Background */
    body { background-color: #f9fafb; }

    /* Section headers */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }

    /* Advisory card */
    .advisory-box {
        background: #e8f5e9;
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #2e7d32;
        margin-bottom: 20px;
    }

    /* Chat bubbles */
    .chat-user {
        background: #d1e7ff;
        padding: 10px 14px;
        border-radius: 16px;
        margin: 6px 0;
        max-width: 80%;
        align-self: flex-end;
    }
    .chat-assistant {
        background: #f1f0f0;
        padding: 10px 14px;
        border-radius: 16px;
        margin: 6px 0;
        max-width: 80%;
        align-self: flex-start;
    }
    </style>
""", unsafe_allow_html=True)

# ======================
# 🧠 Session State Setup
# ======================
for key, default in [
    ("uploaded_img_path", None),
    ("annotated_img_path", None),
    ("detections", None),
    ("advisory_markdown", None),
    ("chat_history", []),
    ("weather_data", None),
    ("weather_advice", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ======================
# 🎯 Helper: save annotated image
# ======================
def produce_annotated_image(image_path: str) -> str:
    tmpdir = Path(tempfile.mkdtemp(prefix="cotton_pred_"))
    results = yolo_model.predict(
        source=image_path,
        save=True,
        project=str(tmpdir),
        name="preds",
        exist_ok=True,
        verbose=False
    )
    save_dir = Path(results[0].save_dir)
    out_path = save_dir / Path(image_path).name
    if not out_path.exists():
        candidates = list(save_dir.glob("*"))
        if candidates:
            return str(candidates[0])
    return str(out_path)

# ======================
# 🧩 System Prompt
# ======================
def system_prompt_with_context():
    det = st.session_state.detections or []
    det_json = json.dumps(det, ensure_ascii=False)
    weather_info = st.session_state.get("weather_info", None)

    weather_text = "No weather data."
    if weather_info:
        weather_text = (
            f"📍 {weather_info['city']}, {weather_info['country']} | "
            f"🌡️ {weather_info['temp']}°C | 💧 {weather_info['humidity']}% humidity | "
            f"🌬️ {weather_info['wind_speed']} m/s wind | ☁️ {weather_info['condition']}"
        )


    return f"""
You are a Cotton Leaf Disease Advisory Assistant for farmers in Pakistan. 
Your job is to give farmers advisory and chat answers based on:
- Disease detections
- Current weather conditions
- ONLY answer questions about cotton leaf diseases, severity, monitoring, prevention, and treatments.  
- If unrelated, say: *"I’m a cotton leaf disease advisory assistant, so I can’t help with that topic."*  

CONTEXT (latest detection):  
{det_json}

WEATHER CONTEXT:  
{weather_text}
💡 Always consider temperature, humidity, wind, and condition when advising spraying.
Refuse questions unrelated to cotton disease management.
Be simple, practical, and step-by-step.
""".strip()

# ======================
# 🖼️ UI — Header
# ======================
st.title("🌿 Cotton Leaf Disease Detection & Advisory")
st.divider()


# Sidebar
with st.sidebar:
    st.header("ℹ️ Guide")
    st.write("➡️ Upload a cotton leaf image\n➡️ Get detection results & advisory\n\n➡️ Ask follow-up questions in chat")
    if st.button("🔄 Reset Session"):
        st.session_state.clear()
        st.rerun()
        
# Show Weather        
with st.sidebar:
    st.subheader("🌦️ Local Weather")
    city = st.text_input("Enter your city", "Multan")
    
    if st.button("Get Weather"):
        weather = get_weather(city=city, country="PK")
        st.session_state.weather_data = weather
        if "error" in weather:
            st.error(weather["error"])
        else:
            st.success(
                f"📍 {weather['location']}\n"
                f"🌡️ {weather['temp']}°C\n"
                f"💧 {weather['humidity']}% humidity\n"
                f"🌬️ {weather['wind_speed']} m/s wind\n"
                f"☁️ {weather['condition']}"
            )
            st.session_state.weather_advice = format_weather_advice(weather)

            
# ======================
# 📤 Upload
# ======================
uploaded_file = st.file_uploader("📤 Upload a cotton leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    temp_dir = Path(tempfile.mkdtemp(prefix="cotton_img_"))
    img_path = str(temp_dir / uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.uploaded_img_path = img_path
    st.success("✅ Image uploaded. Click **Analyze Image** to proceed.")

# ======================
# 🔍 Analyze
# ======================
if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
    if not st.session_state.uploaded_img_path:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Analyzing image…"):
            detections = run_detection(st.session_state.uploaded_img_path, yolo_model)
            annotated_path = produce_annotated_image(st.session_state.uploaded_img_path)
            weather_context = st.session_state.weather_advice if st.session_state.weather_advice else ""
            advisory_md = generate_advisory(
                                            detections,
                                            location_hint="Pakistan",
                                            city=st.session_state.get("city", "Multan") 
                                        )

            st.session_state.detections = detections
            st.session_state.annotated_img_path = annotated_path
            st.session_state.advisory_markdown = advisory_md
        st.success("✅ Analysis complete")
     
# ======================
# 🖼️ Images
# ======================
if st.session_state.uploaded_img_path or st.session_state.annotated_img_path:
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.session_state.uploaded_img_path:
            st.image(st.session_state.uploaded_img_path, caption="📷 Uploaded Leaf", width=300)
    with col2:
        if st.session_state.annotated_img_path and os.path.exists(st.session_state.annotated_img_path):
            st.image(st.session_state.annotated_img_path, caption="📦 Detected Diseases", width=300)

# ======================
# 📋 Advisory Result
# ======================
if st.session_state.advisory_markdown:
    st.markdown("### 📋 Advisory Result")
    st.markdown(f"<div class='advisory-box'>{st.session_state.advisory_markdown}</div>", unsafe_allow_html=True)

# ======================
# 💬 Chat Assistant
# ======================
st.markdown("### 💬 Farmer Chat Assistant")

for m in st.session_state.chat_history:
    if m["role"] == "user":
        st.markdown(f"<div class='chat-user'>{m['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-assistant'>{m['content']}</div>", unsafe_allow_html=True)

user_msg = st.chat_input("Ask about severity, treatment, or monitoring...")
if user_msg:
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    st.markdown(f"<div class='chat-user'>{user_msg}</div>", unsafe_allow_html=True)

    sys_msg = SystemMessage(content=system_prompt_with_context())
    history_msgs = []
    for m in st.session_state.chat_history:
        if m["role"] == "user":
            history_msgs.append(HumanMessage(content=m["content"]))
        else:
            history_msgs.append(AIMessage(content=m["content"]))

    chat_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    reply = chat_llm.invoke([sys_msg, *history_msgs]).content

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.markdown(f"<div class='chat-assistant'>{reply}</div>", unsafe_allow_html=True)
