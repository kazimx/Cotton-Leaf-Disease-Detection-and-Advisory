# ui.py
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
from agent import agent as lc_agent 

# ======================
# ⚙️ Config & Model Load
# ======================
st.set_page_config(page_title="🌱 Cotton Leaf Disease Advisor", layout="wide")
MODEL_PATH = r"D:\best.pt"
yolo_model = load_model(MODEL_PATH)

# ======================
# 🎨 Custom CSS
# ======================
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 0rem; }
h1, h2, h3 { font-family: 'Segoe UI', sans-serif; font-weight: 600; }

/* Cards */
.advisory-box {
  background: #e8f5e9; padding: 18px; border-radius: 12px;
  border-left: 6px solid #2e7d32; margin-bottom: 20px;
}
.summary-box {
  background: #fff8e1; padding: 16px; border-radius: 12px;
  border-left: 6px solid #f9a825; margin-top: 10px;
}

/* Chat bubbles */
.chat-user {
  background: #d1e7ff; padding: 10px 14px; border-radius: 16px;
  margin: 6px 0; max-width: 80%; align-self: flex-end;
}
.chat-assistant {
  background: #f1f0f0; padding: 10px 14px; border-radius: 16px;
  margin: 6px 0; max-width: 80%; align-self: flex-start;
}

/* Tiny tag */
.small-tag {
  display: inline-block; padding: 3px 8px; border-radius: 999px;
  background: #eef2ff; font-size: 12px; margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# 🧠 Session State
# ======================
defaults = {
    "uploaded_img_path": None,
    "annotated_img_path": None,
    "detections": None,
    "advisory_markdown": None,   # detailed (4 headings) from tool
    "advisory_summary": None,    # final answer from agent
    "chat_history": [],
    "weather_data": None,
    "weather_advice": "",
    "city": "Multan",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================
# 🧩 Helpers
# ======================
def produce_annotated_image(image_path: str) -> str:
    tmpdir = Path(tempfile.mkdtemp(prefix="cotton_pred_"))
    results = yolo_model.predict(
        source=image_path, save=True, project=str(tmpdir),
        name="preds", exist_ok=True, verbose=False
    )
    save_dir = Path(results[0].save_dir)
    out_path = save_dir / Path(image_path).name
    if not out_path.exists():
        candidates = list(save_dir.glob("*"))
        if candidates:
            return str(candidates[0])
    return str(out_path)

def system_prompt_with_context():
    det = st.session_state.detections or []
    det_json = json.dumps(det, ensure_ascii=False)
    weather = st.session_state.weather_data
    if weather and "error" not in weather:
        weather_text = (
            f"{weather['location']} | {weather['temp']}°C | "
            f"{weather['humidity']}% humidity | {weather['wind_speed']} m/s wind | "
            f"{weather['condition']}"
        )
    else:
        weather_text = "No weather data."
    return f"""
You are a Cotton Leaf Disease Advisory Assistant for farmers in Pakistan.

SCOPE:
- Only answer about cotton plants, cotton leaf diseases (curl, leaf enation, sooty), severity, prevention,
  monitoring, and treatments (bio first; chemicals last with safety).
- If the question is unrelated, reply: "I’m a cotton leaf disease advisory assistant, so I can’t help with that topic."

CONTEXT (latest detections):
{det_json}

WEATHER:
{weather_text}

GUIDELINES:
- Be concise, practical, and step-by-step.
- If the farmer asks for a "next 5 days" plan, produce a 5-day schedule grounded in the detections + weather.
- Never invent detections beyond the context above.
""".strip()

# ======================
# 🖼️ Header
# ======================
st.title("🌿 Cotton Leaf Disease Detection & Advisory")
st.caption("YOLOv8 • Severity Calculator • Weather-aware LLM • Streamlit UI")

# ======================
# 🧭 Sidebar
# ======================
with st.sidebar:
    st.header("ℹ️ How it works")
    st.write("1) Upload a cotton leaf image\n2) Analyze to get detections & advisory\n3) Ask follow-ups in chat")
    use_agent = st.checkbox("Use LangChain Agent (ReAct)", value=True,
                            help="Let the agent call tools and return both the detailed advisory and summary.")
    if st.button("🔄 Reset Session"):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.subheader("🌦️ Local Weather")
    city = st.text_input("City", st.session_state["city"])
    st.session_state["city"] = city

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
            # Always detect + annotate for the UI
            detections = run_detection(st.session_state.uploaded_img_path, yolo_model)
            annotated_path = produce_annotated_image(st.session_state.uploaded_img_path)

            # Weather text (optional context for prompt)
            weather = st.session_state.get("weather_data")
            if weather and "error" not in weather:
                weather_text = (
                    f"{weather['location']} | {weather['temp']}°C | "
                    f"{weather['humidity']}% RH | {weather['wind_speed']} m/s wind | {weather['condition']}"
                )
            else:
                weather_text = ""

            # Advisory path
            if use_agent:
                prompt = (
                    f"Analyze this cotton leaf image: {st.session_state.uploaded_img_path}. "
                    f"Weather (if relevant for spraying): {weather_text}"
                )

                # 👇 Use invoke so we can read intermediate tool outputs
                result = lc_agent.invoke({"input": prompt})

                # Final summary from the agent
                final_answer = result.get("output", "")

                # Grab the detailed 4-section advisory from AdvisoryGenerator observation
                detailed_md = None
                for action, obs in result.get("intermediate_steps", []):
                    if getattr(action, "tool", "") == "AdvisoryGenerator":
                        detailed_md = obs  # this is the markdown you want
                        break

                # Fallbacks
                if not detailed_md:
                    detailed_md = final_answer

                st.session_state.advisory_markdown = detailed_md
                st.session_state.advisory_summary = final_answer

            else:
                # Direct (no agent)
                detailed_md = generate_advisory(detections, location_hint="Pakistan")
                st.session_state.advisory_markdown = detailed_md
                st.session_state.advisory_summary = None  # not applicable

            # Save common state
            st.session_state.detections = detections
            st.session_state.annotated_img_path = annotated_path

        st.success("✅ Analysis complete")

# ======================
# 🖼️ Images
# ======================
if st.session_state.uploaded_img_path or st.session_state.annotated_img_path:
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.session_state.uploaded_img_path:
            st.image(st.session_state.uploaded_img_path, caption="📷 Uploaded Leaf", width=320)
    with col2:
        if st.session_state.annotated_img_path and os.path.exists(st.session_state.annotated_img_path):
            st.image(st.session_state.annotated_img_path, caption="📦 Detected Diseases (YOLO)", width=320)

# ======================
# 📋 Advisory Result
# ======================
if st.session_state.advisory_markdown:
    st.markdown("### 📋 Advisory Result <span class='small-tag'>Detailed</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='advisory-box'>{st.session_state.advisory_markdown}</div>", unsafe_allow_html=True)

if st.session_state.advisory_summary:
    st.markdown("### <span class='small-tag'>Summary</span>", unsafe_allow_html=True)
    #st.markdown(f"<div class='summary-box'>{st.session_state.advisory_summary}</div>", unsafe_allow_html=True)

# ======================
# 💬 Chat Assistant
# ======================
st.markdown("### 💬 Farmer Chat Assistant")

for m in st.session_state.chat_history:
    if m["role"] == "user":
        st.markdown(f"<div class='chat-user'>{m['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-assistant'>{m['content']}</div>", unsafe_allow_html=True)

user_msg = st.chat_input("Ask about severity, 5-day plan, spraying, or monitoring…")
if user_msg:
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    st.markdown(f"<div class='chat-user'>{user_msg}</div>", unsafe_allow_html=True)

    sys_msg = SystemMessage(content=system_prompt_with_context())
    history_msgs = []
    for m in st.session_state.chat_history:
        history_msgs.append(HumanMessage(content=m["content"]) if m["role"] == "user"
                           else AIMessage(content=m["content"]))

    chat_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    reply = chat_llm.invoke([sys_msg, *history_msgs]).content

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.markdown(f"<div class='chat-assistant'>{reply}</div>", unsafe_allow_html=True)

# ======================
# ℹ️ Help
# ======================
with st.expander("What can I ask?"):
    st.markdown("""
- “Interpret the severity and what it means for yield.”
- “Give me a 5-day action plan to stop spread.”
- “What bio options first? If that fails, which chemical and precautions?”
- “How often should I monitor and what signs to watch?”
- “What should I do if severity goes above 60%?”
""")
