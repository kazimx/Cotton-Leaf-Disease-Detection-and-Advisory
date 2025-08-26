# agent.py
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from detector import run_detection, load_model
from advisory import generate_advisory
import ast

# ------------- Config -------------
MODEL_PATH = r"D:\best.pt"

# ------------- Load YOLO once -------------
yolo_model = load_model(MODEL_PATH)

# ------------- Tools -------------
def _advisory_wrapper(detections_or_payload):
    """
    Accepts either:
      - list[dict] detections
      - str literal of list[dict]
      - dict {"detections": [...], "weather_text": "..."}  (weather_text optional)
    """
    if isinstance(detections_or_payload, str):
        detections_or_payload = ast.literal_eval(detections_or_payload)

    if isinstance(detections_or_payload, dict):
        detections = detections_or_payload.get("detections", [])
    else:
        detections = detections_or_payload

    return generate_advisory(detections, location_hint="Pakistan")

tools = [
    Tool(
        name="DiseaseDetector",
        func=lambda img_path: run_detection(img_path, yolo_model),
        description="Detect disease, stage, and severity from a cotton leaf image path."
    ),
    Tool(
        name="AdvisoryGenerator",
        func=_advisory_wrapper,
        description="Generate a structured advisory (prevention, immediate actions, monitoring, safety) from detections."
    ),
]

# ------------- LLM & Agent -------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    agent_kwargs={
        "system_message": (
            "You are a cotton leaf disease advisory assistant. "
            "Only answer questions about cotton leaf diseases (curl, leaf enation, sooty), "
            "their detection, stages, severity, prevention, monitoring, and treatments "
            "(bio options first, chemicals last with safety). "
            "If asked something unrelated, reply: "
            "'I am a cotton leaf disease advisory assistant and cannot provide information on that topic.'"
        )
    },
    return_intermediate_steps=True,  
)
