from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from detector import run_detection, load_model
from advisory import generate_advisory
import json
import ast  

# Load YOLO model
model_path = r"C:\\Users\\pc_37\\Desktop\\Final_Project\\runs\\detect\\cotton_leaf_disease_yolo11n3\\weights\\best.pt"
yolo_model = load_model(model_path)

# Define tools
tools = [
    Tool(
        name="DiseaseDetector",
        func=lambda img_path: run_detection(img_path, yolo_model),
        description="Detect disease, stage, and severity from a leaf image path"
    ),
    Tool(
        name="AdvisoryGenerator",
        func=lambda detections: (
            generate_advisory(
                ast.literal_eval(detections) if isinstance(detections, str) else detections
            )
        ),
        description="Generates advisories for detected cotton diseases."
    )
]

# Define the LLM with a system message guardrail
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a cotton leaf disease advisory assistant. "
                "You ONLY provide answers about cotton leaf disease detection, "
                "management, prevention, severity, and treatment. "
                "If the user asks something unrelated (like math, politics, general chit-chat), "
                "politely respond: 'I am a cotton leaf disease advisory assistant and cannot provide information on that topic.'"
            ),
        }
    ]
)

# Initialize agent
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)
