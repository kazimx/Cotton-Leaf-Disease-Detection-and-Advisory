import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import json
from weather import get_weather as get_weather_info  # ✅ import weather module

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_advisory_prompt(severity_df, location_hint="Pakistan"):
    rows, ctx = [], []

    # --- Normalize input ---
    if isinstance(severity_df, dict):
        severity_df = [severity_df]
    elif isinstance(severity_df, list):
        if not all(isinstance(r, dict) for r in severity_df):
            raise ValueError("List input must contain only dicts.")
    else:
        raise ValueError("Unsupported type for severity_df. Must be dict or list of dicts.")

    # --- Build advisory context ---
    for r in severity_df:
        name = str(r.get("disease", "unknown"))
        stage = r.get("stage") if r.get("stage") not in (None, "None", "") else "N/A"
        sev = float(r.get("severity_pct", 0.0))
        rows.append(f"- Disease: {name} | Stage: {stage} | Severity: {sev}%")
        ctx.append(f"{name}: adapt measures for severity {sev}%")

    table_block = "\n".join(rows) if rows else "- No disease detected."
    context_block = "\n".join(ctx) if ctx else "None"

    system = (
        "You are an agricultural advisory assistant for cotton farmers in Pakistan. "
        "Given detected diseases, stages, and severities, produce a concise, structured advisory. "
        "Rules:\n"
        "1. Start each advisory with a clear title: 'Advisory for <Disease Name> (Stage X, Severity: Y%)'.\n"
        "2. Always mention the disease name, stage, and severity again inside each section.\n"
        "3. Use 4 clear sections: PREVENTIVE MEASURES, IMMEDIATE TREATMENT OPTIONS, MONITORING FREQUENCY, CHEMICAL SAFETY GUIDANCE.\n"
        "4. Each section should have 2–4 bullet points, simple farmer-friendly language.\n"
        "5. If multiple diseases are detected, write separate advisories (one per disease).\n"
        "6. Avoid generic text, ground advice in disease + stage explicitly.\n"
        "7. If weather info is provided, give weather-specific recommendations (e.g., spraying safety, irrigation, humidity).\n"
    )

    user = (
        f"LOCATION: {location_hint}\n"
        f"DETECTIONS:\n{table_block}\n\n"
        "CONTEXT HINTS:\n"
        f"{context_block}\n\n"
        "Generate the advisory now."
    )
    return system, user


def generate_advisory(
    severity_df,
    location_hint="Pakistan",
    city=None,   # ✅ added city param
    model="gpt-4o-mini",
    temperature=0.4,
    max_tokens=800
):
    system, user = build_advisory_prompt(severity_df, location_hint)

    # ✅ Get weather context if city provided
    weather_context = ""
    if city:
        weather_context = get_weather_info(city)
        if weather_context:
            user += f"\n\nWeather conditions to consider: {weather_context}"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()
