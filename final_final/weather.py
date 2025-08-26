# weather.py
import os, requests
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API")

def get_weather(city="Multan", country="PK"):
    """Fetch current weather for a given city."""
    if not API_KEY:
        return {"error": "Missing OpenWeather API key in .env file"}

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city},{country}&appid=c9f7395f662da830b716e1e6b55fb91b&units=metric"
    
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if resp.status_code != 200:
            return {"error": data.get("message", "Weather API failed")}

        return {
            "location": f"{data['name']}, {data['sys']['country']}",
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "condition": data["weather"][0]["description"].capitalize(),
            "wind_speed": data["wind"]["speed"],
        }
    except Exception as e:
        return {"error": str(e)}

def format_weather_advice(weather: dict) -> str:
    """Turn raw weather data into farmer-friendly advice."""
    if not weather or "error" in weather:
        return ""

    advice = []
    temp, hum, cond = weather["temp"], weather["humidity"], weather["condition"].lower()

    if "rain" in cond:
        advice.append("🌧️ Rain expected → avoid spraying chemicals today.")
    if hum > 70:
        advice.append("💧 High humidity → higher fungal risk, monitor closely.")
    if temp > 38:
        advice.append("🔥 Very hot → avoid spraying midday, prefer morning/evening.")
    if temp < 15:
        advice.append("❄️ Cold weather → slower recovery, monitor regularly.")

    if not advice:
        advice.append("✅ Weather conditions are favorable for field activity.")

    return "\n".join(advice)
