from google import genai
from config import *

client = genai.Client(api_key=GEMINI_API_KEY)

for model in client.models.list():
    actions = getattr(model, "supported_actions", [])

    if "generateContent" in actions:
        print(model.name)