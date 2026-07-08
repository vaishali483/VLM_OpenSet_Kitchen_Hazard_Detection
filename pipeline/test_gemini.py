from google import genai
from config import *

client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
    model=GEMINI_MODEL,
    contents="Describe what happens when someone cuts an onion with a kitchen knife."
)

print(response.text)