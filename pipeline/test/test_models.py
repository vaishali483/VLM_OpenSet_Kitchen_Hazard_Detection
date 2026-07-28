from google import genai
from google.genai import errors
from config import *

client = genai.Client(api_key=GEMINI_API_KEY)

models = [
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

print()

for model in models:
    print(f"Testing {model:<30}", end="")

    try:
        response = client.models.generate_content(
            model=model,
            contents="Reply with only the word OK."
        )

        print("  ✓ WORKS")

    except Exception as e:
        print(f"  ✗ {type(e).__name__}")
        print(f"    {e}")

print()