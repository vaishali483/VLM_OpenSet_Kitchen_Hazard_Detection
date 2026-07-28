from google import genai
from config import *

# ==========================================================
# Gemini Client
# ==========================================================

client = genai.Client(api_key=GEMINI_API_KEY)

print("=" * 60)
print("Testing Gemini API")
print("=" * 60)

print(f"Model : {GEMINI_MODEL}")

try:

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents="In one sentence, describe someone chopping an onion safely.",
    )

    print("\n✓ Connection successful\n")

    print("Response:")
    print("-" * 60)
    print(response.text)
    print("-" * 60)

    # Optional metadata
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        print("\nUsage:")
        print(response.usage_metadata)

    print("\nGemini API is working correctly.")

except Exception as e:

    print("\n✗ Gemini API test failed.\n")
    print(type(e).__name__)
    print(e)