import os
from langchain_google_genai import GoogleGenerativeAI

class LLMConnector:
    def __init__(self):
        # Ensure the environment variable is set
        if 'GOOGLE_AI_KEY' not in os.environ:
            raise ValueError("GOOGLE_AI_KEY environment variable is not set")
    
    def get_llm(self):
        return GoogleGenerativeAI(
            google_api_key=os.environ['GOOGLE_AI_KEY'],
            model="gemini-1.5-pro",
            max_output_tokens=1024,
            temperature=0.2,
            top_p=0.8,
            top_k=40
            )

