import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


class DiagnosticGenerator:

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

    def generate(self, prompts: dict) -> dict:

        response = self.client.responses.create(
            model = self.model,
            instructions = prompts["system_prompt"],
            input = prompts["user_prompt"],
            text = {"format": {"type": "json_object"}}
        )

        return json.loads(response.output_text)
