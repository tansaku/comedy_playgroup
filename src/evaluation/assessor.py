import argparse
import json
import os
from typing import Dict

import openai

# Configuration
DEFAULT_ASSESSMENT_MODEL = "gpt-5"
ASSESSMENT_CACHE = os.path.join("caches", "assessment_cache.json")


class JokeAssessor:
    def __init__(self, model: str = DEFAULT_ASSESSMENT_MODEL):
        self.model = model
        self.client = openai.OpenAI()
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load assessment cache to avoid re-evaluating the same jokes."""
        if os.path.exists(ASSESSMENT_CACHE):
            with open(ASSESSMENT_CACHE, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Save assessment cache to disk."""
        os.makedirs(os.path.dirname(ASSESSMENT_CACHE), exist_ok=True)
        with open(ASSESSMENT_CACHE, "w") as f:
            json.dump(self.cache, f, indent=2)

    def evaluate_joke(self, joke: str, use_cache: bool = True) -> Dict:
        """
        Evaluate a single joke and return detailed assessment.

        Returns:
            Dict with keys: score, reasoning, categories, strengths, weaknesses
        """
        if use_cache and joke in self.cache:
            return self.cache[joke]

        prompt = f"""
        You are an expert comedy critic. Evaluate this joke on multiple dimensions:

        Joke: "{joke}"

        Provide a comprehensive assessment including:
        1. Overall score (1-10, where 10 is hilarious)
        2. Detailed reasoning for the score
        3. Comedy categories it fits (wordplay, observational, absurd, etc.)
        4. Specific strengths
        5. Areas for improvement

        Be thorough but concise in your analysis.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "joke_assessment",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {
                                "type": "number",
                                "description": "Overall humor score from 1-10",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Detailed explanation of the score",
                            },
                            "categories": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Comedy categories this joke fits",
                            },
                            "strengths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific strengths of the joke",
                            },
                            "weaknesses": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Areas where the joke could be improved",
                            },
                        },
                        "required": [
                            "score",
                            "reasoning",
                            "categories",
                            "strengths",
                            "weaknesses",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
        )

        try:
            assessment = json.loads(response.choices[0].message.content)
            if use_cache:
                self.cache[joke] = assessment
                self._save_cache()
            return assessment
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARN] Failed to parse assessment response: {e}")
            # Fallback simple assessment
            fallback = {
                "score": 5.0,
                "reasoning": "Assessment parsing failed",
                "categories": ["unknown"],
                "strengths": ["unknown"],
                "weaknesses": ["assessment error"],
            }
            return fallback


def main():
    parser = argparse.ArgumentParser(description="Joke Assessment Tool")
    parser.add_argument("--evaluate", type=str, help="Evaluate a single joke")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_ASSESSMENT_MODEL,
        choices=["gpt-5", "gpt-5-mini", "gpt-5-nano"],
        help="GPT model to use (default: gpt-5)",
    )

    args = parser.parse_args()
    assessor = JokeAssessor(args.model)

    if args.evaluate:
        result = assessor.evaluate_joke(args.evaluate)
        print(f"Joke: {args.evaluate}\n")
        print(f"Score: {result['score']}/10\n")
        print(f"Reasoning: {result['reasoning']}\n")
        print(f"Categories: {', '.join(result['categories'])}\n")
        print(f"Strengths: {', '.join(result['strengths'])}\n")
        print(f"Weaknesses: {', '.join(result['weaknesses'])}\n")


if __name__ == "__main__":
    main()
