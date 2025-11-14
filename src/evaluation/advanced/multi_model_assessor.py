#!/usr/bin/env python3
"""
Multi Model Assessor - Multi-model scoring via OpenRouter (ensemble of 5 AIs)

Approach:
- Uses OpenRouter (OPENROUTER_API_KEY) to query multiple models
- Each model is asked to rate a joke 0–10 and return only the number
- Final score is the arithmetic mean of model scores
- Caches per (joke, model) result in assessment_cache_multi_model.json

Usage (as library):
    from multi_model_assessor import MultiModelJokeAssessor
    assessor = MultiModelJokeAssessor()
    result = assessor.evaluate_joke("example joke")
    print(result["score"], result["per_model"])

Notes:
- Keeps the instruction minimal to match v2 behavior (numeric only output)
- If a model fails or returns no numeric token, assigns 0.0 for that model
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List

from openai import OpenAI  # type: ignore


ASSESSMENT_CACHE_MULTI_MODEL = os.path.join(
    "caches", "assessment_cache_multi_model.json"
)

# Default set of five diverse models on OpenRouter
DEFAULT_MODELS: List[str] = [
    "openai/gpt-5",
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-pro",
    "deepseek/deepseek-chat-v3-0324",
    "meta-llama/llama-3.1-405b-instruct",
]


class MultiModelJokeAssessor:
    def __init__(self, models: List[str] | None = None):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set in environment")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.models = models or DEFAULT_MODELS
        self.cache: Dict[str, Dict[str, float]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, float]]:
        if os.path.exists(ASSESSMENT_CACHE_MULTI_MODEL):
            try:
                with open(ASSESSMENT_CACHE_MULTI_MODEL, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Ensure numeric values
                cleaned: Dict[str, Dict[str, float]] = {}
                for joke, model_map in data.items() if isinstance(data, dict) else []:
                    if not isinstance(model_map, dict):
                        continue
                    cleaned[joke] = {}
                    for model, val in model_map.items():
                        try:
                            cleaned[joke][model] = float(val)
                        except (TypeError, ValueError):
                            continue
                return cleaned
            except (json.JSONDecodeError, OSError, ValueError):
                return {}
        return {}

    def _save_cache(self) -> None:
        try:
            with open(ASSESSMENT_CACHE_MULTI_MODEL, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    @staticmethod
    def _parse_score(text: str) -> float:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not match:
            return 0.0
        try:
            score = float(match.group(0))
        except ValueError:
            return 0.0
        # clamp to 0–10
        if score < 0:
            score = 0.0
        if score > 10:
            score = 10.0
        return score

    def _score_with_model(self, model: str, joke: str) -> float:
        # Check cache
        cached_for_joke = self.cache.get(joke, {})
        if model in cached_for_joke:
            return cached_for_joke[model]

        system = (
            "Role: You are one a panel of five human joke evaluators. Your task is to rate the funniness of a joke on a 0-4 scale, emulating typical human scoring patterns."
            ""
            "Rating Guidelines:"
            "	•	0: Not funny at all. The joke fails to elicit humor; it may be confusing, nonsensical, offensive without wit, or entirely predictable with no comedic payoff.:"
            "	•	1: Slightly amusing to at least one person, but weak overall. Punchline is obvious, stale, or poorly executed. Minimal originality or cleverness.:"
            "	•	2: Moderately funny to the group. Contains some wit, a modestly clever twist, or relatable humor, but is not especially strong or memorable. Mixed reactions among raters.:"
            "	•	3: Generally funny to most people. Clear setup and punchline, good timing, creative premise, and delivers a satisfying laugh to the majority.:"
            "	•	4: Very funny to nearly everyone. Original, well-crafted, unexpected, and consistently elicits strong laughter. Excellent comedic timing and broad appeal.:"
            ":"
            "Scoring Instructions::"
            "	•	Imagine hearing the joke in a neutral, everyday setting.:"
            "	•	Consider clarity, structure, originality, punchline strength, and audience relatability.:"
            "	•	If humor is niche, estimate how mixed reactions would average out in a group of five diverse raters.:"
            "	•	Output only the numerical rating from 0 to 4. Round to the nearest whole number.:"
            ""
            "Examples:"
            "Score ≈ 1 (rounded to 1)"
            "Joke: “I just bought a woollen sweater,” said Tom sheepishly.:"
            "Human Average: 1.20:"
            "Explanation: Simple pun, predictable, and low novelty; elicits minimal laughter from most listeners.:"
            ""
            "Score ≈ 2 (rounded to 2):"
            "Joke: Animals that tunnel in the soil have to have an escape root.:"
            "Human Average: 2.20:"
            "Explanation: Mild wordplay; some will appreciate the cleverness, others will find it obvious or flat.:"
            ""
            "Score ≈ 3 (rounded to 3)"
            "Joke: She told me the drink was non-alcoholic, but where was the proof?"
            "Human Average: 2.40"
            "Explanation: Play on “proof” (alcohol content vs. evidence). Moderate wit; gets a chuckle but not strong laughs."
            ""
            "Score ≈ 4 (rounded to 4)"
            "Joke: I still miss my ex-husband. But my aim is improving."
            "Human Average: 3.40"
            "Explanation: Dark humor with a clever twist; delivers a surprise punchline that many will find funny."
            ""
            "Output format:"
            "Rating: <number>"
        )
        user = f"Joke:\n{joke}"

        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = resp.choices[0].message.content.strip()
        except Exception:
            text = "0"

        score = self._parse_score(text)
        # Update cache
        if joke not in self.cache:
            self.cache[joke] = {}
        self.cache[joke][model] = score
        self._save_cache()
        return score

    def evaluate_joke(self, joke: str, use_cache: bool = True) -> Dict:
        per_model: Dict[str, float] = {}

        # If fully cached and allowed, short-circuit
        if (
            use_cache
            and joke in self.cache
            and all(m in self.cache[joke] for m in self.models)
        ):
            per_model = {m: float(self.cache[joke][m]) for m in self.models}
        else:
            for model in self.models:
                per_model[model] = self._score_with_model(model, joke)

        # Compute mean
        if per_model:
            mean = sum(per_model.values()) / len(per_model)
        else:
            mean = 0.0

        return {"score": mean, "per_model": per_model}


__all__ = ["MultiModelJokeAssessor", "DEFAULT_MODELS"]
