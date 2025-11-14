import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

import openai

# ----------------------------- Configuration -----------------------------
DEFAULT_ASSESSMENT_MODEL = "gpt-5"
BASELINE_FILE = os.path.join("results", "baseline_jokes.json")
ASSESSMENT_CACHE = os.path.join("caches", "assessment_cache.json")
PAIRWISE_CACHE = os.path.join("caches", "pairwise_cache.json")
ASSESSOR_VERSION = "1.0.0"
RANDOM_SEED = 42  # For reproducible pairwise comparisons

# Popular OpenRouter models for random selection
POPULAR_OPENROUTER_MODELS = [
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-haiku",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-pro-1.5",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct",
    "mistralai/mistral-large",
    "x-ai/grok-2",
]


# ------------------------------- Utilities -------------------------------
def get_git_commit_hash():
    """Get current git commit hash, or 'unknown' if not in git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


# -------------------------- Joke Assessment (LLM) -------------------------
class JokeAssessor:
    def __init__(self, model: str = DEFAULT_ASSESSMENT_MODEL):
        self.model = model
        # Check if using OpenRouter
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_api_key:
            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/tansaku/comedy_playgroup",
                    "X-Title": "Comedy Playgroup",
                }
            )
        else:
            self.client = openai.OpenAI()
        self.cache = self._load_cache()
        self.pairwise_cache = self._load_pairwise_cache()

    def get_assessor_metadata(self):
        """Get comprehensive version metadata for assessments."""
        return {
            "assessor_version": ASSESSOR_VERSION,
            "git_commit": get_git_commit_hash(),
            "python_version": sys.version,
            "assessment_model": self.model,
            "timestamp": time.time(),
        }

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

    def _load_pairwise_cache(self) -> Dict:
        """Load pairwise comparison cache."""
        if os.path.exists(PAIRWISE_CACHE):
            with open(PAIRWISE_CACHE, "r") as f:
                return json.load(f)
        return {}

    def _save_pairwise_cache(self):
        """Save pairwise comparison cache to disk."""
        os.makedirs(os.path.dirname(PAIRWISE_CACHE), exist_ok=True)
        with open(PAIRWISE_CACHE, "w") as f:
            json.dump(self.pairwise_cache, f, indent=2)

    def _get_pairwise_cache_key(self, joke_a: str, joke_b: str) -> str:
        """Generate a consistent cache key for joke pairs (order-independent)."""
        # Sort jokes to ensure A,B and B,A produce the same key
        jokes = sorted([joke_a, joke_b])
        return f"{jokes[0]}|||{jokes[1]}"

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

    def evaluate_collection(
        self, jokes: List[str], show_progress: bool = True
    ) -> List[Dict]:
        """Evaluate a collection of jokes and return sorted results."""
        results = []

        for i, joke in enumerate(jokes):
            if show_progress:
                print(f"[{i+1}/{len(jokes)}] Evaluating: {joke[:50]}...")

            assessment = self.evaluate_joke(joke)
            results.append(
                {
                    "joke": joke,
                    "assessment": assessment,
                    "version_metadata": self.get_assessor_metadata(),
                    "rank": 0,  # Will be filled after sorting
                }
            )

        # Sort by score (descending) and assign ranks
        results.sort(key=lambda x: x["assessment"]["score"], reverse=True)
        for i, result in enumerate(results):
            result["rank"] = i + 1

        return results

    def establish_baseline(
        self, jokes: List[str], baseline_name: str = "default"
    ) -> Dict:
        """Create a quality baseline from a collection of jokes."""
        print(
            f"[INFO] Establishing baseline '{baseline_name}' from {len(jokes)} jokes..."
        )

        results = self.evaluate_collection(jokes)

        # Calculate statistics
        scores = [r["assessment"]["score"] for r in results]
        baseline = {
            "name": baseline_name,
            "total_jokes": len(jokes),
            "mean_score": sum(scores) / len(scores),
            "median_score": sorted(scores)[len(scores) // 2],
            "top_10_percent_threshold": sorted(scores, reverse=True)[
                max(0, len(scores) // 10)
            ],
            "top_25_percent_threshold": sorted(scores, reverse=True)[
                max(0, len(scores) // 4)
            ],
            "created_at": time.time(),
            "version_metadata": self.get_assessor_metadata(),
            "results": results,
        }

        # Save baseline
        baselines = self._load_baselines()
        baselines[baseline_name] = baseline
        self._save_baselines(baselines)

        print(f"[INFO] Baseline '{baseline_name}' established:")
        print(f"  Mean score: {baseline['mean_score']:.2f}")
        print(f"  Top 10% threshold: {baseline['top_10_percent_threshold']:.2f}")
        print(f"  Top 25% threshold: {baseline['top_25_percent_threshold']:.2f}")

        return baseline

    def _load_baselines(self) -> Dict:
        """Load existing baselines from disk."""
        if os.path.exists(BASELINE_FILE):
            with open(BASELINE_FILE, "r") as f:
                return json.load(f)
        return {}

    def _save_baselines(self, baselines: Dict):
        """Save baselines to disk."""
        with open(BASELINE_FILE, "w") as f:
            json.dump(baselines, f, indent=2)

    def pairwise_compare_to_baseline(
        self,
        joke: str,
        baseline_name: str = "default",
        min_score_threshold: float = 7.0,
    ) -> Dict:
        """
        Compare a joke pairwise against baseline jokes if it meets the score threshold.

        Args:
            joke: The joke to compare
            baseline_name: Name of the baseline to compare against
            min_score_threshold: Minimum score required to proceed with pairwise comparison

        Returns:
            Dict with pairwise comparison results
        """
        baselines = self._load_baselines()

        if baseline_name not in baselines:
            raise ValueError(
                f"Baseline '{baseline_name}' not found. Available: {list(baselines.keys())}"
            )

        baseline = baselines[baseline_name]
        assessment = self.evaluate_joke(joke)
        score = assessment["score"]

        # Basic comparison data
        comparison = {
            "joke": joke,
            "score": score,
            "assessment": assessment,
            "baseline_name": baseline_name,
            "baseline_mean": baseline["mean_score"],
            "meets_score_threshold": score >= min_score_threshold,
            "pairwise_performed": False,
            "comparison_metadata": self.get_assessor_metadata(),
        }

        # Only do pairwise comparison if score meets threshold
        if score >= min_score_threshold:
            baseline_jokes = [r["joke"] for r in baseline["results"]]
            wins = 0
            total_comparisons = len(baseline_jokes)

            print(
                f"[INFO] Performing pairwise comparison against {total_comparisons} baseline jokes..."
            )

            for i, baseline_joke in enumerate(baseline_jokes):
                pairwise_result = self.pairwise_compare_jokes(joke, baseline_joke)
                if pairwise_result["winner"] == "A":  # Our joke won
                    wins += 1
                print(
                    f"  vs baseline #{i+1}: {'New joke' if pairwise_result['winner'] == 'A' else 'Baseline joke'} wins"
                )

            win_rate = wins / total_comparisons
            comparison.update(
                {
                    "pairwise_performed": True,
                    "wins_against_baseline": wins,
                    "total_baseline_comparisons": total_comparisons,
                    "win_rate": win_rate,
                    "beats_majority": win_rate > 0.5,
                    "qualifies_for_baseline": win_rate
                    >= 0.6,  # Must beat 60% to be baseline-worthy
                }
            )

            print(
                f"[INFO] Pairwise results: {wins}/{total_comparisons} wins ({win_rate:.1%})"
            )
        else:
            print(
                f"[INFO] Score {score:.1f} below threshold {min_score_threshold}, skipping pairwise comparison"
            )

        return comparison

    def pairwise_compare_jokes(self, joke_a: str, joke_b: str) -> Dict:
        """
        Compare two jokes directly and determine which is funnier.

        Returns:
            Dict with keys: winner, joke_a_score, joke_b_score, reasoning
        """
        cache_key = self._get_pairwise_cache_key(joke_a, joke_b)
        if cache_key in self.pairwise_cache:
            return self.pairwise_cache[cache_key]

        prompt = f"""
        You are an expert comedy critic. Compare these two jokes and determine which is funnier:

        Joke A: "{joke_a}"
        Joke B: "{joke_b}"

        Provide:
        1. Which joke is funnier (A or B, or "tie" if they're equally funny)
        2. A score for each joke (1-10)
        3. Brief reasoning for your decision

        Be decisive - avoid ties unless the jokes are truly equally funny.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "pairwise_comparison",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "winner": {
                                "type": "string",
                                "enum": ["A", "B", "tie"],
                                "description": "Which joke is funnier",
                            },
                            "joke_a_score": {
                                "type": "number",
                                "description": "Score for joke A (1-10)",
                            },
                            "joke_b_score": {
                                "type": "number",
                                "description": "Score for joke B (1-10)",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of the decision",
                            },
                        },
                        "required": [
                            "winner",
                            "joke_a_score",
                            "joke_b_score",
                            "reasoning",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
        )

        try:
            result = json.loads(response.choices[0].message.content)
            self.pairwise_cache[cache_key] = result
            self._save_pairwise_cache()
            return result
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARN] Failed to parse pairwise comparison response: {e}")
            # Fallback
            return {
                "winner": "tie",
                "joke_a_score": 5.0,
                "joke_b_score": 5.0,
                "reasoning": "Comparison parsing failed",
            }


def main():
    import random

    parser = argparse.ArgumentParser(description="Joke Assessment Tool")
    parser.add_argument("--evaluate", type=str, help="Evaluate a single joke")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_ASSESSMENT_MODEL,
        help="Model to use (e.g., gpt-5, anthropic/claude-3.5-sonnet, openai/gpt-4o). For OpenRouter models, set OPENROUTER_API_KEY env variable.",
    )
    parser.add_argument(
        "--random-model",
        action="store_true",
        help="Use a random popular OpenRouter model (requires OPENROUTER_API_KEY)",
    )
    parser.add_argument("--baseline", type=str, help="Create baseline from file")
    parser.add_argument(
        "--baseline-name", type=str, default="default", help="Name for the baseline"
    )
    parser.add_argument(
        "--list-baselines", action="store_true", help="List available baselines"
    )
    args = parser.parse_args()

    # Handle random model selection
    if args.random_model:
        if not os.getenv("OPENROUTER_API_KEY"):
            print("[WARN] --random-model requires OPENROUTER_API_KEY to be set")
            print("[INFO] Available popular OpenRouter models:")
            for model in POPULAR_OPENROUTER_MODELS:
                print(f"  - {model}")
            sys.exit(1)
        args.model = random.choice(POPULAR_OPENROUTER_MODELS)
        print(f"[INFO] Randomly selected model: {args.model}")

    assessor = JokeAssessor(args.model)

    if args.evaluate:
        result = assessor.evaluate_joke(args.evaluate)
        print(f"Joke: {args.evaluate}\n")
        print(f"Score: {result['score']}/10\n")
        print(f"Reasoning: {result['reasoning']}\n")
        print(f"Categories: {', '.join(result['categories'])}\n")
        print(f"Strengths: {', '.join(result['strengths'])}\n")
        print(f"Weaknesses: {', '.join(result['weaknesses'])}\n")
    elif args.list_baselines:
        baselines = assessor._load_baselines()
        if baselines:
            print("Available baselines:")
            for name, baseline in baselines.items():
                print(
                    f"  {name}: {baseline['total_jokes']} jokes, mean score {baseline['mean_score']:.2f}"
                )
        else:
            print("No baselines found.")
    elif args.baseline:
        with open(args.baseline, "r") as f:
            jokes = [line.strip() for line in f if line.strip()]

        assessor.establish_baseline(jokes, args.baseline_name)


if __name__ == "__main__":
    main()
