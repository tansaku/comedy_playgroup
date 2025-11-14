#!/usr/bin/env python3
"""
Idiom Joke Generator - Create high-quality jokes from idioms/stock phrases using GPT-5

Features:
- Reads idioms from data/all_idioms.txt (or a custom file)
- Generates jokes by artfully transforming idioms with phonetic + semantic mechanisms
- Structured outputs via JSON schema for reliable parsing
- Caching to avoid repeat API calls (per idiom, model, prompt version)
- Optional cache bypass for speed testing with timing measurements
- Model selection support (gpt-5, gpt-5-mini, gpt-5-nano)
- Optional assessment and pairwise comparison against an existing baseline
- Reproducible sampling via random seed
"""


import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import openai

# ----------------------------- Configuration -----------------------------
GPT_MODEL = "gpt-5"
RANDOM_SEED = 42
PROMPT_VERSION = "1.1-idiom"
SCORE_THRESHOLD = 7.0
IDIOMS_FILE_DEFAULT = os.path.join("data", "all_idioms.txt")
RESULTS_FILE = os.path.join("results", "idiom_jokes_results.json")
CACHE_FILE = os.path.join("caches", "idiom_joke_cache.json")
BASELINE_NAME_DEFAULT = "refined_7s"

# Popular OpenRouter models for random selection
# Note: Only including models verified to work with our prompts
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
]

# Models that support OpenAI's structured output (json_schema)
STRUCTURED_OUTPUT_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
]


# ------------------------------ Data classes -----------------------------
@dataclass
class GenerationConfig:
    model: str
    prompt_version: str
    random_seed: Optional[int]
    use_cache: bool = True


# ------------------------------- Utilities -------------------------------
def get_git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


def load_json(path: str, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ModuleNotFoundError:
    pass


def read_idioms(path: str) -> List[str]:
    idioms: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            phrase = line.strip()
            if not phrase:
                continue
            # Skip extremely short fragments
            if len(phrase) < 6:
                continue
            idioms.append(phrase)
    return idioms


def sample_idioms(idioms: List[str], count: int, seed: Optional[int]) -> List[str]:
    rnd = random.Random(seed)
    if count >= len(idioms):
        return idioms[:]
    return rnd.sample(idioms, count)


# ------------------------------- Caching ---------------------------------
_CACHE: Dict[str, Dict] = load_json(CACHE_FILE, {})


def _cache_key(idiom: str, cfg: GenerationConfig) -> str:
    return f"{idiom.strip().lower()}|{cfg.model}|{cfg.prompt_version}"


# -------------------------- Joke Generation (LLM) -------------------------
# @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def generate_joke_from_idiom(idiom: str, cfg: GenerationConfig) -> Dict:
    """Generate a joke from an idiom using GPT-5 with structured output."""
    key = _cache_key(idiom, cfg)
    if cfg.use_cache and key in _CACHE:
        cached_result = _CACHE[key].copy()
        cached_result["from_cache"] = True
        return cached_result

    # Instruction: leverage two archetype patterns user supplied
    # 1) Phonetic/semantic twist reinforcing the meaning (e.g., consonant echo/alliteration + semantic alignment)
    # 2) Stock brag-setup flipped by humiliating image that undercuts the speaker
    prompt = f"""
You are a top-tier comedy writer. Transform this idiom or stock phrase into a genuinely funny, modern one-liner.

IDIOM: "{idiom}"

TARGETED MECHANISMS (pick whichever fits best for this idiom, or combine tastefully):
1) Phonetic + Semantic Reinforcement:
   - Create a witty transformation whose sound pattern (alliteration, initial-consonant echo, rhyme) mirrors the idiom's cadence
   - While the new content semantically reinforces or cleverly inverts the idiom's meaning
   - Example of the underlying idea (do NOT copy directly): turning a phrase like "expect the unexpected" into "expect the unexporcupine" (Michael Stranney) where the sentence seems to be completing normally but ends with something unexpected that until the last moment seems like part of the normal cadence of the idiom

2) Brag-Setup → Humiliating Reveal:
   - Start with a common boasty setup (a chestnut ad copy or self-help cliché)
   - End with a vivid, embarrassing image that undercuts the brag in a way that's surprising yet inevitable
   - Example of the underlying idea (do NOT copy directly): subverting "my friends laughed when I said I'd be a stand-up comic, but no one's laughting now" (Bob Monkhouse) i.e. nobody laughs now because the performance is bombing, but usually nobody laughing now would mean success

QUALITY RULES:
- Keep the idiom recognizable in spirit, but produce an original line (no direct reuse of the above examples)
- Use tight language; land on the funniest, most concrete word
- Specific words or slightly uncommon words can be funnier than generic ones, e.g. "porcupine" is funnier than "animal"
- Avoid forced puns; the phonetics should feel natural, not contorted
- The joke must stand on its own; don't explain it inside the joke

IMPORTANT: Return ONLY a valid JSON object with this exact structure:
{{
    "idiom": "{idiom}",
    "joke": "your funny one-liner here (1-2 sentences)",
    "transformation": "brief description of how you transformed the idiom",
    "mechanisms": ["mechanism1", "mechanism2"],
    "phonetic_devices": ["device1", "device2"],
    "semantic_devices": ["device1", "device2"],
    "explanation": "brief explanation of why this works"
}}
"""

    schema = {
        "type": "object",
        "properties": {
            "idiom": {"type": "string", "description": "The source idiom"},
            "joke": {
                "type": "string",
                "description": "The final joke (1–2 sentences)",
            },
            "transformation": {
                "type": "string",
                "description": "Short description of how the idiom was transformed",
            },
            "mechanisms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Mechanisms used (e.g., alliteration, cadence echo, inversion, humiliation reveal)",
            },
            "phonetic_devices": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Phonetic devices exploited (e.g., initial-consonant echo, rhyme, meter)",
            },
            "semantic_devices": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Semantic devices (reinforcement, inversion, misdirection, specificity)",
            },
            "explanation": {
                "type": "string",
                "description": "Why this joke works (brief)",
            },
        },
        "required": [
            "idiom",
            "joke",
            "transformation",
            "mechanisms",
            "phonetic_devices",
            "semantic_devices",
            "explanation",
        ],
        "additionalProperties": False,
    }

    # Check if using OpenRouter
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key:
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/tansaku/comedy_playgroup",
                "X-Title": "Comedy Playgroup",
            },
        )
    else:
        client = openai.OpenAI()

    # Check if model supports structured outputs (json_schema)
    supports_structured_output = cfg.model in STRUCTURED_OUTPUT_MODELS

    start_time = time.time()

    # Build request parameters
    request_params = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if supports_structured_output:
        # Use strict structured output for compatible models
        request_params["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "idiom_joke", "strict": True, "schema": schema},
        }
    else:
        # For other models, just use simple JSON mode
        request_params["response_format"] = {"type": "json_object"}

    if cfg.random_seed is not None:
        request_params["seed"] = cfg.random_seed

    response = client.chat.completions.create(**request_params)
    api_time = time.time() - start_time

    try:
        content = response.choices[0].message.content
        result = json.loads(content)

        # Ensure idiom field is set correctly
        result["idiom"] = idiom

        # Handle different field naming conventions (Claude sometimes uses "one_liner" instead of "joke")
        if not result.get("joke") and result.get("one_liner"):
            result["joke"] = result["one_liner"]
        elif not result.get("joke") and result.get("one-liner"):
            result["joke"] = result["one-liner"]

        # Ensure we have required fields with defaults
        if not result.get("joke"):
            print(
                f"[DEBUG] Response missing 'joke' field. Available fields: {list(result.keys())}"
            )
            result["joke"] = ""

        if not result.get("transformation") and result.get("mechanism_used"):
            result["transformation"] = result["mechanism_used"]

        if not result.get("mechanisms"):
            result["mechanisms"] = [result.get("mechanism_used", "unknown")]

        if not result.get("phonetic_devices"):
            result["phonetic_devices"] = []

        if not result.get("semantic_devices"):
            result["semantic_devices"] = []

        if not result.get("explanation"):
            result["explanation"] = result.get("explanation", "")

    except (json.JSONDecodeError, KeyError) as e:
        # Fallback minimal structure with debugging
        content = response.choices[0].message.content.strip()
        print(f"[DEBUG] JSON parsing failed: {e}")
        print(f"[DEBUG] Raw response (first 500 chars): {content[:500]}")

        # Try to extract joke from the response even if JSON parsing failed
        result = {
            "idiom": idiom,
            "joke": content if len(content) < 500 else "",
            "transformation": "fallback - parsing failed",
            "mechanisms": ["unknown"],
            "phonetic_devices": [],
            "semantic_devices": [],
            "explanation": f"Parsing failed: {str(e)}",
        }

    # Add timing information
    result["api_time_seconds"] = round(api_time, 3)
    result["from_cache"] = False

    # Cache and return
    if cfg.use_cache:
        _CACHE[key] = result
        save_json(CACHE_FILE, _CACHE)
    return result


def attach_metadata(record: Dict, cfg: GenerationConfig) -> Dict:
    enriched = {
        **record,
        "metadata": {
            "model": cfg.model,
            "prompt_version": cfg.prompt_version,
            "random_seed": cfg.random_seed,
            "use_cache": cfg.use_cache,
            "timestamp": time.time(),
            "git_commit": get_git_commit_hash(),
            "python_version": sys.version,
        },
    }
    return enriched


def evaluate_with_assessor(jokes: List[Dict], baseline_name: str) -> List[Dict]:
    try:
        from src.evaluation.assessor import JokeAssessor

        assessor = JokeAssessor()
    except (ImportError, RuntimeError):
        print("[WARN] assessor unavailable")
        return jokes

    enhanced: List[Dict] = []
    for j in jokes:
        joke_text = j.get("joke", "").strip()
        if not joke_text:
            enhanced.append(j)
            continue

        assessment = assessor.evaluate_joke(joke_text)
        j["assessment"] = assessment
        score = assessment.get("score", 0)

        if score >= SCORE_THRESHOLD:
            try:
                comparison = assessor.pairwise_compare_to_baseline(
                    joke_text, baseline_name, min_score_threshold=SCORE_THRESHOLD
                )
                j["baseline_comparison"] = comparison
            except ValueError as err:
                j["baseline_comparison"] = {"error": str(err)}

        enhanced.append(j)

    return enhanced


# --------------------------------- Main ----------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate jokes from idioms with GPT-5"
    )
    parser.add_argument(
        "--file", type=str, default=IDIOMS_FILE_DEFAULT, help="Idioms file path"
    )
    parser.add_argument(
        "--random",
        type=int,
        metavar="N",
        default=3,
        help="Generate jokes for N random idioms",
    )
    parser.add_argument(
        "--idiom", type=str, help="Generate a joke for a specific idiom"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=GPT_MODEL,
        help="Model to use (e.g., gpt-5, anthropic/claude-3.5-sonnet, openai/gpt-4o). For OpenRouter models, set OPENROUTER_API_KEY env variable.",
    )
    parser.add_argument(
        "--random-model",
        action="store_true",
        help="Use a random popular OpenRouter model (requires OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip cache to test actual API response times",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--assess",
        action="store_true",
        help="Evaluate jokes with assessor and compare to baseline",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=BASELINE_NAME_DEFAULT,
        help="Baseline name for comparison",
    )

    args = parser.parse_args()

    # Seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)

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

    # Load idioms
    try:
        idioms = read_idioms(args.file)
    except FileNotFoundError:
        print(f"[ERROR] Idioms file not found: {args.file}")
        sys.exit(1)

    # Determine target idioms
    targets: List[str]
    if args.idiom:
        targets = [args.idiom]
    else:
        targets = sample_idioms(idioms, args.random, args.seed)

    cfg = GenerationConfig(
        model=args.model,
        prompt_version=PROMPT_VERSION,
        random_seed=args.seed,
        use_cache=not args.no_cache,
    )

    cache_status = "cache disabled" if args.no_cache else "cache enabled"
    print(
        f"[INFO] Generating {len(targets)} idiom-based jokes | model={cfg.model} | {cache_status}"
    )

    results: List[Dict] = []
    for i, idiom in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {idiom}")
        gen = generate_joke_from_idiom(idiom, cfg)
        enriched = attach_metadata(gen, cfg)
        results.append(enriched)

        # Print the joke immediately for live feedback
        joke_text = enriched.get("joke", "").strip()
        if joke_text:
            print(f"   → {joke_text}")
        else:
            print("   → [no joke generated]")

        # Print timing information
        if enriched.get("from_cache"):
            print("   ⏱ [from cache]")
        elif "api_time_seconds" in enriched:
            print(f"   ⏱ {enriched['api_time_seconds']:.3f}s")

    if args.assess:
        print("[INFO] Assessing jokes and comparing against baseline…")
        results = evaluate_with_assessor(results, args.baseline)

    # Persist
    existing = load_json(RESULTS_FILE, [])
    existing.extend(results)
    save_json(RESULTS_FILE, existing)

    print(f"[INFO] Saved {len(results)} results → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
