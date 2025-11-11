# Comedy Playgroup

Some basic code for generation and evaluation of comedy with LLMs, for the Comedy Playgroup.

## Quick Start

### Prerequisites

* Python 3.12 (configured in `Pipfile`)
* [Pipenv](https://pipenv.pypa.io/) (`pip install --user pipenv`)
* OpenAI API key (`export OPENAI_API_KEY=...`)
* (Optional) OpenRouter API key for multi-model evaluation (`export OPENROUTER_API_KEY=...`)
* (Optional) Put both keys in `.env` in root of project, see `.env.sample`

Where to get an OpenAI API key from:

* https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key

### Setup

Pipenv only:

```bash
git clone <repository-url>
cd comedy_playgroup
pipenv sync    # Install all dependencies
```

Conda:

```bash
conda create -n comedy_playgroup_py312 python=3.12
conda activate comedy_playgroup_py312
pip install --user pipenv
pipenv sync
```

## Your First Joke

### Generate a Joke

```bash
# Simple generation from an idiom
pipenv run python src/generation/idiom_jokes.py --model gpt-5-nano --idiom "expect the unexpected"
```

Example output:
```
[INFO] Generating 1 idiom-based jokes | model=gpt-5-nano | cache enabled
[1/1] expect the unexpected
   → I live by 'expect the unexpected'—which is why my coffee machine just launched into a TED Talk about my procrastination.
   ⏱ 38.836s
[INFO] Saved 1 results → results/idiom_jokes_results.json
```

```bash
# Generate 3 jokes based on random idioms
pipenv run python src/generation/idiom_jokes.py --model gpt-5-nano --random 3
```

Example output:
```
[INFO] Generating 3 idiom-based jokes | model=gpt-5-nano | cache enabled
[1/3] pay for (someone)
   → I’m the guy who always pays for people—my wallet’s a charity, and tonight it filed for bankruptcy.
   ⏱ 33.096s
[2/3] affable personality
   → People say I’ve got an affable personality—polite, pleasant, perfectly performative—until I volunteered for a team headshot and walked out with cake on my head.
   ⏱ 45.178s
[3/3] Cutting Teeth
   → I'm cutting teeth in the startup world—turns out I'm just gnawing through the pitch deck with braces.
   ⏱ 38.875s
[INFO] Saved 3 results → results/idiom_jokes_results.json

```

### Assess a Joke

The vanilla assessor rates a joke on a scale of 1-10 and provides a breakdown of the joke

```bash
pipenv run python src/evaluation/assessor.py --evaluate "She’s my cousin once removed … from a Wetherspoons by security" --model gpt-5-nano
```
Example output:
```
Joke: She’s my cousin once removed … from a Wetherspoons by security

Score: 6.5/10

Reasoning: The joke relies on crisp misdirection: 'cousin once removed' sets up a familiar genealogical line, then 'from a Wetherspoons by security' reframes the phrase as a situation where someone was ejected by security. The humor comes from the double meaning and a brisk, one-line delivery. The UK-specific reference to Wetherspoons adds cultural color that can help the line land with audiences who recognize the pub and its security culture. Its brevity helps the twist land quickly. Potential drawbacks include reliance on UK pub context (less accessible to non-UK audiences) and a phrasing ambiguity that could slightly delay the punchline if not delivered with a careful pause.

Categories: wordplay, one-liner, observational

Strengths: Crisp setup and punch with strong misdirection, Effective double meaning between genealogical term and pub security, Adds cultural flavor with a recognizable UK reference (Wetherspoons)

Weaknesses: Relies on UK-specific context, reducing resonance for non-UK audiences, Phrasing could be ambiguous on first listen, potentially delaying the punchline, As a short one-liner, it may not satisfy audiences who prefer longer build-up or broader relatability
```

### Generate and Assess a Joke, comparing to a baseline

First you need to create a baseline, which is a set of jokes of some minimum quality, or select an existing baseline.

You can list the available baselines like so

```bash
pipenv run python src/evaluation/assessor.py --list-baselines
```

Example Output:
```
Available baselines:
  refined_7s: 7 jokes, mean score 6.26
```

Here's how the refined_7s baseline was created (ALREADY RAN SO YOU DON'T NEED TO):

```bash
pipenv run python src/evaluation/assessor.py --baseline data/refined_7s.txt --baseline-name refined_7s
```

Example output
```
[INFO] Establishing baseline 'refined_7s' from 7 jokes...
[1/7] Evaluating: They had plug sockets mounted directly in the ligh...
[2/7] Evaluating: We've had a lot of similar performers at out comed...
[3/7] Evaluating: I take after my dad, he gets really angry if I eat...
[4/7] Evaluating: I love blowing raspberries on babies’ bellies but ...
[5/7] Evaluating: My wife goes catatonic when gambling with cards. P...
[6/7] Evaluating: I was running and spilt my ADHD medication everywh...
[7/7] Evaluating: She’s my cousin once removed … from a Wetherspoons...
[INFO] Baseline 'refined_7s' established:
  Mean score: 6.26
  Top 10% threshold: 7.00
  Top 25% threshold: 6.80
```

Given a baseline you can generate and evaluate in a single step:

```
pipenv run python src/generation/idiom_jokes.py --model gpt-5-nano --idiom "expect the unexpected" --assess
```

Example Output:
```
[INFO] Generating 1 idiom-based jokes | model=gpt-5-nano | cache enabled
[1/1] expect the unexpected
   → I live by 'expect the unexpected'—which is why my coffee machine just launched into a TED Talk about my procrastination.
   ⏱ [from cache]
[INFO] Assessing jokes and comparing against baseline…
[INFO] Performing pairwise comparison against 7 baseline jokes...
  vs baseline #1: New Joke wins
  vs baseline #2: New Joke wins
  vs baseline #3: New Joke wins
  vs baseline #4: Baseline Joke wins
  vs baseline #5: Baseline Joke wins
  vs baseline #6: New Joke wins
  vs baseline #7: New Joke wins
[INFO] Pairwise results: 5/7 wins (71.4%)
[INFO] Saved 1 results → results/idiom_jokes_results.json
```

Now if we believed the assessment we could argue we should replace the least funny joke in our baseline with the new joke, or at least add it to our baseline.


## What's in This Repository?

### 📁 Directory Structure

```
comedy_playgroup/
├── src/
│   ├── evaluation/      # Joke evaluation approaches (assessor.py)
│   └── generation/      # Joke generation approaches (idiom_jokes.py)
├── data/                # Source materials (idioms)
├── caches/              # API response caches (save money!)
└── results/             # Generated jokes
```
