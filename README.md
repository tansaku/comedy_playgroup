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

[INFO] Generating 1 idiom-based jokes | model=gpt-5-nano | cache enabled
[1/1] expect the unexpected
   → I live by 'expect the unexpected'—which is why my coffee machine just launched into a TED Talk about my procrastination.
   ⏱ 38.836s
[INFO] Saved 1 results → results/idiom_jokes_results.json
```

```bash
# Generate 3 jokes based on random idioms
pipenv run python src/generation/idiom_jokes.py --model gpt-5-nano --random 3

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


## What's in This Repository?

### 📁 Directory Structure

```
comedy_playgroup/
├── src/
│   └── generation/      # Joke generation approaches
├── data/                # Source materials (idioms)
├── caches/              # API response caches (save money!)
└── results/             # Generated jokes
```
