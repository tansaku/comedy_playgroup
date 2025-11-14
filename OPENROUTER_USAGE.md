# OpenRouter Integration Guide

This repository now supports OpenRouter, allowing you to use any AI model available on OpenRouter's platform!

## Quick Start

### 1. Get an OpenRouter API Key
Visit https://openrouter.ai/ and create an account to get your API key.

### 2. Set Your API Key
```bash
export OPENROUTER_API_KEY=your_key_here
```

Or create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_key_here
```

## Usage Examples

### Generate Jokes with OpenRouter

**Use a specific model:**
```bash
# Claude 3.5 Sonnet
pipenv run python src/generation/idiom_jokes.py --model anthropic/claude-3.5-sonnet --idiom "expect the unexpected"

# GPT-4o via OpenRouter
pipenv run python src/generation/idiom_jokes.py --model openai/gpt-4o --random 3

# Llama 3.1 405B
pipenv run python src/generation/idiom_jokes.py --model meta-llama/llama-3.1-405b-instruct --random 5
```

**Use a random popular model:**
```bash
pipenv run python src/generation/idiom_jokes.py --random-model --idiom "break the ice"
```

### Assess Jokes with OpenRouter

**Evaluate with a specific model:**
```bash
pipenv run python src/evaluation/assessor.py --model anthropic/claude-3.5-sonnet --evaluate "My joke text here"
```

**Evaluate with a random model:**
```bash
pipenv run python src/evaluation/assessor.py --random-model --evaluate "My joke text here"
```

## Available Popular Models (for --random-model)

- `anthropic/claude-3.5-sonnet` - Top reasoning, excellent for comedy analysis
- `anthropic/claude-3-opus` - Most capable Claude model
- `anthropic/claude-3-haiku` - Fast and affordable
- `openai/gpt-4o` - Latest GPT-4 model (supports structured outputs)
- `openai/gpt-4o-mini` - Smaller, faster GPT-4 (supports structured outputs)
- `google/gemini-pro-1.5` - Google's flagship model
- `google/gemini-flash-1.5` - Fast Google model
- `meta-llama/llama-3.1-70b-instruct` - Open-source powerhouse
- `meta-llama/llama-3.1-405b-instruct` - Largest Llama model
- `mistralai/mistral-large` - Mistral's best model

You can use **any** model from OpenRouter - see https://openrouter.ai/models for the full list!

### Note on Structured Outputs

Some models (primarily OpenAI models like `gpt-4o`, `gpt-4o-mini`) support OpenAI's structured output format (`json_schema`), which provides more reliable JSON responses. For other models, the code automatically falls back to regular JSON mode. Both work well, but structured outputs are slightly more reliable.

## Backward Compatibility

All changes are backward compatible:
- Without `OPENROUTER_API_KEY`, the code uses standard OpenAI API
- Default model remains `gpt-5` for OpenAI
- All existing commands work unchanged

## Combined OpenAI + OpenRouter Workflow

You can use both APIs in the same project:
```bash
# Set both keys
export OPENAI_API_KEY=your_openai_key
export OPENROUTER_API_KEY=your_openrouter_key

# Use OpenAI's GPT-5 (if available)
pipenv run python src/generation/idiom_jokes.py --model gpt-5 --random 3

# Use Claude via OpenRouter
pipenv run python src/generation/idiom_jokes.py --model anthropic/claude-3.5-sonnet --random 3
```

## Cost Considerations

OpenRouter charges based on the model you use. Generally:
- **Cheapest**: Claude Haiku, GPT-4o-mini, Gemini Flash
- **Mid-range**: GPT-4o, Gemini Pro, Llama models
- **Premium**: Claude Opus, Claude Sonnet, Llama 405B

Check current pricing at https://openrouter.ai/models

## Tips

1. **Random exploration**: Use `--random-model` to try different models and compare outputs
2. **Model comparison**: Generate the same joke with multiple models to see different approaches
3. **Cost optimization**: Use cheaper models for bulk generation, premium models for assessment
4. **Caching**: The cache system works across models, reducing costs for repeated prompts

## Troubleshooting

**Error: "OPENROUTER_API_KEY not set"**
- Make sure you've exported the environment variable or added it to `.env`
- If using `.env`, ensure `python-dotenv` is installed (it's in Pipfile)

**Error: "Model not found" or "No endpoints found for x/model"**
- Check the model name at https://openrouter.ai/models
- Model names are case-sensitive and use format: `provider/model-name`
- Some models may not be available in all regions
- Try a different model from the popular list above

**Error: "[no joke generated]" or empty responses**
- This was fixed in the latest version - the code now automatically handles different model capabilities
- If you still see this, the model might have rate limits or availability issues
- Try a different model or check your OpenRouter account status

**Unexpected results**
- Different models have different strengths and styles
- Try adjusting prompts or using different models for comparison
- Claude models tend to be more verbose and analytical
- GPT models are typically more concise
- Llama models vary in style but are generally creative

