# Money on the Mind: Do LLMs Get Selfish?

An NLP research project investigating whether monetary priming affects decision-making in Large Language Models (LLMs).

## Research Question

This project explores whether exposure to money-related concepts influences LLM behavior in strategic decision-making scenarios, specifically testing if LLMs become more selfish or cooperative when primed with different framings of money.

## Project Structure (will get bigger with time)

```
money_games/
├── NLP_group_20_Project_Runner.ipynb    # Main experiment notebook
├── requirements.txt                      # Project dependencies
└── src/
    ├── experiment/
    │   └── runner.py                    # Main experiment orchestration
    ├── models/
    │   ├── config.py                    # Model constants
    │   ├── model_manager.py             # Model loading and management
    │   ├── open_source_model.py         # Open source model wrapper
    │   └── gemini.py                    # Gemini API wrapper
    ├── prompts/
    │   ├── prompt_builder.py            # Prompt construction utilities
    │   └── configs/                     # constants for the prompts
    │       ├── money.py                 
    │       └── games.py                
    └── analysis/
        └── token_probs.py               # Token probability analysis tools
```

