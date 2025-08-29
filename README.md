# Money on the Mind: Do LLMs Get Selfish?

An NLP research project investigating whether monetary priming affects decision-making in Large Language Models (LLMs).

## Research Question

This project explores whether exposure to money-related concepts influences LLM behavior in strategic decision-making scenarios, specifically testing if LLMs become more selfish or cooperative when primed with different framings of money.

## Project Structure (will get bigger with time)

```
money_games/
├── NLP_group_20_Project_Runner.ipynb    # Reference notebook (end-to-end example)
├── requirements.txt                      # Project dependencies
└── src/
    ├── experiment/
    │   ├── main.py                      # CLI entry point to run experiments
    │   └── runner.py                    # Experiment orchestration logic
    ├── models/
    │   ├── config.py                    # Model constants
    │   ├── model_manager.py             # Model loading and management
    │   ├── open_source_model.py         # Open source model wrapper
    │   └── gemini.py                    # Gemini API wrapper
    ├── prompts/
    │   ├── prompt_builder.py            # Prompt construction utilities
    │   └── configs/                     # Constants for the prompts
    │       ├── money.py                 
    │       └── games.py                
    └── analysis/
        └── token_probs.py               # Token probability analysis tools
```

## Setup (with virtual environment)

It's recommended to use a virtual environment.

Windows (PowerShell):
```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

macOS/Linux (bash/zsh):
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quickstart (CLI)

Run using module execution with the venv's Python:

- Run both open-source and Gemini experiments (saves results to `results/experiment_results.csv`):
```bash
python -m src.experiment.main --hf-token YOUR_HF_TOKEN --gemini-key YOUR_GEMINI_KEY
```

- Run only open-source models:
```bash
python -m src.experiment.main --experiment open_source --hf-token YOUR_HF_TOKEN --output open_source_results.csv
```

- Run only Gemini:
```bash
python -m src.experiment.main --experiment gemini --gemini-key YOUR_GEMINI_KEY --output gemini_results.csv
```

- Dry run (print plan without executing):
```bash
python -m src.experiment.main --experiment both --hf-token YOUR_HF_TOKEN --gemini-key YOUR_GEMINI_KEY --dry-run
```

### Your example

```bash
python -m src.experiment.main --experiment open_source --hf-token YOUR_HF_TOKEN --gemini-key aa --output open_source_results.csv
```

### Arguments

- `--experiment|-e`: `open_source` | `gemini` | `both` (default: `both`)
- `--hf-token|-t`: HuggingFace token (required for `open_source` or `both`)
- `--gemini-key|-g`: Gemini API key (required for `gemini` or `both`)
- `--output|-o`: Output CSV filename (default: `experiment_results.csv`)
- `--output-dir`: Output directory (default: `results`)
- `--models`: Optional list to override models (uses defaults in `src/models/config.py` if omitted)
- `--clear-results`: Clear in-memory results before running
- `--dry-run`: Show configuration without executing
- `--verbose|-v`: Verbose error output

Environment variables are set automatically for the run, but you can also pre-set them:
- `HUGGINGFACE_TOKEN`
- `GEMINI_API_KEY`

## What the CLI prints/saves

- A tabular summary of results: `model`, `prefix_type`, `paraphrase_index`, `response`
- Counts per model/prefix and overall totals
- Decision token probability analysis for open-source models (if available)
- Response pattern analysis (samples by prefix and by model)
- Results saved as CSV to `--output-dir/--output`

## Notebook (reference)

You can still follow the full end-to-end example in `NLP_group_20_Project_Runner.ipynb`. The CLI mirrors the same flow (setup → run → summarize → analyze → save) and is better suited for automation.

