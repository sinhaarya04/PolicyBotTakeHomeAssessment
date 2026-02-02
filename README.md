# PolicyBotTakeHomeAssessment

[![Repo](https://img.shields.io/badge/GitHub-PolicyBotTakeHomeAssessment-181717?logo=github)](https://github.com/sinhaarya04/PolicyBotTakeHomeAssessment)

## Overview

A simple Python tool that finds medical codes (like CPT and ICD-10) in medical policy documents.

## Features

- Source code under `src/`

## Tech Stack

- Python

## Getting Started

### Prerequisites

- Git
- A recent runtime for the stack above (e.g., Python 3.10+ or Node 18+)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run / Usage

```bash
# See notebooks/scripts in this repo for entrypoints
```

## Project Structure

- `src/`
- `APPROACH_WRITEUP.md`
- `bert_pipeline.py`
- `bert_pipeline_review.py`
- `hcpcs.csv`
- `icd10cm.csv`
- `inferred_codes.csv`
- `inferred_codes_modular.csv`
- `LICENSE`
- `policies_cleaned.csv`

## Roadmap

- [ ] Add clearer usage examples and expected outputs
- [ ] Add tests / CI (if applicable)
- [ ] Document data sources and assumptions (if applicable)

## License

MIT
