# Copilot Instructions for OIAD-1-2025

## Project Overview
OIAD-1-2025 is a collaborative, multi-user Python project focused on data analysis and experimentation, with each contributor working in their own subdirectory. The project includes Jupyter notebooks, datasets, and Python dependencies for scientific computing and machine learning.

## Directory Structure
- `Boguk/`, `Dziga/`, `Kazlouski/`, `Kuharchuk/`, `Osotov/`, `Vorobeva/`: Contributor folders for individual work, often containing notebooks and text files.
- `labs/`: Shared lab notebooks (e.g., `1.ipynb`).
- `datasets/`: Contains datasets for analysis (e.g., `teen_phone_addiction_dataset.csv`).
- `requirements.txt`: Lists all Python dependencies (Jupyter, pandas, scikit-learn, matplotlib, etc.).
- `.venv/`: Local Python virtual environment (not tracked in git).

## Setup & Workflow
- **Environment Setup:**
  1. Create a virtual environment: `python -m venv .venv`
  2. Activate it:
     - Windows: `.venv\Scripts\activate`
     - Linux: `. .venv/bin/activate`
  3. Install dependencies: `pip install -r requirements.txt`
- **Notebook Usage:**
  - Jupyter notebooks are the main format for analysis and experimentation.
  - Each contributor works in their own folder; shared work goes in `labs/`.
- **Data Access:**
  - Use datasets from `datasets/` for analysis. Reference paths directly (e.g., `datasets/teen_phone_addiction_dataset.csv`).

## Conventions & Patterns
- **No central application entrypoint.** Work is organized by contributor and lab.
- **No custom build or test scripts.** Standard Python and Jupyter workflows apply.
- **Dependencies:** All required packages are in `requirements.txt`. Use the virtual environment for isolation.
- **Collaboration:**
  - Do not modify others' folders without permission.
  - Shared resources (datasets, labs) should be referenced, not copied.

## Example: Loading a Dataset in a Notebook
```python
import pandas as pd
df = pd.read_csv('datasets/teen_phone_addiction_dataset.csv')
df.head()
```

## Key Files & Directories
- `requirements.txt`: Dependency list
- `datasets/`: Data files
- `labs/`: Shared notebooks
- Contributor folders: Individual workspaces

## Guidance for AI Agents
- Respect the contributor folder boundaries.
- Use standard Python and Jupyter practices.
- Reference datasets and shared notebooks by relative path.
- Follow the setup instructions in `README.md` for environment management.
