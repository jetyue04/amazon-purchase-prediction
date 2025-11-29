# Amazon Purchase Prediction
CSE 158 Assignment 2 - Predicting if a customer will purchase an item

## Setup

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Data Script

To process the data files:

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On macOS/Linux

# Run the script
python data.py
```

This will:
- Load `All_Beauty.jsonl.gz` and `meta_All_Beauty.jsonl.gz`
- Split the data into train/validation/test sets (80/10/10 split based on timestamps)
- Save processed CSV files to the `data/` directory:
  - `data/train.csv`
  - `data/val.csv`
  - `data/test.csv`
  - `data/items.csv`
