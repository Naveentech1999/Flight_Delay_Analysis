# Flight Delay Analysis with Machine Learning

This project demonstrates an end-to-end Python machine learning workflow for analyzing and predicting flight departure delays. It includes a small sample dataset, reusable preprocessing code, model training, evaluation metrics, and a command-line prediction script.

## What the project does

- Loads flight data from a CSV file.
- Cleans and preprocesses categorical and numeric columns.
- Trains a Random Forest classifier to predict whether a flight will be delayed by 15 minutes or more.
- Prints accuracy, precision, recall, F1 score, and ROC-AUC metrics.
- Saves the trained model pipeline to disk.
- Provides a local command-line script for predicting delay risk for a single flight.

## Project structure

```text
Flight_Delay_Analysis/
├── data/
│   └── sample_flights.csv        # Small demo dataset for local testing
├── models/                       # Generated model files are saved here
├── src/
│   ├── predict.py                # CLI script for single-flight predictions
│   └── train_model.py            # Training and evaluation workflow
├── .gitignore
├── README.md
└── requirements.txt
```

## Prerequisites

Install the following before running the project locally:

- Python 3.10 or newer
- pip, which is included with most Python installations
- Git, if you want to clone the repository from a remote source

Check your Python version:

```bash
python --version
```

If your system uses `python3` instead of `python`, replace `python` with `python3` in the commands below.

## Installation on a local machine

### 1. Clone the repository

```bash
git clone <repository-url>
cd Flight_Delay_Analysis
```

If you already have the project folder locally, open a terminal in the `Flight_Delay_Analysis` directory instead.

### 2. Create a virtual environment

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## How to execute the project locally

### Train and evaluate the model

Run the training script with the included sample dataset:

```bash
python src/train_model.py --data data/sample_flights.csv --model-output models/flight_delay_model.joblib
```

Expected output includes model evaluation metrics similar to:

```text
Model evaluation
Accuracy: 0.80
Precision: 0.75
Recall: 0.75
F1 Score: 0.75
ROC-AUC: 0.88
Saved model to models/flight_delay_model.joblib
```

The exact values can vary when you use a larger or different dataset.

### Predict delay risk for one flight

After training, run:

```bash
python src/predict.py \
  --model models/flight_delay_model.joblib \
  --airline AA \
  --origin JFK \
  --destination LAX \
  --scheduled-departure-hour 18 \
  --day-of-week 5 \
  --month 7 \
  --distance 2475 \
  --carrier-delay 10 \
  --weather-delay 5
```

Example output:

```text
Prediction: Delayed
Delay probability: 0.67
```

## Using your own dataset

Your CSV file should include these columns:

| Column | Type | Description |
| --- | --- | --- |
| `airline` | Text | Airline or carrier code, such as `AA`, `DL`, or `UA` |
| `origin` | Text | Origin airport code |
| `destination` | Text | Destination airport code |
| `scheduled_departure_hour` | Number | Scheduled local departure hour from 0 to 23 |
| `day_of_week` | Number | Day of week from 1 to 7 |
| `month` | Number | Month from 1 to 12 |
| `distance` | Number | Flight distance in miles |
| `carrier_delay` | Number | Carrier delay minutes known before/at analysis time |
| `weather_delay` | Number | Weather delay minutes known before/at analysis time |
| `delayed` | Number | Target label: `1` if delayed by 15+ minutes, otherwise `0` |

Train with your own data:

```bash
python src/train_model.py --data path/to/your_flights.csv --model-output models/flight_delay_model.joblib
```

## Troubleshooting

- If `ModuleNotFoundError` appears, confirm your virtual environment is active and run `python -m pip install -r requirements.txt` again.
- If the model file is missing, run the training command before running `src/predict.py`.
- If your dataset has different column names, rename them to match the table above or update `FEATURE_COLUMNS` in `src/train_model.py`.
