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

After training, run this one-line command first. It is the safest option to copy and paste into any terminal:

```bash
python src/predict.py --model models/flight_delay_model.joblib --airline AA --origin JFK --destination LAX --scheduled-departure-hour 18 --day-of-week 5 --month 7 --distance 2475 --carrier-delay 10 --weather-delay 5
```

If you prefer a multi-line command in macOS/Linux Bash, each `\` must be the very last character on that line. Do not add spaces after it:
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

For Windows PowerShell, use backticks instead of backslashes:

```powershell
python src/predict.py `
  --model models/flight_delay_model.joblib `
  --airline AA `
  --origin JFK `
  --destination LAX `
  --scheduled-departure-hour 18 `
  --day-of-week 5 `
  --month 7 `
  --distance 2475 `
  --carrier-delay 10 `
  --weather-delay 5
```

If you see `bash: --model: command not found`, your shell treated each option as a separate command. Use the one-line command above, or make sure each Bash line-continuation `\` has no trailing spaces.

Example output:

```text
Prediction: Delayed
Delay probability: 0.67
```


### Train with Indian sample airline data

This repository also includes a small Indian domestic flight sample dataset for practice:

```bash
python src/train_model.py --data data/indian_sample_flights.csv --model-output models/indian_flight_delay_model.joblib
```

Then predict an Indian domestic flight with the trained Indian sample model:

```bash
python src/predict.py --model models/indian_flight_delay_model.joblib --airline 6E --origin DEL --destination BOM --scheduled-departure-hour 8 --day-of-week 1 --month 1 --distance 708 --carrier-delay 0 --weather-delay 25
```

In this sample, airline codes include `6E` for IndiGo, `AI` for Air India, `UK` for Vistara, `SG` for SpiceJet, and `QP` for Akasa Air. Airport codes include `DEL`, `BOM`, `BLR`, `HYD`, `MAA`, `CCU`, `GOI`, `AMD`, `IXB`, and `SXR`. The `delayed` column is still the output label: `1` means delayed and `0` means not delayed. This sample file is for learning and local testing, not official airline performance reporting.

Important: the `delayed` column is required only when you train the model because it is the answer the model learns. When you run `src/predict.py`, you do not pass `delayed`; the model creates that output for you as `Prediction: Delayed` or `Prediction: On time`.

If every prediction is showing `Delayed`, try an on-time style example with no known carrier or weather delay:

```bash
python src/predict.py --model models/indian_flight_delay_model.joblib --airline UK --origin BLR --destination DEL --scheduled-departure-hour 6 --day-of-week 2 --month 3 --distance 1080 --carrier-delay 0 --weather-delay 0
```

Try a delayed style example with weather delay:

```bash
python src/predict.py --model models/indian_flight_delay_model.joblib --airline 6E --origin DEL --destination BOM --scheduled-departure-hour 8 --day-of-week 1 --month 1 --distance 708 --carrier-delay 0 --weather-delay 25
```

To get a `Prediction: On time` result, first retrain the Indian model and then use a flight with no known carrier or weather delay:

```bash
python src/train_model.py --data data/indian_sample_flights.csv --model-output models/indian_flight_delay_model.joblib
python src/predict.py --model models/indian_flight_delay_model.joblib --airline UK --origin BLR --destination DEL --scheduled-departure-hour 6 --day-of-week 2 --month 3 --distance 1080 --carrier-delay 0 --weather-delay 0
```

Expected prediction style:

```text
Prediction: On time
Delay probability: 0.20
```

The exact probability can change, but the prediction should be lower risk than the weather-delay example.

If you keep testing the delayed example with `--weather-delay 25`, the model should usually return `Prediction: Delayed` because you are telling it that there is already a weather delay. Use `--carrier-delay 0 --weather-delay 0` when you want to test a not-delayed style case.

With this small learning dataset, predictions can be biased because there are only 25 rows. For better results, train with more real rows that include both `delayed = 0` and `delayed = 1`.

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
