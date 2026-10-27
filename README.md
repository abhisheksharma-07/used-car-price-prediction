# Used Car Price Prediction Using Artificial Neural Network (ANN)

A machine learning project that estimates the resale price of used cars in India using vehicle specifications, ownership information, and usage details. The project implements Artificial Neural Network (ANN) regression models, compares their performance with traditional machine learning models, and provides an interactive prediction interface using Streamlit.

## 1. Project Overview

Used car prices depend on several factors, including manufacturer, model, manufacturing year, kilometers driven, fuel type, transmission, engine specifications, and ownership history.

The objective of this project is to develop a data-driven system that estimates used car prices based on these attributes.

The project covers the complete machine learning workflow:

- Dataset exploration and data quality assessment
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature preparation and transformation
- ANN model training and evaluation
- Comparison with baseline and alternative regression models
- Model serialization and reuse
- Interactive price prediction through a Streamlit web application

**Current application model:** Random Forest Regressor, selected because it achieved the lowest Mean Absolute Error (MAE) among the evaluated models.

ANN models remain a core part of the project and are evaluated alongside the other regression models.

## 2. Project Objectives

1. Understand the factors influencing used car resale prices.
2. Clean and prepare vehicle data for machine learning.
3. Implement an Artificial Neural Network for regression using scikit-learn.
4. Compare ANN performance with Linear Regression and Random Forest.
5. Evaluate models using MAE, RMSE, and R².
6. Save the selected model and reuse it for predictions.
7. Develop an interactive application that estimates vehicle resale prices.

## 3. Dataset

**Dataset:** Vehicle Dataset from CarDekho

**Source:** [Kaggle — Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)

The dataset contains information about used vehicles, including their prices, specifications, and ownership details.

The raw dataset contains approximately **2,059 records and 20 columns**.

### Important features

| Feature | Description |
|---|---|
| `make` | Vehicle manufacturer |
| `model` | Vehicle model |
| `price` | Vehicle price; prediction target |
| `year` | Manufacturing year |
| `kilometer` | Distance driven |
| `fuel_type` | Fuel type |
| `transmission` | Manual or automatic transmission |
| `location` | Vehicle location |
| `color` | Exterior color |
| `owner` | Ownership history |
| `seller_type` | Type of seller |
| `engine` | Engine capacity |
| `max_power` | Maximum engine power |
| `max_torque` | Maximum torque |
| `drivetrain` | Vehicle drivetrain |
| `length` | Vehicle length |
| `width` | Vehicle width |
| `height` | Vehicle height |
| `seating_capacity` | Number of seats |
| `fuel_tank_capacity` | Fuel tank capacity |

The target variable is `price`, and the model uses the remaining 19 features as input.

## 4. Technology Stack

- **Programming language:** Python
- **Data manipulation:** Pandas, NumPy
- **Data visualization:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn
- **Neural network:** `MLPRegressor`
- **Model persistence:** Joblib
- **Web application:** Streamlit
- **Development environment:** Jupyter Notebook
- **Dataset storage:** CSV files

## 5. Project Structure

```text
used_car_price_prediction/
│
├── data/
│   ├── raw/
│   │   └── car_details.csv
│   │
│   └── processed/
│       └── car_details_cleaned.csv
│
├── models/
│   └── random_forest_pipeline.joblib
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── reports/
│   ├── figures/
│   │   ├── original_ann_actual_vs_predicted.png
│   │   ├── log_ann_actual_vs_predicted.png
│   │   ├── random_forest_actual_vs_predicted.png
│   │   ├── model_error_comparison.png
│   │   └── r2_comparison.png
│   │
│   └── metrics/
│       └── model_comparison.csv
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_loader.py
│   ├── prediction.py
│   └── ui.py
│
├── app.py
├── README.md
└── requirements.txt
```

The `.venv/` virtual environment is used locally and does not need to be committed to version control.

## 6. Project Workflow

### Phase 1: Data Exploration

Notebook: `01_data_exploration.ipynb`

The dataset was examined to understand its structure, feature distributions, missing values, numerical ranges, and relationships between vehicle attributes and prices.

The analysis included:

- Dataset dimensions and data types
- Missing-value analysis
- Duplicate-record checks
- Numerical summary statistics
- Categorical feature distributions
- Price distribution and outlier investigation
- Relationships between vehicle attributes and price

### Phase 2: Data Preprocessing

Notebook: `02_data_preprocessing.ipynb`

Module: `src/data_preprocessing.py`

The preprocessing workflow includes:

- Standardizing column names into snake_case.
- Cleaning text values.
- Converting numerical and measurement values into usable numeric features.
- Extracting numerical values from engine specifications, power, and torque.
- Validating required columns and numerical values.
- Checking duplicates and missing values.
- Saving the cleaned dataset to `data/processed/car_details_cleaned.csv`.

Missing values in technical specifications are retained for imputation during model training rather than discarding otherwise useful records.

### Phase 3: Model Training

Notebook: `03_model_training.ipynb`

The cleaned dataset is divided into training and testing subsets using an 80:20 split with `random_state=42`.

- Training records: 1,647
- Testing records: 412
- Input features: 19
- Target: `price`

The preprocessing pipeline handles numerical and categorical features separately.

**Numerical preprocessing**
- Median imputation for missing values
- Standardization using `StandardScaler`

**Categorical preprocessing**
- Most-frequent-value imputation
- One-hot encoding with unknown-category handling

The preprocessing transformations are incorporated into model pipelines so that prediction data receives the same transformations as training data.

### Phase 4: ANN Implementation

The ANN is implemented using scikit-learn's `MLPRegressor`.

The original-target ANN uses the following principal configuration:

| Parameter | Value |
|---|---|
| Hidden layers | 128 and 64 neurons |
| Activation function | ReLU |
| Optimizer | Adam |
| Regularization (`alpha`) | 0.001 |
| Initial learning rate | 0.001 |
| Maximum iterations | 500 |
| Early stopping | Enabled |
| Random state | 42 |

The ANN learns nonlinear relationships between vehicle features and price through interconnected neurons arranged in hidden layers.

Two ANN approaches were evaluated:

1. **Original-target ANN:** Trained directly on the vehicle price.
2. **Log-target ANN:** Trained on `log1p(price)`, with predictions converted back to the original price scale using `expm1`.

The logarithmic transformation helps reduce the influence of the highly skewed price distribution during training.

### Phase 5: Model Comparison

Three regression approaches were evaluated:

- Linear Regression — baseline model
- ANN with the original price target
- ANN with a logarithmically transformed target
- Random Forest Regressor — alternative ensemble model

The Random Forest configuration uses 300 estimators, `min_samples_leaf=2`, and `random_state=42`.

### Phase 6: Model Evaluation

Notebook: `04_model_evaluation.ipynb`

The models are evaluated using three standard regression metrics.

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted prices. Lower values indicate better average prediction accuracy.

**Root Mean Squared Error (RMSE)**

Measures prediction error while penalizing large errors more strongly. Lower values are better.

**R² Score**

Measures how much of the target's variance is explained by the model relative to a mean-prediction baseline. Higher values are generally better; a negative value indicates performance worse than that baseline.

The evaluation also includes actual-versus-predicted plots, model error comparisons, and investigation of errors for different price segments.

## 7. Model Evaluation Results

The following results were obtained from the project's held-out test set.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | ₹18,30,941.44 | ₹42,40,733.15 | -1.5742 |
| Original-target ANN | ₹7,08,097.33 | ₹16,83,744.40 | 0.5942 |
| Log-target ANN | ₹5,10,533.47 | ₹11,58,513.52 | 0.8079 |
| Random Forest | ₹3,06,199.29 | ₹11,76,929.73 | 0.8017 |

### Interpretation

- Linear Regression performed poorly on this dataset, suggesting that a simple linear relationship was insufficient for the observed price patterns.
- The original-target ANN improved upon the baseline model.
- The log-target ANN achieved a higher R² and lower RMSE than the original-target ANN.
- Random Forest achieved the lowest MAE among the evaluated models.
- The log-target ANN achieved a slightly better RMSE and R² than Random Forest.

**Current model selection:** Random Forest is used by the Streamlit application because it achieved the lowest MAE. This selection prioritizes average absolute prediction error rather than claiming that Random Forest is superior on every metric.

### Price-segment evaluation

The test-set errors also varied by vehicle price range.

| Price segment | Test records | Log-target ANN MAE | Random Forest MAE |
|---|---:|---:|---:|
| Budget: up to ₹10 lakh | 235 | ₹1,15,463 | ₹82,617 |
| Mid-range: ₹10–30 lakh | 116 | ₹4,86,137 | ₹2,63,942 |
| Premium: above ₹30 lakh | 61 | ₹20,78,919 | ₹12,47,901 |

The results indicate that prediction errors increase substantially for premium vehicles. Rare luxury models have limited representation in the training data, which can make their prices more difficult to estimate accurately.

## 8. Visualizations and Evaluation Artifacts

The following figures are saved in `reports/figures/`:

| File | Purpose |
|---|---|
| `original_ann_actual_vs_predicted.png` | Evaluates original-target ANN predictions |
| `log_ann_actual_vs_predicted.png` | Evaluates log-target ANN predictions |
| `random_forest_actual_vs_predicted.png` | Evaluates Random Forest predictions |
| `model_error_comparison.png` | Compares model MAE and RMSE |
| `r2_comparison.png` | Compares model R² scores |

The model metrics are also stored in `reports/metrics/model_comparison.csv`.

## 9. Interactive Streamlit Application

The application provides a form for estimating used car resale prices.

### Features

- Dynamic manufacturer and car-model selection
- Manufacturing year and kilometers-driven inputs
- Fuel type, transmission, location, color, ownership, seller type, and drivetrain selection
- Optional technical specifications
- Required-field validation
- Prediction using the saved Random Forest pipeline
- Estimated resale price displayed in Indian rupees

The interface uses the trained model and preprocessing pipeline saved in `models/random_forest_pipeline.joblib`.

The application is designed to run locally and does not require public deployment.

## 10. Installation and Setup

### Prerequisites

- Python compatible with the versions specified in `requirements.txt`
- pip
- The project files and saved model artifact
- The dataset files required by the application

### Step 1: Clone the repository

If the project is hosted on GitHub:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd used_car_price_prediction
```

Replace the placeholder with your repository URL.

If the project is already on your computer, open the project root directory in your terminal.

### Step 2: Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify the required files

Confirm that the following files exist:

- `data/processed/car_details_cleaned.csv`
- `models/random_forest_pipeline.joblib`
- `src/model_loader.py`
- `src/prediction.py`
- `src/ui.py`

If the raw data or trained model is not included in your repository, obtain the dataset from the source above and run the relevant preprocessing and training notebooks before launching the application.

### Step 5: Run the application

From the project root directory:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal, commonly `http://localhost:8501`. Open that URL in your browser.

## 11. Running the Notebooks

To explore the complete machine learning workflow, launch Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebooks in this order:

1. `01_data_exploration.ipynb`
2. `02_data_preprocessing.ipynb`
3. `03_model_training.ipynb`
4. `04_model_evaluation.ipynb`

The training notebook generates the model comparison results, figures, and saved model artifacts used by the application.

## 12. Limitations

- The dataset contains a limited number of records compared with the full used-car market.
- Prices and market conditions may change over time.
- Some technical specifications are missing and must be imputed.
- Rare luxury vehicles may be underrepresented in the training data.
- Premium vehicle predictions can have substantial absolute errors.
- Vehicle condition, accident history, maintenance records, and other potentially important factors may not be available as model inputs.
- The predicted price is an estimate, not a guaranteed sale price or professional vehicle valuation.

## 13. Future Enhancements

Potential improvements include:

- Collecting a larger and more recent dataset.
- Improving prediction accuracy for premium and rare vehicles.
- Exploring hyperparameter tuning for ANN and Random Forest.
- Adding additional vehicle-condition features where reliable data is available.
- Evaluating models with cross-validation and further robustness checks.
- Adding confidence or uncertainty information to predictions.
- Improving automated testing and application error handling.
- Evaluating the trained model on newer vehicle listings to assess generalization.

## 14. Conclusion

This project demonstrates an end-to-end machine learning workflow for used car price prediction, from data exploration and preprocessing to model training, evaluation, serialization, and interactive inference.

The ANN experiments demonstrate the application of neural networks to regression, including the effect of transforming a skewed target variable. Comparison with Linear Regression and Random Forest provides a practical basis for model selection.

Random Forest is currently used in the Streamlit application because it achieved the lowest MAE on the held-out test set, while the log-target ANN delivered a slightly better RMSE and R². These findings highlight the importance of evaluating multiple metrics when selecting a model for a real-world prediction task.
