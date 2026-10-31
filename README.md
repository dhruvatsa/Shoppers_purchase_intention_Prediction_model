# Online Shoppers Purchase Intention Prediction

A machine learning project that predicts whether an online shopping session is likely to result in a purchase. The project applies exploratory data analysis, data preprocessing, feature engineering, hyperparameter tuning, and multiple classification algorithms to analyse online shopper behaviour.

## Project Overview

E-commerce websites generate large amounts of behavioural data from user sessions. Understanding whether a visitor is likely to complete a purchase can help businesses improve customer engagement, marketing strategies, website design, and conversion rates.

This project uses the **Online Shoppers Purchasing Intention Dataset** to develop a binary classification system that predicts the `Revenue` variable:

- `True` → Purchase
- `False` → No Purchase

Four machine learning models are developed and compared:

1. K-Nearest Neighbours (KNN)
2. Support Vector Machine (SVM)
3. Logistic Regression
4. Decision Tree

The project also includes an interactive **Streamlit dashboard** for data exploration, model comparison, and purchase prediction.

---

## Objectives

The main objectives of this project are:

1. Analyse online shopping behaviour and identify patterns associated with purchasing and non-purchasing sessions.
2. Clean and transform the dataset for machine learning.
3. Create meaningful behavioural features such as `Total_Pages` and `Total_Duration`.
4. Develop and compare multiple machine learning classification models.
5. Evaluate the models using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
6. Develop an interactive dashboard for exploring the data and making purchase predictions.

---

## Dataset

### Online Shoppers Purchasing Intention Dataset

The dataset contains session-level information collected from users browsing an e-commerce website.

**Dataset source:** UCI Machine Learning Repository

**Dataset:** Online Shoppers Purchasing Intention Dataset

**Original reference:** Sakar, C. O., & Kastro, Y. (2018)

The dataset contains:

- 12,330 shopping sessions
- 18 original attributes
- Numerical and categorical variables
- A binary target variable called `Revenue`

### Important Features

Some of the main attributes include:

- `Administrative`
- `Administrative_Duration`
- `Informational`
- `Informational_Duration`
- `ProductRelated`
- `ProductRelated_Duration`
- `BounceRates`
- `ExitRates`
- `PageValues`
- `SpecialDay`
- `Month`
- `OperatingSystems`
- `Browser`
- `Region`
- `TrafficType`
- `VisitorType`
- `Weekend`
- `Revenue`

---

## Project Workflow

The project follows the following predictive analytics workflow:

```text
Dataset
   ↓
Data Loading
   ↓
Data Understanding
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Hyperparameter Tuning
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Feature Importance Analysis
   ↓
Streamlit Dashboard
```

---

## Exploratory Data Analysis

The EDA stage investigates:

- Purchase vs non-purchase distribution
- Purchase behaviour by month
- Purchase behaviour by visitor type
- Purchase behaviour on weekends
- Numerical feature distributions
- Relationships between numerical variables
- Correlation between features

The analysis helps identify patterns in visitor behaviour before applying machine learning algorithms.

---

## Data Preprocessing

The following preprocessing steps were performed:

### Duplicate Records

Duplicate records were identified and removed before model training.

### Binary Variables

The following Boolean variables were converted into numerical form:

```text
Weekend → 0 / 1
Revenue → 0 / 1
```

### Categorical Variables

`Month` and `VisitorType` were converted using one-hot encoding.

The first category was dropped using:

```python
drop_first=True
```

### Feature Engineering

Two additional behavioural features were created:

```text
Total_Pages
= Administrative + Informational + ProductRelated
```

```text
Total_Duration
= Administrative_Duration
+ Informational_Duration
+ ProductRelated_Duration
```

These features provide a broader representation of the visitor's overall website engagement.

---

## Train-Test Split

The dataset was divided into:

- **80% training data**
- **20% testing data**

The split uses:

```python
random_state=42
```

and stratification was applied to preserve the class distribution.

Feature scaling using `StandardScaler` was applied where required, particularly for distance- and margin-based algorithms.

---

## Machine Learning Models

### 1. K-Nearest Neighbours

KNN predicts the class of a session based on the classes of nearby observations.

The following values of `k` were tested:

```text
1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21
```

Five-fold cross-validation with F1-score was used to select the best value of `k`.

---

### 2. Support Vector Machine

An RBF-kernel Support Vector Machine was used for classification.

Configuration included:

```text
Kernel: RBF
class_weight: balanced
probability: True
```

Feature scaling was applied before training.

---

### 3. Logistic Regression

Logistic Regression was used as a linear classification model.

Configuration included:

```text
class_weight: balanced
max_iter: 1000
```

Feature scaling was applied before training.

---

### 4. Decision Tree

A Decision Tree classifier was developed to capture non-linear relationships between the variables.

The following maximum depths were evaluated:

```text
3, 5, 10, 15, 20, None
```

Five-fold cross-validation using F1-score was used for hyperparameter selection.

The final model used:

```text
class_weight: balanced
random_state: 42
```

---

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

F1-score is given particular importance because the dataset contains substantially more non-purchasing sessions than purchasing sessions.

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| KNN | 0.873 | 0.641 | 0.421 | 0.509 | 0.772 |
| SVM | 0.865 | 0.549 | 0.767 | 0.640 | 0.901 |
| Logistic Regression | 0.851 | 0.516 | 0.785 | 0.623 | 0.911 |
| Decision Tree | 0.848 | 0.509 | 0.861 | 0.640 | 0.921 |

### Best Model

Based on the primary evaluation criterion of **F1-score**, SVM and Decision Tree achieved the highest F1-score of **0.640**.

The **Decision Tree** provides an additional advantage through its higher:

- Recall: **0.861**
- ROC-AUC: **0.921**

Therefore, the Decision Tree is considered the preferred model when giving additional importance to identifying purchasing sessions.

KNN achieves the highest accuracy, but its low recall and F1-score show why accuracy alone is not sufficient for this imbalanced classification problem.

---

## Feature Importance

Feature importance analysis was performed to understand which variables contribute most to purchase prediction.

Two approaches were used:

### Decision Tree

The built-in feature importance values of the Decision Tree were used to identify the top 10 influential features.

### Logistic Regression

The absolute values of the Logistic Regression coefficients were used to identify the top 10 influential features.

This analysis provides additional interpretability and helps identify behavioural characteristics that are strongly associated with purchasing intention.

---

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard with four main sections:

### Project Overview

Provides:

- Project description
- Dataset overview
- Key project statistics
- Predictive analytics workflow

### Exploratory Data Analysis

Interactive visualizations covering:

- Purchase distribution
- Purchase behaviour by category
- Numerical feature analysis
- Correlation analysis

### Model Comparison

Displays:

- Model performance metrics
- Model comparison chart
- Confusion matrix
- Hyperparameter tuning results
- Feature importance

### Predict Purchase

Allows the user to enter characteristics of an online shopping session and obtain:

- Predicted purchase outcome
- Purchase probability
- Model-based prediction

---

## Project Structure

```text
Online-Shoppers-Purchase-Intention/
│
├── online_shoppers_intention.csv
│
├── notebooks/
│   ├── 01_import_libraries.ipynb
│   ├── 02_load_and_understand_data.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_clean_and_feature_engineering.ipynb
│   ├── 05_train_test_split_and_scaling.ipynb
│   ├── 06_knn_tuning.ipynb
│   ├── 07_decision_tree_tuning.ipynb
│   ├── 08_train_models.ipynb
│   ├── 09_model_evaluation.ipynb
│   ├── 10_feature_importance.ipynb
│   └── 11_conclusion.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── report/
```

Adjust the filenames or folders if your final GitHub repository uses a different structure.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- VS Code
- Streamlit

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Create a virtual environment if required:

```bash
python -m venv venv
```

Activate the environment.

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Running the Jupyter Notebooks

Start Jupyter:

```bash
jupyter notebook
```

Open the notebooks in numerical order, beginning with:

```text
01_import_libraries.ipynb
```

and ending with:

```text
11_conclusion.ipynb
```

---

## Running the Streamlit Dashboard

Make sure the dataset is in the same location expected by the application.

Run:

```bash
streamlit run app.py
```

The dashboard will open in your web browser.

---

## Key Findings

The project demonstrates that visitor behaviour contains useful information for predicting purchase intention.

Important observations include:

- Online shopping sessions are strongly dominated by non-purchasing sessions.
- Website engagement variables provide useful predictive information.
- Different machine learning algorithms produce different trade-offs between precision and recall.
- Accuracy alone can give a misleading impression of performance on an imbalanced dataset.
- SVM and Decision Tree achieved the highest F1-score in the final comparison.
- Decision Tree achieved the highest recall and ROC-AUC among the four models.
- Feature importance analysis provides additional insight into the variables associated with purchasing behaviour.

---

## Future Scope

Possible improvements include:

- Testing ensemble models such as Random Forest and Gradient Boosting
- Exploring XGBoost and other advanced algorithms
- More extensive hyperparameter optimisation
- Real-time purchase prediction
- Integration with live e-commerce data
- Model monitoring after deployment
- Improved dashboard visualizations
- Personalised recommendations based on visitor behaviour

---

## Dataset Reference

Sakar, C. O., & Kastro, Y. (2018). **Online Shoppers Purchasing Intention Dataset**. UCI Machine Learning Repository.

---

## Author

**Dhruv Vatsa**

INT234 – Predictive Analytics

Academic Year: 2025–2026

---

## License

This project is intended for **academic and educational purposes**.
