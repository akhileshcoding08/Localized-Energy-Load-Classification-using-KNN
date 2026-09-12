# ⚡ Localized Energy Load Classification using KNN

A machine learning project that classifies household/localized energy **demand levels (Low, Medium, High)** across different zones of Mumbai using the **K-Nearest Neighbors (KNN)** algorithm, based on smart-meter readings such as voltage, power, sub-metering, temperature, and humidity.

---

## 📌 Project Overview

Electricity distribution companies need to anticipate localized demand spikes to manage grid load, avoid outages, and plan resource allocation. This project builds a supervised classification model that predicts the **demand class** of a locality at a given point in time using electrical and weather-based features recorded from smart meters across 10 Mumbai zones.

The pipeline covers the complete data science lifecycle:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature encoding and scaling
- Model training with hyperparameter tuning (KNN)
- Model evaluation and result visualization

---

## 🗂️ Dataset

**File:** `localized_energy_load_classification.csv`

| Property | Details |
|---|---|
| Rows | ~221,100 |
| Columns | 17 |
| Target Variable | `demand_class` (Low / Medium / High) |
| Zones Covered | Navi Mumbai, Powai, Colaba, Thane, Borivali, Dadar, Kurla, Malad, Andheri, Bandra |

### Feature Description

| Column | Description |
|---|---|
| `timestamp` | Date and time of the reading |
| `zone` | Locality/zone in Mumbai |
| `month`, `hour`, `weekday` | Time-based features |
| `temperature_C` | Ambient temperature (°C) |
| `humidity_pct` | Relative humidity (%) |
| `voltage_V` | Supply voltage (V) |
| `global_intensity_A` | Current intensity (A) |
| `global_active_power_kW` | Active power consumption (kW) |
| `global_reactive_power_kW` | Reactive power consumption (kW) |
| `power_factor` | Power factor of the load |
| `sub_metering_1_Wh`, `sub_metering_2_Wh`, `sub_metering_3_Wh` | Sub-metered energy readings (Wh) for different appliance groups |
| `energy_consumed_kWh` | Total energy consumed (kWh) |
| `demand_class` | **Target** — Low / Medium / High demand |

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Model:** K-Nearest Neighbors (KNN) Classifier
- **Environment:** Jupyter Notebook

---

## 🔄 Project Workflow

1. **Data Loading** — Import the CSV dataset using pandas.
2. **Data Cleaning**
   - Removed duplicate records.
   - Imputed missing numeric values using the column median.
3. **Exploratory Data Analysis (EDA)**
   - Boxplots for outlier detection across all numeric features.
   - Distribution plots for temperature, humidity, and target class.
   - Zone-wise and time-wise (hourly/monthly/weekday) demand pattern analysis.
   - Correlation heatmap to study multicollinearity.
4. **Feature Engineering**
   - Sampled the dataset (10,000 rows) for computational efficiency.
   - Label-encoded categorical columns (`zone`, `demand_class`).
   - Split data into train (80%) and test (20%) sets.
   - Standardized features using `StandardScaler`.
5. **Model Building**
   - Trained a KNN Classifier across odd values of **K (1–19)**.
   - Evaluated each K using Accuracy, Precision, Recall, and F1-score.
6. **Model Evaluation**
   - Selected the best-performing K based on accuracy.
   - Visualized residual error distribution.
   - Plotted a predicted vs. actual demand timeline for a 200-hour sample window.

---

## 📊 Results

| K (Neighbors) | Accuracy |
|---|---|
| 1 | 86.5% |
| 3 | 89.2% |
| 5 | 89.9% |
| 7 | 90.0% |
| 9 | 90.0% |
| **11** | **90.4% (Best)** |
| 13 | 89.8% |
| 15 | 89.4% |
| 17 | 89.9% |
| 19 | 89.9% |

The model achieves its **best accuracy of ~90.4% at K = 11**, with balanced precision and recall across all three demand classes.

---

## 📁 Repository Structure

```
├── localized_energy_load_classification.csv     # Dataset
├── Localized_Energy_Load_KNN_Classification.ipynb  # Main notebook (EDA + Model)
├── README.md                                      # Project documentation
└── requirements.txt                               # Python dependencies
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/localized-energy-load-classification.git
cd localized-energy-load-classification
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebook
```bash
jupyter notebook Localized_Energy_Load_KNN_Classification.ipynb
```

---

## 📦 requirements.txt

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 🔮 Future Improvements

- Test additional classifiers (Random Forest, XGBoost, SVM) for comparison.
- Use the full dataset (221K+ rows) with optimized/approximate KNN (e.g., KD-Tree, Ball-Tree) for scalability.
- Perform hyperparameter tuning with `GridSearchCV`.
- Deploy the model via a simple Flask/Streamlit web app for real-time demand prediction.

---

## 🙋 Author

Feel free to connect for feedback, suggestions, or collaboration opportunities.
