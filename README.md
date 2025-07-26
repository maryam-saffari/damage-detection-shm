# 🏗️ Damage Detection in 5-Story Building Using One-Class SVM

This project applies machine learning techniques to detect and assess structural damage in a 5-story building using vibration data collected by accelerometers. The focus is on unsupervised anomaly detection via **One-Class SVM**, using only healthy-state data.

## 📁 Dataset Description

Accelerometer data were collected from each floor of a 5-story structure under both:
- **Healthy condition** (`/Healthy/`)
- **Four damage scenarios** (`/Case_1/`, `/Case_2/`, `/Case_3/`, `/Case_4/`)

Each dataset includes:
- Time series acceleration data from all 5 floors
- 6 records per healthy state, and multiple per damaged cases
- Sampling under different loading conditions

## 🧠 Methodology

### ✅ Machine Learning Model: `One-Class SVM`

- Unsupervised learning: trained only on healthy samples
- Targets **anomaly detection** without needing labeled damaged data
- Learns the normal feature distribution and flags deviations as damage

### 🔬 Signal Features Extracted

From each floor's acceleration time series:
- **AR model coefficients** (temporal structure)
- **Statistical features** of AR residuals (mean, variance, skewness, kurtosis)
- **Probability Density Function (PDF)** via kernel density estimation

These features capture both linear dynamics and subtle nonlinear changes due to damage.

## 📊 Output & Damage Levels

Each sample is classified as:
- `Healthy`, or
- Damaged: `Level 1` (minor) → `Level 4` (severe)

Each floor is evaluated independently, and then an **overall building health score** is aggregated per case.

| Case | Overall | Floor 1 | Floor 2 | Floor 3 | Floor 4 | Floor 5 |
|------|---------|---------|---------|---------|---------|---------|
| Case 1 | Level 4 | Level 2 | Level 3 | Level 3 | Level 3 | Level 2 |
| Case 2 | Level 3 | Level 3 | Level 3 | Level 2 | Level 3 | Level 2 |
| Case 3 | Level 1 | Level 1 | Level 2 | Healthy | Level 3 | Level 2 |
| Case 4 | Level 2 | Level 2 | Level 4 | Healthy | Level 2 | Level 3 |

## 📈 Visualization

Key results and score trends were visualized using:
- Decision function scores
- Damage level classification heatmaps (included in the notebook)

## 📓 Notebook

You can explore all code, training, and evaluation steps in the main Jupyter Notebook:

➡️ [`SHM_anomaly_detection.ipynb`](./SHM_anomaly_detection.ipynb)

## 📌 Key Takeaways

- One-Class SVM performed well in identifying anomalous behavior
- AR-based time series features are sensitive to structural damage
- System worked across varying load cases without needing damage labels

## 🛠️ Future Work

- Add supervised models if labeled damaged data become available
- Expand to frequency-domain or modal-based features
- Deploy real-time monitoring with streaming sensor input

---

© Maryam Saffari - Structural Health Monitoring Project - 2025
