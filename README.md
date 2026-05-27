# 🎗️ Breast Cancer Prediction ML

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [ML Models Used](#ml-models-used)
- [Results](#results)
- [How to Run](#how-to-run)
- [License](#license)

---

## 📖 About the Project

Breast cancer is one of the most common cancers worldwide. Early and accurate detection is critical for effective treatment. This project builds a **classification model** that predicts whether a tumor is:

- 🔴 **Malignant** (Cancerous)
- 🟢 **Benign** (Non-Cancerous)

Using features extracted from digitized images of fine needle aspirate (FNA) of breast mass.

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository - Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- **Samples:** 569
- **Features:** 30 numeric features
- **Target:** Malignant (M) / Benign (B)

### Key Features:
| Feature | Description |
|---|---|
| radius_mean | Mean of distances from center to points on the perimeter |
| texture_mean | Standard deviation of gray-scale values |
| perimeter_mean | Mean size of the core tumor |
| area_mean | Mean area of the tumor |
| smoothness_mean | Mean of local variation in radius lengths |

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.8+ | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Computing |
| Scikit-learn | Machine Learning Models |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Plots |
| Jupyter Notebook | Development Environment |

---

## 📁 Project Structure

```
Breast_Cancer_Prediction_ML/
│
├── data/
│   └── dataset.csv        # Dataset
│
├── NoteBooks/
│   └── B_Cancer.ipynb   # Main Notebook
│
├── .gitignore                   # Git ignore file
├── LICENSE                      # MIT License
└── README.md                    # Project Documentation
```

---

## 🤖 ML Models Used

| Model | Description |
|---|---|
| K-Nearest Neighbors (KNN) | Distance-based classification |
---

## 📈 Results

| Model | Accuracy |
|---|---|
| KNN | ~94% |
---

## ▶️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/charan-d11/Breast_Cancer_Prediction_ML.git
cd Breast_Cancer_Prediction_ML
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. Run the Notebook
```bash
jupyter notebook notebooks/breast_cancer_prediction.ipynb
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙋‍♂️ Author

**Durga Charan Mallick**
- GitHub: [@charan-d11](https://github.com/charan-d11)

---

⭐ **If you found this project helpful, please give it a star!** ⭐
