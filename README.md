# 🌲 Random Forest Analyzer
This project is a machine learning web application (via Streamlit) for interactive Random Forest modeling and dataset exploration. 
- Upload your dataset, clean and encode it automatically, train a Random Forest model (classification or regression, selected automatically), visualize the results, and export both your trained model and cleaned data in one place.

Author: Samantha Gauthier
- Email: samantha.gauthier7@gmail.com

---

## 🚀 Features

### 📂 Upload and Preview CSV Data
- Load CSV files directly from your computer and view your samples.

### 🧹 Automatic Cleaning and Encoding
- Fills missing numbers with the median
- Fills missing categories with the mode
- Maps categorical columns using LabelEncoder

### 🌳 Model Training
- Supports classification and regression via Random Forest
- Automatically detects problem type and selects correct model
- Adjustable test size, tree count, and CPU usage

### 📊 Visualization and Analytics
- Classifier: view confusion matrix and model prediction accuracy
- Regressor: view regression fit and residual plots
- Feature importance rankings
- PCA 2D projection with Plotly
- Correlation network using NetworkX
- Pairplot (Seaborn) to view data relationships
- Custom chart creation (scatter, bar, box and whisker, etc.)

### 🔮 Make Predictions
- Input custom feature values and generate predictions

### 💾 Export Results
- Download cleaned dataset as .csv
- Download trained Random Forest model as .joblib

### ♻️ Load Pretrained Models
- Reuse previous .joblib models to avoid retraining and maintain model accuracy
- Verfies dataset compatibility.

---

## 🧩 Dependencies
| Library                    | Purpose                                                     |
| -------------------------- | ----------------------------------------------------------- |
| **pandas**                 | Data manipulation and CSV handling                          |
| **numpy**                  | Numerical computations                                      |
| **seaborn**                | Statistical plots (pairplots, heatmaps, regression lines)   |
| **matplotlib**             | Data visualization                                          |
| **streamlit**              | Interactive web app framework                               |
| **plotly.express**         | Interactive visualizations (PCA, scatterplots)              |
| **networkx**               | Correlation network graph construction                      |
| **scikit-learn (sklearn)** | Data processing, Random Forest algorithms, model evaluation |
| **joblib**                 | Model saving/loading                                        |
| **io**                     | In-memory file operations for download buttons              |

> [NOTE]
> All dependencies are listed in requirements.txt for installation.

---

## 🌐 Accessing RF Analyzer
This program can be used locally or via web browser.

### 🔗 Web Browser Application
Link: https://rf-analyzer-baptiste.streamlit.app/
- Copy and paste into your preferred web browser.

> [NOTE]
> Project is hosted via the Streamlit Cloud.

### 💻 Installation (local)
1. Clone the repository:
``` bash
git clone https://github.com/sam-gauthier/RF-Analyzer
cd RF-Analyzer
```

2. Install dependencies:
``` bash
pip install -r requirements.txt
```

3. Run app:
``` bash
streamlit run RF.py
```
> [NOTE]
> Streamlit will automatically open the app in-browser on your local machine.
> The app may take a while to load.

---

## 🧠 Usage
1. Upload a CSV - Your dataset will be loaded and previewed
2. Model Setup - Choose the target feature and adjust the parameters
3. Train Model - Model training runs automatically (may take a moment). Preexisting model can also be uploaded optionally.
4. Visualize - View regression lines, feature importance for predictions, confusion matrix, etc.
5. Predit - Enter feature values to make your own predictions using the trained model.
6. Export - Download cleaned dataset and trained model.

---

## 📄 License
This project is open-source under the MIT License.
Feel free to use, modify, and distribute for academic or research purposes.
