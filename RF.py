"""
RF.py
------------
Description:
Random Forest web application via Streamlit.
- Upload, analyze, and visualize datasets (csv)
- Clean and encode data automatically
- Make predictions using Random Forest Classification / Regression models
- Calculates model accurary, feature importance, confusion matrix, etc.
- Export / upload trained models (joblib)
- Export cleaned data (csv)

Dependencies:
- pandas: for data manipulation of csv files
- numpy: for numerical computations
- seaborn: for statistical plotting (pairplots, regression lines, heatmaps)
- matplotlib: for data visulization
- streamlit: to build interactive web application
- plotly.express: for interactive visualizations (PCA and scatterplots)
- networkx: to build / display correlation networks
- sklearn / scikit-learn: for data processing, RF classification and regression models, model evaluation
- joblib: for saving and loading trained models
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, root_mean_squared_error, r2_score, confusion_matrix
from sklearn.inspection import PartialDependenceDisplay
from sklearn.decomposition import PCA
import joblib

st.set_page_config(page_title="RF Analyzer")
st.title("Random Forest Analyzer")
st.write("Upload a dataset, train a Random Forest model, and visualize your data.")

# -------------------------------
# Caching to prevent slow load times.
# -------------------------------
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
def train_model(X_train, y_train, problem_type, num_trees, n_jobs):
    if problem_type == "classification":
        m = RandomForestClassifier(n_estimators=num_trees, random_state=42, n_jobs=n_jobs)
    else:
        m = RandomForestRegressor(n_estimators=num_trees, random_state=42, n_jobs=n_jobs)
    m.fit(X_train, y_train)
    return m

# -------------------------------
# Upload & cleaning data 
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is None:
    st.info("Please upload a CSV file to start.")
    st.stop()

df = load_data(uploaded_file)
st.subheader("Preview of Uploaded Data")
st.dataframe(df.head())

# Clean and encode
df_clean = df.copy()
for col in df_clean.columns:
    if pd.api.types.is_numeric_dtype(df_clean[col]):
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    else:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else "")

label_encoders = {}
for col in df_clean.select_dtypes(include=["object", "category"]).columns:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le

st.success("✅ Data cleaned and encoded successfully!")

# -------------------------------
# Upload Previously Saved Model
# -------------------------------
st.subheader("🔄 Load Previously Saved Model (optional)")

uploaded_model = st.file_uploader("Upload a saved model (.joblib)", type=["joblib"])
loaded_model = None
if uploaded_model is not None:
    try:
        loaded_model = joblib.load(uploaded_model)
        st.success("✅ Model loaded successfully!")

        # Verify feature compatibility
        if hasattr(loaded_model, "feature_names_in_"):
            model_features = list(loaded_model.feature_names_in_)
            dataset_features = list(df.columns)

            missing_in_data = [f for f in model_features if f not in dataset_features]
            extra_in_data = [f for f in dataset_features if f not in model_features]

            if missing_in_data:
                st.warning(f"⚠️ The dataset is missing features required by the model: {missing_in_data}")
            elif extra_in_data:
                st.info(f"ℹ️ Note: The dataset has extra columns not used by the model: {extra_in_data}")

        # Store model to be used later
        model = loaded_model
    except Exception as e:
        st.error(f"Error loading model: {e}")

# -------------------------------
# Target selection
# -------------------------------
st.subheader("Model Setup")
target = st.selectbox("Select target column (Y):", df_clean.columns)
X = df_clean.drop(columns=[target])
y = df_clean[target]

is_int_like = pd.api.types.is_integer_dtype(y) or pd.api.types.is_bool_dtype(y)
problem_type = "classification" if (len(y.unique()) <= 10 and is_int_like) else "regression"
st.write(f"Detected problem type: **{problem_type}**")

# Train/test
test_size = st.slider("Test size", 0.05, 0.5, 0.25)
n_trees = st.number_input("Number of trees", 10, 1000, 150, 10)
use_all_cores = st.checkbox("Use all CPU cores (n_jobs=-1)", value=True)
n_jobs = -1 if use_all_cores else 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
with st.spinner("Training Random Forest..."):
    model = train_model(X_train, y_train, problem_type, int(n_trees), n_jobs)

# -------------------------------
# Evaluation
# -------------------------------
st.subheader("Model Evaluation")
if problem_type == "classification":
    preds = model.predict(X_test)
    st.write("Accuracy:", round(accuracy_score(y_test, preds), 4))
    st.text(classification_report(y_test, preds))
else:
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    st.write(f"Root Mean Squared Error: {rmse:.4f}")
    st.write(f"R² Score: {r2:.4f}")

if problem_type == "regression":
    # Predicted vs Actual with regression line
    fig, ax = plt.subplots()
    sns.regplot(x=y_test, y=preds, line_kws={"color": "red"}, ax=ax)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Regression Fit: Predicted vs Actual")
    st.pyplot(fig)

    # Residual Plot
    fig, ax = plt.subplots()
    residuals = y_test - preds
    sns.scatterplot(x=preds, y=residuals, ax=ax)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    ax.set_title("Residual Plot")
    st.pyplot(fig)

elif problem_type == "classification":
    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # Optional ROC Curve
    from sklearn.metrics import RocCurveDisplay
    if len(np.unique(y_test)) == 2:  # Binary classification only
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        st.pyplot(plt.gcf())

# -------------------------------
# Feature Importance
# -------------------------------
st.subheader("Feature Importance")
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, min(6, 0.3 * len(feat_imp))))
sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax)
ax.set_title("Feature Importances")
st.pyplot(fig)

# -------------------------------
# PCA Visualization
# -------------------------------
st.subheader("PCA (2D Projection)")
n_samples_pca = st.slider("Max samples for PCA", 50, min(1000, len(df_clean)), 300, 50)
pca_sample = df_clean.sample(min(n_samples_pca, len(df_clean)), random_state=42)
pca = PCA(n_components=2)
comps = pca.fit_transform(pca_sample.drop(columns=[target]))
pca_df = pd.DataFrame(comps, columns=["PC1", "PC2"])
pca_df[target] = pca_sample[target]
fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color=target, title="PCA (2D Projection)")
st.plotly_chart(fig_pca, use_container_width=True)

# -------------------------------
# Correlation Network
# -------------------------------
st.subheader("Correlation Network")
corr_thresh = st.slider("Correlation threshold", 0.3, 0.95, 0.7, 0.05)
corr = df_clean.corr().abs()
G = nx.Graph()
for i, c1 in enumerate(corr.columns):
    for j, c2 in enumerate(corr.columns):
        if j <= i:
            continue
        if corr.loc[c1, c2] >= corr_thresh:
            G.add_edge(c1, c2, weight=corr.loc[c1, c2])
if G.number_of_edges() == 0:
    st.info("No correlations above threshold.")
else:
    fig_net, ax_net = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=600, node_color="skyblue", edge_color="gray", ax=ax_net)
    st.pyplot(fig_net)

# -------------------------------
# Pairplot
# -------------------------------
st.subheader("Pairplot (Sampled Data)")

@st.cache_resource
def generate_pairplot(data, target_col, problem_type):
    pair_df = data.sample(min(300, len(data)), random_state=42)
    pp = sns.pairplot(
        pair_df,
        diag_kind="kde",
        hue=target_col if problem_type == "classification" else None
    )
    return pp

try:
    pairplot_fig = generate_pairplot(df_clean, target, problem_type)
    st.pyplot(pairplot_fig)
except Exception as e:
    st.warning(f"Pairplot not available: {e}")


# -------------------------------
# Custom Graph Section
# -------------------------------
st.subheader("📊 Custom Visualization")
st.write("Select a chart type and choose which features to plot. If dataset is large, up to 100 samples will be used.")

chart_type = st.selectbox(
    "Choose a chart type:",
    ["Scatter", "Line", "Box", "Violin", "Histogram", "Bar"]
)

x_col = st.selectbox("X-axis:", df_clean.columns)
y_col = st.selectbox("Y-axis:", df_clean.columns)

# Sample limit
plot_df = df_clean.sample(min(len(df_clean), 100), random_state=42)

# Plot generation
fig, ax = plt.subplots(figsize=(6, 4))
try:
    if chart_type == "Scatter":
        sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=target, ax=ax)
    elif chart_type == "Line":
        sns.lineplot(data=plot_df, x=x_col, y=y_col, hue=target, ax=ax)
    elif chart_type == "Box":
        sns.boxplot(data=plot_df, x=x_col, y=y_col, ax=ax)
    elif chart_type == "Violin":
        sns.violinplot(data=plot_df, x=x_col, y=y_col, ax=ax)
    elif chart_type == "Histogram":
        sns.histplot(data=plot_df, x=x_col, hue=target, multiple="stack", ax=ax)
    elif chart_type == "Bar":
        sns.barplot(data=plot_df, x=x_col, y=y_col, ax=ax)
    ax.set_title(f"{chart_type} Plot of {y_col} vs {x_col}")
    st.pyplot(fig)
except Exception as e:
    st.error(f"Could not generate plot: {e}")

#################
# Predictions
################

st.subheader("Make Your Own Predictions")

# Automatically build input fields for each feature
user_input = {}
encoders = {}

for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        # Encode categories
        options = list(X[col].unique())
        selected = st.selectbox(f"Select {col}:", options)
        user_input[col] = selected
        # Store encoder mapping
        encoders[col] = {label: idx for idx, label in enumerate(options)}
    else:
        # Numeric input (sliders)
        min_val = float(X[col].min())
        max_val = float(X[col].max())
        mean_val = float(X[col].mean())
        user_input[col] = st.number_input(f"Enter {col}:")

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Encode categorical inputs for prediction
for col in encoders:
    input_df[col] = input_df[col].map(encoders[col])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)

    if problem_type == "regression":
        st.success(f"Predicted Value: **{prediction[0]:.4f}**")

    elif problem_type == "classification":
        pred_class = prediction[0]
        st.success(f"Predicted Class: **{pred_class}**")

        # Show probabilities (if supported)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            prob_df = pd.DataFrame({
                "Class": model.classes_,
                "Probability": probs
            }).sort_values(by="Probability", ascending=False)
            st.write("### Class Probabilities:")
            st.dataframe(prob_df)

# -------------------------------
# Save outputs
# -------------------------------
st.subheader("Export / Save")

# --- Save Cleaned CSV ---
csv_filename = st.text_input("Enter filename for cleaned CSV:", "cleaned_data.csv")

csv_bytes = df_clean.to_csv(index=False).encode()
st.download_button(
    label="⬇️ Download Cleaned CSV",
    data=csv_bytes,
    file_name=csv_filename if csv_filename.endswith(".csv") else f"{csv_filename}.csv",
    mime="text/csv"
)

# --- Save Trained Model ---
model_filename = st.text_input("Enter filename for model:", "rf_model.joblib")

# Convert trained model to bytes (in-memory)
import io
model_buffer = io.BytesIO()
joblib.dump(model, model_buffer)
model_buffer.seek(0)

st.download_button(
    label="⬇️ Download Trained Model",
    data=model_buffer,
    file_name=model_filename if model_filename.endswith(".joblib") else f"{model_filename}.joblib",
    mime="application/octet-stream"
)