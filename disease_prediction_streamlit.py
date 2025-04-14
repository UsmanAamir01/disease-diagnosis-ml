import streamlit as st
import pandas as pd
import numpy as np
import ast
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler  
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title("TF-IDF vs One-Hot Encoding")

@st.cache_data
def load_data():
    features = pd.read_csv('disease_features.csv')
    labels = pd.read_csv('encoded_output2.csv')
    return features, labels

features_df, labels_df = load_data()

st.header("TF-IDF Feature Extraction")
for col in ['Risk Factors', 'Symptoms', 'Signs']:
    features_df[col] = features_df[col].apply(ast.literal_eval).apply(lambda lst: ' '.join(lst))
    
tfidf_risk = TfidfVectorizer().fit(features_df['Risk Factors'])
tfidf_sym = TfidfVectorizer().fit(features_df['Symptoms'])
tfidf_sign = TfidfVectorizer().fit(features_df['Signs'])

tfidf_risk_mat = tfidf_risk.transform(features_df['Risk Factors'])
tfidf_sym_mat = tfidf_sym.transform(features_df['Symptoms'])
tfidf_sign_mat = tfidf_sign.transform(features_df['Signs'])
tfidf_matrix = sparse.hstack([tfidf_risk_mat, tfidf_sym_mat, tfidf_sign_mat])

onehot_df = labels_df.drop(columns=['Disease'])
onehot_matrix = onehot_df.values

st.write("TF-IDF matrix shape:", tfidf_matrix.shape)
st.write("One-Hot matrix shape:", onehot_matrix.shape)
tfidf_sparsity = 1 - tfidf_matrix.count_nonzero() / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])
onehot_sparsity = 1 - np.count_nonzero(onehot_matrix) / (onehot_matrix.shape[0] * onehot_matrix.shape[1])
st.write(f"TF-IDF sparsity: {tfidf_sparsity:.4f}")
st.write(f"One-Hot sparsity: {onehot_sparsity:.4f}")

st.header("Dimensionality Reduction")
tfidf_dense = tfidf_matrix.toarray()
onehot_dense = onehot_matrix
y = features_df['Disease']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

pca_tfidf = PCA(n_components=2, random_state=42).fit(tfidf_dense)
X_pca_tfidf = pca_tfidf.transform(tfidf_dense)
exp_pca_tfidf = pca_tfidf.explained_variance_ratio_

pca_onehot = PCA(n_components=2, random_state=42).fit(onehot_dense)
X_pca_onehot = pca_onehot.transform(onehot_dense)
exp_pca_onehot = pca_onehot.explained_variance_ratio_

svd_tfidf = TruncatedSVD(n_components=2, random_state=42).fit(tfidf_matrix)
X_svd_tfidf = svd_tfidf.transform(tfidf_matrix)
exp_svd_tfidf = svd_tfidf.explained_variance_ratio_

svd_onehot = TruncatedSVD(n_components=2, random_state=42).fit(onehot_dense)
X_svd_onehot = svd_onehot.transform(onehot_dense)
exp_svd_onehot = svd_onehot.explained_variance_ratio_

exp_df = pd.DataFrame({
    'Method': ['PCA TF-IDF', 'PCA One-Hot', 'SVD TF-IDF', 'SVD One-Hot'],
    'Component 1': [exp_pca_tfidf[0], exp_pca_onehot[0], exp_svd_tfidf[0], exp_svd_onehot[0]],
    'Component 2': [exp_pca_tfidf[1], exp_pca_onehot[1], exp_svd_tfidf[1], exp_svd_onehot[1]],
})
st.subheader("Explained Variance Ratio")
st.table(exp_df)

category_map = {
    'Acute Coronary Syndrome': 'Cardiovascular', 
    'Aortic Dissection': 'Cardiovascular', 
    'Atrial Fibrillation': 'Cardiovascular',
    'Cardiomyopathy': 'Cardiovascular', 
    'Heart Failure': 'Cardiovascular', 
    'Hyperlipidemia': 'Cardiovascular',
    'Hypertension': 'Cardiovascular', 
    'Pulmonary Embolism': 'Cardiovascular', 
    'Adrenal Insufficiency': 'Endocrine',
    'Diabetes': 'Endocrine', 
    'Pituitary Disease': 'Endocrine', 
    'Thyroid Disease': 'Endocrine',
    'Alzheimer': 'Neurological', 
    'Epilepsy': 'Neurological', 
    'Migraine': 'Neurological',
    'Multiple Sclerosis': 'Neurological', 
    'Stroke': 'Neurological', 
    'Asthma': 'Respiratory', 
    'COPD': 'Respiratory',
    'Pneumonia': 'Infectious', 
    'Tuberculosis': 'Infectious', 
    'Gastritis': 'Gastrointestinal',
    'Gastro-oesophageal Reflux Disease': 'Gastrointestinal', 
    'Peptic Ulcer Disease': 'Gastrointestinal',
    'Upper Gastrointestinal Bleeding': 'Gastrointestinal'
}
categories = features_df['Disease'].map(category_map)
color_map = {
    'Cardiovascular': 'red', 
    'Endocrine': 'orange', 
    'Neurological': 'blue',
    'Respiratory': 'green', 
    'Infectious': 'purple', 
    'Gastrointestinal': 'brown'
}

all_x = np.concatenate([X_pca_tfidf[:,0], X_pca_onehot[:,0], X_svd_tfidf[:,0], X_svd_onehot[:,0]])
all_y = np.concatenate([X_pca_tfidf[:,1], X_pca_onehot[:,1], X_svd_tfidf[:,1], X_svd_onehot[:,1]])
xlim = (all_x.min() - 1, all_x.max() + 1)
ylim = (all_y.min() - 1, all_y.max() + 1)

def plot_scatter(X, title, xlim=None, ylim=None):
    fig, ax = plt.subplots()
    for cat, color in color_map.items():
        idx = categories == cat
        ax.scatter(X[idx, 0], X[idx, 1], label=cat, color=color)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    return fig

st.subheader("PCA Scatter Plots")
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_scatter(X_pca_tfidf, "TF-IDF PCA", xlim, ylim))
with col2:
    st.pyplot(plot_scatter(X_pca_onehot, "One-Hot PCA", xlim, ylim))

st.subheader("SVD Scatter Plots")
col3, col4 = st.columns(2)
with col3:
    st.pyplot(plot_scatter(X_svd_tfidf, "TF-IDF SVD", xlim, ylim))
with col4:
    st.pyplot(plot_scatter(X_svd_onehot, "One-Hot SVD", xlim, ylim))

def safe_cross_validate(model, X, y, cv):
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    valid_folds = 0
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_test)) < 2:
            continue
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average="macro", zero_division=0))
        recalls.append(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        valid_folds += 1

    if valid_folds > 0:
        return {
            'accuracy': np.mean(accuracies),
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1': np.mean(f1_scores)
        }
    else:
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

st.header("Classification and Custom Evaluation")
scaler = StandardScaler()
tfidf_dense_norm = scaler.fit_transform(tfidf_dense)
onehot_dense_norm = scaler.fit_transform(onehot_dense)

st.sidebar.markdown("### Model Configuration")
selected_feature = st.sidebar.selectbox("Select Feature Encoding", ['TF-IDF', 'One-Hot'])
selected_model = st.sidebar.radio("Select Model", ("KNN", "Logistic Regression"))

X_selected = tfidf_dense_norm if selected_feature == "TF-IDF" else onehot_dense_norm

if selected_model == "KNN":
    selected_k = st.sidebar.slider("Select k value", min_value=3, max_value=15, step=2, value=3)
    selected_metric = st.sidebar.selectbox("Select Distance Metric", ['euclidean', 'manhattan', 'cosine'])

desired_folds = st.sidebar.slider("Number of folds for cross-validation", min_value=2, max_value=5, value=5)
label_counts = pd.Series(y_encoded).value_counts()
min_samples = label_counts.min()

if min_samples < 2:
    fallback = True
else:
    fallback = False

if st.sidebar.button("Run Model"):
    results = []
    if fallback:
        model = KNeighborsClassifier(n_neighbors=selected_k, metric=selected_metric) if selected_model == "KNN" else LogisticRegression(max_iter=1000)
        model.fit(X_selected, y_encoded)
        y_pred = model.predict(X_selected)
        results.append({
            'Model': selected_model,
            'Feature': selected_feature,
            'k': selected_k if selected_model=="KNN" else 'N/A',
            'Metric': selected_metric if selected_model=="KNN" else 'N/A',
            'Accuracy': accuracy_score(y_encoded, y_pred),
            'Precision': precision_score(y_encoded, y_pred, average="macro", zero_division=0),
            'Recall': recall_score(y_encoded, y_pred, average="macro", zero_division=0),
            'F1-Score': f1_score(y_encoded, y_pred, average="macro", zero_division=0)
        })
    else:
        n_folds = min_samples if desired_folds > min_samples else desired_folds
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        model = KNeighborsClassifier(n_neighbors=selected_k, metric=selected_metric) if selected_model == "KNN" else LogisticRegression(max_iter=1000)
        cv_res = safe_cross_validate(model, X_selected, y_encoded, cv)
        results.append({
            'Model': selected_model,
            'Feature': selected_feature,
            'k': selected_k if selected_model=="KNN" else 'N/A',
            'Metric': selected_metric if selected_model=="KNN" else 'N/A',
            'Accuracy': cv_res['accuracy'],
            'Precision': cv_res['precision'],
            'Recall': cv_res['recall'],
            'F1-Score': cv_res['f1']
        })
    
    results_df = pd.DataFrame(results)
    st.subheader("Selected Model Performance")
    st.dataframe(results_df)

detailed_results = []
all_results = []

for encoding, X in [('TF-IDF', tfidf_dense_norm), ('One-Hot', onehot_dense_norm)]:
    for k in [3, 5, 7]:
        for metric in ['euclidean', 'manhattan', 'cosine']:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(X, y_encoded)
            y_pred = knn.predict(X)
            all_results.append({
                'Encoding': encoding,
                'Model': f'KNN (k={k}, metric={metric})',
                'Accuracy': accuracy_score(y_encoded, y_pred),
                'Precision': precision_score(y_encoded, y_pred, average="macro", zero_division=0),
                'Recall': recall_score(y_encoded, y_pred, average="macro", zero_division=0),
                'F1-score': f1_score(y_encoded, y_pred, average="macro", zero_division=0)
            })
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y_encoded)
    y_pred = lr.predict(X)
    all_results.append({
        'Encoding': encoding,
        'Model': 'Logistic Regression',
        'Accuracy': accuracy_score(y_encoded, y_pred),
        'Precision': precision_score(y_encoded, y_pred, average="macro", zero_division=0),
        'Recall': recall_score(y_encoded, y_pred, average="macro", zero_division=0),
        'F1-score': f1_score(y_encoded, y_pred, average="macro", zero_division=0)
    })

results_df_full = pd.DataFrame(all_results)
results_df_full.sort_values(by='F1-score', ascending=False, inplace=True)
results_df_full.reset_index(drop=True, inplace=True)
results_df_full.index = results_df_full.index + 1
results_df_full.insert(0, 'Rank', results_df_full.index)

st.subheader("Comprehensive Model Performance Results")
st.dataframe(results_df_full)

csv = results_df_full.to_csv(index=False).encode('utf-8')
st.download_button("Export Comprehensive Results as CSV", data=csv, file_name="comprehensive_results.csv", mime="text/csv")

st.subheader("Bar Charts: Model Performance Comparison")
avg_metrics = results_df_full.groupby("Encoding")[['Accuracy', 'Precision', 'Recall', 'F1-score']].mean().reset_index()
fig1, ax1 = plt.subplots(figsize=(8, 5))
bar_width = 0.2
x = np.arange(len(avg_metrics['Encoding']))
for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-score']):
    ax1.bar(x + i * bar_width, avg_metrics[metric], width=bar_width, label=metric)
ax1.set_xticks(x + bar_width * 1.5)
ax1.set_xticklabels(avg_metrics['Encoding'])
ax1.set_xlabel("Feature Encoding")
ax1.set_ylabel("Metric Score")
ax1.set_title("Average Performance Metrics by Feature Encoding")
ax1.legend()
st.pyplot(fig1)

st.subheader("Heatmap: Detailed Model Performance")
heatmap_data = results_df_full[['Model', 'Encoding', 'Accuracy', 'Precision', 'Recall', 'F1-score']].copy()
heatmap_data['Model-Encoding'] = heatmap_data['Encoding'] + " - " + heatmap_data['Model']
heatmap_data.set_index('Model-Encoding', inplace=True)
heatmap_data = heatmap_data[['Accuracy', 'Precision', 'Recall', 'F1-score']]
fig2, ax2 = plt.subplots(figsize=(12, max(6, len(heatmap_data) * 0.3)))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax2)
ax2.set_title("Heatmap of Model Performance Metrics")
st.pyplot(fig2)

st.subheader("Label Distribution")
st.write(label_counts.sort_index())

st.header("Disease Predictor")
st.markdown("Enter new patient information for **Risk Factors**, **Symptoms**, and **Signs** below. Note that the predictor uses the TF-IDF feature set.")

risk_input = st.text_area("Risk Factors", "Enter risk factors here...", height=100)
symptoms_input = st.text_area("Symptoms", "Enter symptoms here...", height=100)
signs_input = st.text_area("Signs", "Enter signs here...", height=100)

pred_model_choice = st.selectbox("Select Model for Prediction", ["KNN", "Logistic Regression"])
if pred_model_choice == "KNN":
    pred_k = st.slider("Select k value for prediction", min_value=3, max_value=15, step=2, value=3)
    pred_metric = st.selectbox("Select Distance Metric for prediction", ['euclidean', 'manhattan', 'cosine'])

if st.button("Train Predictor & Predict Disease"):
    new_risk = tfidf_risk.transform([risk_input])
    new_symptoms = tfidf_sym.transform([symptoms_input])
    new_signs = tfidf_sign.transform([signs_input])
    new_tfidf = sparse.hstack([new_risk, new_symptoms, new_signs])
    new_tfidf_dense = new_tfidf.toarray()
    
    tfidf_scaler = StandardScaler().fit(tfidf_dense)
    new_tfidf_norm = tfidf_scaler.transform(new_tfidf_dense)
    
    if pred_model_choice == "KNN":
        predictor = KNeighborsClassifier(n_neighbors=pred_k, metric=pred_metric)
    else:
        predictor = LogisticRegression(max_iter=1000)
    predictor.fit(tfidf_dense_norm, y_encoded)
    
    pred_encoded = predictor.predict(new_tfidf_norm)
    predicted_disease = label_encoder.inverse_transform(pred_encoded)
    st.success(f"**Predicted Disease:** {predicted_disease[0]}")
