import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🎗️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🎗️ Breast Cancer Classification System")
st.markdown("### Machine Learning Models for Cancer Diagnosis")
st.markdown("---")

# Constants
EXPECTED_FEATURES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

MODEL_FILES = {
    'Logistic Regression': 'model/model_logistic_regression.pkl',
    'Decision Tree': 'model/model_decision_tree.pkl',
    'K-Nearest Neighbor': 'model/model_k-nearest_neighbor.pkl',
    'Naive Bayes': 'model/model_naive_bayes.pkl',
    'Random Forest': 'model/model_random_forest.pkl',
    'XGBoost': 'model/model_xgboost.pkl'
}

MODELS_REQUIRING_SCALING = ['Logistic Regression', 'K-Nearest Neighbor', 'Naive Bayes']

# Sidebar
st.sidebar.header("📊 Model Configuration")
st.sidebar.markdown("Select a machine learning model for breast cancer classification.")

selected_model = st.sidebar.selectbox(
    "Choose a Model:",
    list(MODEL_FILES.keys())
)

# Load model function
@st.cache_resource
def load_model(model_path):
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load scaler
@st.cache_resource
def load_scaler():
    try:
        if os.path.exists('model/scaler.pkl'):
            with open('model/scaler.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading scaler: {str(e)}")
        return None

# Load model comparison results
@st.cache_data
def load_comparison_results():
    try:
        if os.path.exists('model_results.csv'):
            return pd.read_csv('model_results.csv')
        else:
            return None
    except Exception as e:
        st.error(f"Error loading comparison results: {str(e)}")
        return None

def validate_data(df):
    errors = []
    
    if df.shape[1] != len(EXPECTED_FEATURES):
        errors.append(f"Expected {len(EXPECTED_FEATURES)} features, got {df.shape[1]}")
    
    missing_features = set(EXPECTED_FEATURES) - set(df.columns)
    if missing_features:
        errors.append(f"Missing features: {', '.join(list(missing_features)[:5])}...")
    
    extra_features = set(df.columns) - set(EXPECTED_FEATURES)
    if extra_features:
        errors.append(f"Extra features found: {', '.join(list(extra_features)[:5])}...")
    
    for col in df.columns:
        if col in EXPECTED_FEATURES and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must be numeric")
    
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        errors.append(f"Missing values found in columns: {', '.join(null_cols[:5])}...")
    
    return errors

def evaluate_predictions(y_true, y_pred, y_pred_proba=None):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'F1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0
    
    return metrics

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload & Predict", "📈 Model Metrics", "📊 Model Comparison"])

# Tab 1: Upload and Predict
with tab1:
    st.header("Upload Test Dataset")
    st.markdown("Upload a CSV file containing breast cancer patient data for prediction.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.success(f"✓ File uploaded successfully! Shape: {df.shape}")

            with st.expander("📋 View Data Preview"):
                st.dataframe(df.head(10), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())

            st.markdown("---")
            st.subheader("🔄 Data Preprocessing & Prediction")

            if st.button("Run Prediction", type="primary"):
                with st.spinner("Processing data and making predictions..."):
                    has_labels = 'target' in df.columns
                    
                    if has_labels:
                        X = df.drop('target', axis=1)
                        y_true = df['target']
                    else:
                        X = df
                        y_true = None

                    validation_errors = validate_data(X)

                    if validation_errors:
                        st.error("❌ **Data Validation Errors:**")
                        for error in validation_errors:
                            st.error(f"• {error}")
                        
                        error_df = pd.DataFrame({'Validation Errors': validation_errors})
                        csv_errors = error_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Validation Errors",
                            data=csv_errors,
                            file_name="validation_errors.csv",
                            mime="text/csv"
                        )
                    else:
                        model = load_model(MODEL_FILES[selected_model])
                        scaler = load_scaler()

                        if model is None:
                            st.error("❌ Model not found. Please run the training notebook first.")
                        else:
                            X_processed = X.copy()
                            if selected_model in MODELS_REQUIRING_SCALING and scaler is not None:
                                X_processed = pd.DataFrame(
                                    scaler.transform(X_processed),
                                    columns=X_processed.columns
                                )

                            predictions = model.predict(X_processed)
                            
                            if hasattr(model, 'predict_proba'):
                                pred_proba = model.predict_proba(X_processed)
                            else:
                                pred_proba = None

                            st.success("✓ Predictions completed!")

                            results_df = pd.DataFrame({
                                'Prediction': predictions,
                                'Confidence': pred_proba.max(axis=1) if pred_proba is not None else 'N/A'
                            })

                            if has_labels:
                                results_df.insert(0, 'Actual', y_true.values)

                            st.dataframe(results_df, use_container_width=True)

                            csv_output = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions",
                                data=csv_output,
                                file_name=f"predictions_{selected_model.replace(' ', '_').lower()}.csv",
                                mime="text/csv"
                            )

                            st.subheader("📊 Prediction Distribution")
                            fig, ax = plt.subplots(figsize=(8, 4))
                            pred_counts = pd.Series(predictions).value_counts().sort_index()
                            bars = ax.bar(pred_counts.index, pred_counts.values, color=['#FF6B6B', '#4ECDC4'])
                            ax.set_xlabel('Class (0=Malignant, 1=Benign)')
                            ax.set_ylabel('Count')
                            ax.set_title('Distribution of Predictions')
                            ax.set_xticks([0, 1])
                            
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                       f'{int(height)}',
                                       ha='center', va='bottom')
                            
                            st.pyplot(fig)
                            plt.close()

                            if has_labels and y_true is not None:
                                st.markdown("---")
                                st.subheader("📈 Evaluation Metrics")

                                pred_proba_binary = pred_proba[:, 1] if pred_proba is not None else None
                                metrics = evaluate_predictions(y_true, predictions, pred_proba_binary)

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                                    st.metric("Precision", f"{metrics['Precision']:.4f}")

                                with col2:
                                    st.metric("Recall", f"{metrics['Recall']:.4f}")
                                    st.metric("F1 Score", f"{metrics['F1']:.4f}")

                                with col3:
                                    st.metric("AUC Score", f"{metrics['AUC']:.4f}")
                                    st.metric("MCC Score", f"{metrics['MCC']:.4f}")

                                st.markdown("---")
                                st.subheader("🔢 Confusion Matrix")

                                cm = confusion_matrix(y_true, predictions)
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                           xticklabels=['Malignant', 'Benign'],
                                           yticklabels=['Malignant', 'Benign'])
                                ax.set_xlabel('Predicted')
                                ax.set_ylabel('Actual')
                                ax.set_title(f'Confusion Matrix - {selected_model}')
                                st.pyplot(fig)
                                plt.close()

                                st.markdown("---")
                                st.subheader("📋 Classification Report")

                                report = classification_report(y_true, predictions, 
                                                              target_names=['Malignant', 'Benign'],
                                                              output_dict=True, zero_division=0)
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(
                                    report_df.style.background_gradient(cmap='RdYlGn', 
                                                                       subset=['precision', 'recall', 'f1-score']),
                                    use_container_width=True
                                )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            error_df = pd.DataFrame({'Error': [str(e)]})
            csv_errors = error_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Error Details",
                data=csv_errors,
                file_name="error_log.csv",
                mime="text/csv"
            )
    else:
        st.info("👆 Please upload a CSV file to begin prediction.")

        with st.expander("📄 Expected CSV Format"):
            st.markdown(f"""
            The CSV file should contain **{len(EXPECTED_FEATURES)} columns** with the following features:
            """)
            
            features_df = pd.DataFrame({
                'Feature Name': EXPECTED_FEATURES[:10]
            })
            st.dataframe(features_df, use_container_width=True)
            st.markdown("... and 20 more features")
            
            st.markdown("""
            **Optional:** Include a 'target' column with binary labels:
            - 0 = Malignant (cancerous)
            - 1 = Benign (non-cancerous)
            """)

# Tab 2: Model Metrics
with tab2:
    st.header(f"Model Performance: {selected_model}")

    comparison_df = load_comparison_results()

    if comparison_df is not None:
        model_key = selected_model.split()[0]
        model_metrics = comparison_df[comparison_df['Model'].str.contains(model_key, case=False, na=False)]

        if not model_metrics.empty:
            st.subheader("📊 Performance Metrics")

            metrics_row = model_metrics.iloc[0]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Accuracy", f"{metrics_row['Accuracy']:.4f}")
                st.metric("Precision", f"{metrics_row['Precision']:.4f}")

            with col2:
                st.metric("Recall", f"{metrics_row['Recall']:.4f}")
                st.metric("F1 Score", f"{metrics_row['F1']:.4f}")

            with col3:
                st.metric("AUC Score", f"{metrics_row['AUC']:.4f}")
                st.metric("MCC Score", f"{metrics_row['MCC']:.4f}")

            st.markdown("---")
            st.subheader("📈 Metrics Visualization")

            metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1']
            values = [metrics_row[m] for m in metrics_to_plot]

            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            bars = ax.bar(metrics_to_plot, values, color=colors)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Score')
            ax.set_title(f'{selected_model} - Performance Metrics')

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom')

            st.pyplot(fig)
            plt.close()

        else:
            st.warning("⚠️ No metrics found for the selected model.")
    else:
        st.error("❌ Model comparison results not found. Please run the training notebook first.")

# Tab 3: Model Comparison
with tab3:
    st.header("All Models Comparison")

    comparison_df = load_comparison_results()

    if comparison_df is not None:
        st.subheader("📊 Comparison Table")

        st.dataframe(
            comparison_df.style.background_gradient(cmap='RdYlGn', 
                                                   subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']),
            use_container_width=True
        )

        best_model_idx = comparison_df['Accuracy'].idxmax()
        best_model = comparison_df.loc[best_model_idx, 'Model']
        best_accuracy = comparison_df.loc[best_model_idx, 'Accuracy']

        st.success(f"🏆 Best Model: **{best_model}** with Accuracy: **{best_accuracy:.4f}**")

        st.markdown("---")
        st.subheader("📈 Visual Comparison")

        metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#DDA15E']

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            bars = ax.bar(range(len(comparison_df)), comparison_df[metric], color=colors[idx], alpha=0.7)
            ax.set_xticks(range(len(comparison_df)))
            ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Score')
            ax.set_title(metric)
            ax.set_ylim(0, 1)

            best_idx = comparison_df[metric].idxmax()
            bars[best_idx].set_color(colors[idx])
            bars[best_idx].set_alpha(1.0)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.subheader("🕸️ Radar Chart Comparison")

        categories = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1']
        N = len(categories)

        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        for idx, row in comparison_df.iterrows():
            values = row[categories].tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Comparison (Radar Chart)', size=16, y=1.05)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        st.pyplot(fig)
        plt.close()

    else:
        st.error("❌ Model comparison results not found. Please run the training notebook first.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Breast Cancer Classification System | Machine Learning Assignment</p>
        <p>Built with Streamlit 🚀 | Dataset: Breast Cancer Wisconsin (Diagnostic)</p>
    </div>
    """, unsafe_allow_html=True)
