# ml-model-metrics
# Machine Learning Classification Assignment

## Problem Statement

This project implements and compares six different machine learning classification algorithms on the Breast Cancer Wisconsin dataset. The goal is to predict whether a breast mass is malignant (cancerous) or benign (non-cancerous) based on various features computed from digitized images of fine needle aspirate (FNA) of breast masses.

The project includes:
- Implementation of 6 classification models
- Comprehensive evaluation using multiple metrics
- Interactive Streamlit web application for predictions
- Model comparison and performance analysis

## Dataset Description

**Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset

**Source:** UCI Machine Learning Repository / sklearn.datasets

**Dataset Characteristics:**
- **Number of Instances:** 569
- **Number of Features:** 30 (all numeric, real-valued)
- **Target Variable:** Binary classification (0 = Malignant, 1 = Benign)
- **Class Distribution:** 
  - Benign (1): 357 instances (62.7%)
  - Malignant (0): 212 instances (37.3%)

**Feature Information:**

The dataset contains 30 features computed from digitized images. For each cell nucleus, ten real-valued features are computed:

1. **Radius** - mean of distances from center to points on the perimeter
2. **Texture** - standard deviation of gray-scale values
3. **Perimeter** - perimeter of the nucleus
4. **Area** - area of the nucleus
5. **Smoothness** - local variation in radius lengths
6. **Compactness** - perimeter² / area - 1.0
7. **Concavity** - severity of concave portions of the contour
8. **Concave points** - number of concave portions of the contour
9. **Symmetry** - symmetry of the nucleus
10. **Fractal dimension** - "coastline approximation" - 1

For each of these 10 features, three values are computed:
- Mean
- Standard Error
- "Worst" (mean of the three largest values)

This results in 30 features total (10 features × 3 measurements).

**Train-Test Split:** 80% training (455 instances), 20% testing (114 instances)

## Models Used

### Model Performance Comparison

<table>
<thead>
  <tr>
    <th>ML Model Name</th>
    <th>Accuracy</th>
    <th>AUC</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1</th>
    <th>MCC</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Logistic Regression</td>
    <td>0.9825</td>
    <td>0.9987</td>
    <td>0.9859</td>
    <td>0.9859</td>
    <td>0.9859</td>
    <td>0.9623</td>
  </tr>
  <tr>
    <td>Decision Tree</td>
    <td>0.9298</td>
    <td>0.9214</td>
    <td>0.9577</td>
    <td>0.9577</td>
    <td>0.9577</td>
    <td>0.8489</td>
  </tr>
  <tr>
    <td>K-Nearest Neighbor</td>
    <td>0.9649</td>
    <td>0.9936</td>
    <td>0.9718</td>
    <td>0.9718</td>
    <td>0.9718</td>
    <td>0.9238</td>
  </tr>
  <tr>
    <td>Naive Bayes</td>
    <td>0.9561</td>
    <td>0.9949</td>
    <td>0.9577</td>
    <td>0.9859</td>
    <td>0.9716</td>
    <td>0.9046</td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td>0.9649</td>
    <td>0.9962</td>
    <td>0.9859</td>
    <td>0.9577</td>
    <td>0.9716</td>
    <td>0.9238</td>
  </tr>
  <tr>
    <td>XGBoost</td>
    <td>0.9737</td>
    <td>0.9962</td>
    <td>0.9859</td>
    <td>0.9718</td>
    <td>0.9788</td>
    <td>0.9430</td>
  </tr>
</tbody>
</table>

### Model Performance Observations

<table>
<thead>
  <tr>
    <th>ML Model Name</th>
    <th>Observation about model performance</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Logistic Regression</td>
    <td><strong>Best overall performance</strong> with highest accuracy (98.25%), AUC (99.87%), and MCC (0.9623). Excellent balance between precision and recall. The model performs exceptionally well on this linearly separable dataset, demonstrating that the features have strong linear relationships with the target variable.</td>
  </tr>
  <tr>
    <td>Decision Tree</td>
    <td><strong>Lowest performance</strong> among all models with accuracy of 92.98% and MCC of 0.8489. The model shows signs of overfitting despite max_depth constraint. Decision trees are sensitive to small variations in data and may not generalize as well as ensemble methods. However, it offers high interpretability.</td>
  </tr>
  <tr>
    <td>K-Nearest Neighbor</td>
    <td><strong>Strong performance</strong> with 96.49% accuracy and high AUC (99.36%). The model benefits from feature scaling and works well with the dataset's structure. Performance is consistent across all metrics, indicating reliable predictions. The choice of k=5 neighbors provides good balance between bias and variance.</td>
  </tr>
  <tr>
    <td>Naive Bayes</td>
    <td><strong>Good performance</strong> with 95.61% accuracy and highest recall (98.59%), making it excellent for minimizing false negatives. Despite its simplistic assumption of feature independence, it performs remarkably well. The high AUC (99.49%) indicates excellent ranking ability. Best choice when minimizing missed cancer cases is critical.</td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td><strong>Excellent ensemble performance</strong> with 96.49% accuracy and very high AUC (99.62%). The ensemble approach reduces overfitting compared to single decision tree. Highest precision (98.59%) among non-logistic models makes it reliable for positive predictions. Provides good feature importance insights.</td>
  </tr>
  <tr>
    <td>XGBoost</td>
    <td><strong>Second-best performance</strong> with 97.37% accuracy and strong balance across all metrics. The gradient boosting approach provides robust predictions with MCC of 0.9430. Excellent F1 score (97.88%) indicates optimal balance between precision and recall. Slightly better than Random Forest due to sequential learning and regularization.</td>
  </tr>
</tbody>
</table>

### Key Insights:

1. **Best Model:** Logistic Regression achieves the highest performance, suggesting the dataset has strong linear separability.

2. **Ensemble Advantage:** Both Random Forest and XGBoost outperform the single Decision Tree significantly, demonstrating the power of ensemble methods.

3. **Feature Scaling Impact:** Models requiring scaling (Logistic Regression, KNN, Naive Bayes) all perform well, indicating proper preprocessing.

4. **Clinical Relevance:** Naive Bayes has the highest recall (98.59%), making it valuable in medical contexts where missing a malignant case is more costly than false alarms.

5. **Consistency:** All models achieve >92% accuracy, indicating the dataset's features are highly predictive of breast cancer diagnosis.

## Project Structure

```
ML-assignment-final/
│
├── app.py                          # Streamlit web application (ONLY .py file)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── model/                          # Model files directory
│   ├── train_models.ipynb         # Jupyter notebook for training models
│   ├── model_logistic_regression.pkl
│   ├── model_decision_tree.pkl
│   ├── model_k-nearest_neighbor.pkl
│   ├── model_naive_bayes.pkl
│   ├── model_random_forest.pkl
│   ├── model_xgboost.pkl
│   └── scaler.pkl                 # StandardScaler for preprocessing
│
├── train_data.csv                 # Training dataset with labels
├── test_data.csv                  # Test dataset with labels
├── test_data_without_labels.csv   # Test dataset for predictions only
└── model_results.csv              # Model comparison results
```

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd ML-assignment-final
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Train the models (optional - pre-trained models included):
```bash
jupyter notebook model/train_models.ipynb
```
Run all cells to train models and generate evaluation metrics.

4. Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Streamlit App Features

The interactive web application includes:

1. **Dataset Upload Option** 📁
   - Upload CSV files with test data
   - Supports files with or without target labels
   - Automatic data validation with error reporting

2. **Model Selection Dropdown** 📊
   - Choose from 6 trained classification models
   - Easy switching between models for comparison

3. **Display of Evaluation Metrics** 📈
   - Accuracy, AUC, Precision, Recall, F1 Score, MCC
   - Visual metric cards for easy interpretation
   - Only displayed when target labels are provided

4. **Confusion Matrix & Classification Report** 🔲
   - Visual confusion matrix heatmap
   - Detailed classification report with per-class metrics
   - Available when ground truth labels are provided

5. **Additional Features:**
   - Data preview and shape information
   - Validation error detection and downloadable error reports
   - Prediction results with probability scores
   - Downloadable prediction outputs as CSV
   - Feature list reference for proper data formatting

## Usage Instructions

### For Predictions Only:

1. Prepare a CSV file with 30 features (no target column)
2. Upload the file using the sidebar
3. Select a model from the dropdown
4. View predictions and download results

### For Model Evaluation:

1. Prepare a CSV file with 30 features + 'target' column
2. Upload the file using the sidebar
3. Select a model from the dropdown
4. View predictions, metrics, confusion matrix, and classification report
5. Download predictions and evaluation results

### Expected CSV Format:

```csv
mean radius,mean texture,mean perimeter,...,worst fractal dimension
17.99,10.38,122.8,...,0.1189
20.57,17.77,132.9,...,0.08902
...
```

With labels (optional):
```csv
mean radius,mean texture,mean perimeter,...,worst fractal dimension,target
17.99,10.38,122.8,...,0.1189,0
20.57,17.77,132.9,...,0.08902,0
...
```

## Deployment

### Streamlit Community Cloud Deployment

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository, branch, and `app.py`
6. Click "Deploy"

Your app will be live at: `https://[your-app-name].streamlit.app`
for this project the url is: https://ml-model-metrics-6aobkdhdiecqneyyoi2mpm.streamlit.app/

## Technologies Used

- **Python 3.8+**
- **scikit-learn** - Machine learning models and evaluation
- **XGBoost** - Gradient boosting classifier
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization

## Model Training Details

- **Algorithm Implementations:** Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, Random Forest, XGBoost
- **Preprocessing:** StandardScaler for feature normalization (applied to LR, KNN, NB)
- **Train-Test Split:** 80-20 stratified split
- **Random State:** 42 (for reproducibility)
- **Evaluation Metrics:** Accuracy, AUC-ROC, Precision, Recall, F1 Score, Matthews Correlation Coefficient

## Results Summary

All six models demonstrate strong performance on the Breast Cancer Wisconsin dataset, with accuracy ranging from 92.98% to 98.25%. Logistic Regression emerges as the top performer, while ensemble methods (Random Forest and XGBoost) show significant improvement over the single Decision Tree classifier.

The high performance across all models indicates that the dataset features are highly discriminative for breast cancer diagnosis, making this a well-suited problem for machine learning classification.

## Author

**Yogesh Tolani**

## License

This project is created for educational purposes as part of a Machine Learning course assignment.

## Acknowledgments

- UCI Machine Learning Repository for the Breast Cancer Wisconsin dataset
- scikit-learn documentation and community
- Streamlit for the excellent web framework
