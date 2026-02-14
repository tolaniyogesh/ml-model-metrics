# Step-by-Step Instructions for Assignment Submission

## 📋 Overview
This guide will help you complete and submit your ML Assignment 2 successfully.

## 🔧 Step 1: Train the Models

Open and run the Jupyter notebook:

```bash
jupyter notebook model/train_models.ipynb
```

Then run all cells in the notebook.

**Expected Output Files:**
- `model/model_logistic_regression.pkl`
- `model/model_decision_tree.pkl`
- `model/model_k-nearest_neighbor.pkl`
- `model/model_naive_bayes.pkl`
- `model/model_random_forest.pkl`
- `model/model_xgboost.pkl`
- `model/scaler.pkl`
- `train_data.csv` (training data with labels)
- `test_data.csv` (test data with labels)
- `test_data_without_labels.csv` (test data for predictions only)
- `model_results.csv`

## 🧪 Step 2: Test the Streamlit App Locally

```bash
streamlit run app.py
```

The app should open at `http://localhost:8501`

**Test the app:**
1. Upload `test_data_without_labels.csv` - should show predictions only
2. Upload `test_data.csv` (with labels) - should show predictions + evaluation metrics
3. Try different models from the dropdown
4. Download predictions and verify the CSV output

## 📦 Step 3: Prepare for GitHub

### Create a GitHub Repository

1. Go to https://github.com
2. Click "New Repository"
3. Name it: `ml-classification-assignment` (or any name you prefer)
4. Make it **Public**
5. **DO NOT** initialize with README (we already have one)
6. Click "Create Repository"

### Upload Files to GitHub

**Option A: Using Git Command Line**
```bash
cd "c:\Users\yogesh.tolani\Downloads\ML assignment final"
git init
git add .
git commit -m "Initial commit: ML classification assignment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

**Option B: Using GitHub Desktop**
1. Download GitHub Desktop
2. File → Add Local Repository
3. Select your project folder
4. Commit all files
5. Publish repository

**Option C: Manual Upload via Web**
1. Go to your repository on GitHub
2. Click "uploading an existing file"
3. Drag and drop all files/folders
4. Commit changes

## 🚀 Step 4: Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click "New app"
4. Fill in the details:
   - **Repository:** Select your GitHub repository
   - **Branch:** main
   - **Main file path:** app.py
5. Click "Deploy"
6. Wait 2-5 minutes for deployment
7. Your app will be live at: `https://YOUR-APP-NAME.streamlit.app`

**Important:** Make sure all model files are committed to GitHub before deploying!

## 📄 Step 5: Create the PDF Submission

Create a PDF document with the following content in this exact order:

### 1. GitHub Repository Link
```
GitHub Repository: https://github.com/YOUR_USERNAME/YOUR_REPO_NAME
```

### 2. Live Streamlit App Link
```
Live App: https://YOUR-APP-NAME.streamlit.app
```

### 3. Screenshot
- Take a screenshot showing the assignment execution on BITS Virtual Lab
- Insert the screenshot in the PDF

### 4. README Content
Copy the entire content from `README.md` file and paste it in the PDF. This includes:
- Problem Statement
- Dataset Description
- Models Used (with comparison table)
- Model Performance Observations (with observations table)
- All other sections from README

## ✅ Step 6: Final Checklist

Before submitting, verify:

- [ ] All 6 models are trained and saved in `model/` folder
- [ ] CSV files generated: `train_data.csv`, `test_data.csv`, `test_data_without_labels.csv`
- [ ] Streamlit app runs locally without errors
- [ ] All files are pushed to GitHub
- [ ] GitHub repository is PUBLIC
- [ ] README.md is visible on GitHub
- [ ] Streamlit app is deployed and accessible
- [ ] App loads without errors when you click the link
- [ ] All 4 required features work in the app:
  - [ ] Dataset upload option
  - [ ] Model selection dropdown
  - [ ] Display of evaluation metrics
  - [ ] Confusion matrix/classification report
- [ ] PDF contains all required elements in correct order:
  - [ ] GitHub link
  - [ ] Streamlit app link
  - [ ] BITS Lab screenshot
  - [ ] Complete README content
- [ ] PDF is properly formatted and readable

## 🎯 Step 7: Submit on Taxila

1. Go to Taxila LMS
2. Navigate to ML Assignment 2
3. Upload your PDF file
4. Click **SUBMIT** (not save as draft)
5. Verify submission is successful

## 📊 Expected Marks Breakdown

- **Model Implementation (10 marks)**
  - Dataset description: 1 mark
  - Comparison table with metrics: 6 marks (1 per model)
  - Observations on performance: 3 marks

- **Streamlit App (4 marks)**
  - Dataset upload option: 1 mark
  - Model selection dropdown: 1 mark
  - Display of evaluation metrics: 1 mark
  - Confusion matrix/classification report: 1 mark

- **BITS Lab Screenshot (1 mark)**

**Total: 15 marks**

## 🆘 Troubleshooting

### Streamlit Deployment Fails
- Check `requirements.txt` has all dependencies
- Ensure all model `.pkl` files are in the repository
- Check app.py for any hardcoded paths
- View deployment logs on Streamlit Cloud

### Models Not Loading in App
- Verify model files are in `model/` folder
- Check file names match exactly in `app.py`
- Ensure `scaler.pkl` is present

### GitHub Push Issues
- Make sure you're in the correct directory
- Check if Git is installed: `git --version`
- Verify remote URL: `git remote -v`

### App Shows Errors
- Test locally first: `streamlit run app.py`
- Check if all required packages are installed
- Verify CSV format matches expected features

## 📞 Need Help?

If you encounter issues:
1. Check the error message carefully
2. Verify all files are in correct locations
3. Test each component individually
4. Review the deployment logs on Streamlit Cloud

Good luck with your assignment! 🎓
