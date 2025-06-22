House Price Prediction (Boston Housing Dataset)

This is my first Machine Learning project, where I predict house prices using the Boston Housing dataset from Kaggle. I used both Linear Regression and XGBoost Regressor to build and compare models.


---

📁 Project Structure

📦 house-price-prediction/
├── .vscode/                    # VS Code settings (optional)
├── static/                    # CSS files for styling
├── templates/                 # HTML templates (Flask)
├── .gitignore                 # Files to ignore in Git
├── House Price Prediction.pkl # Saved ML model (Pickle file)
├── HousePricePrediction.py    # Model training & evaluation code
├── app.py                     # Flask web app
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation


---

🔗 Dataset

Source: Boston Housing Data – Kaggle
Link: https://www.kaggle.com/datasets/arunjathari/bostonhousepricedata


It includes:

Features like crime rate, rooms per dwelling, tax rate, etc.

Target: Median value of owner-occupied homes


---

🧠 Machine Learning Models

Linear Regression – Simple baseline model

XGBoost Regressor – More powerful, tree-based ensemble model


---

🛠️ Technologies Used

Python

Pandas, NumPy, Matplotlib, Seaborn

Scikit-learn

XGBoost

Flask (for web interface)

HTML & CSS (in /templates and /static)


---

📈 Result: Models are trained and evaluated using R² Score and MAE.
