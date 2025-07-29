
# 🔮 AI Salary Oracle – Indian Edition 🇮🇳

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-00639C?style=for-the-badge&logo=xgboost)

A modern web app built with **Streamlit** to predict employee salaries tailored to **Indian industry standards** using an advanced XGBoost model. Built to demonstrate end-to-end ML deployment — from preprocessing to real-time prediction.

---

### 🚀 [Live Demo](https://employe-salary-prediction-using-ml-4xk5frvaqtl3fuvwn5w5cu.streamlit.app/)

![App Screenshot](https://github.com/Ayush03A/Employe-Salary-Prediction-Using-ML/blob/5e98565e81625cc63ca5c2e816d74e0e844d259a/Screenshots/Website.png)

---

## ✨ Features

- 💎 **Beautiful UI** using Glassmorphism + custom CSS  
- 📊 **Smart Salary Prediction** using `XGBoost`  
- 🧠 **Realistic Salary Ranges** instead of flat numbers  
- 📈 **Model Evaluation Section** with performance plots  
- 🎥 **Lottie Animations** for better UX  
- 📦 **Easily Deployable** on Streamlit Cloud  

---

## 🛠️ Tech Stack

- **Python 3.11**
- **Streamlit** for Web App
- **XGBoost**, **Scikit-learn**
- **Pandas**, **Joblib**
- **StandardScaler**, **LabelEncoder**
- **Lottie JSON** animations

---

## 🧪 Model Insights

- ✅ **Algorithm:** XGBoost Regressor  
- 📈 **R² Score:** 94.58% (on test data)  
- 🔢 **Input Features:**
  - Age
  - Gender
  - Education Level
  - Job Title
  - Years of Experience

- 🧪 Preprocessing:
  - `LabelEncoder` for categorical
  - `StandardScaler` for numerical
  - Saved as `salary_predictor_indian.pkl` using `joblib`

---

## 🧑‍💻 How to Run Locally

```bash
git clone https://github.com/yourusername/salary-oracle-india.git
cd salary-oracle-india
python -m venv venv
source venv/bin/activate    # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

---

## 👨‍🎓 Author

**Dinesh M**  
[🔗 LinkedIn Profile](https://www.linkedin.com/in/dinesh-m-2a4480245)
