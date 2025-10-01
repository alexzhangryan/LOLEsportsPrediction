# LoL Esports Win Prediction

![Python](https://img.shields.io/badge/Python-3.9-blue) 
![PyTorch](https://img.shields.io/badge/PyTorch-ML-red) 
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-GradientBoost-green) 
![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen) 
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-lightgrey)  

---

## Overview
This project builds an **end-to-end machine learning pipeline** to predict the outcomes of professional *League of Legends* esports matches.  
- Dataset: **100,000+ datapoints** of historical pro matches.  
- Models: **Gradient Boosted Decision Tree (XGBoost)** and a **Neural Network (PyTorch)**.  
- Frontend: **Interactive Streamlit app** where users can input team matchups and simulate tournament outcomes.  

The project demonstrates skills in **data preprocessing, feature engineering, machine learning model development, and full-stack deployment**.

---

## Results
- **Gradient Boosted Decision Tree (XGBoost): 69% accuracy on unseen matchups**  
- **Neural Network (PyTorch): 68% accuracy**  
- Key Features Driving Prediction:  
  - Team Elo Rating
  - Individual Player and Team Elo Differential
  - Side Advantage (Blue vs Red)
  - Region Strength Calculation
  - Player Performance Metrics  

---

## Tech Stack
- **Languages**: Python  
- **Libraries**: Pandas, NumPy, Scikit-Learn, PyTorch, XGBoost, Matplotlib  
- **Frontend**: Streamlit  
- **Tools**: GitHub, Jupyter, VS Code  

---

## Demo
**[Live Streamlit App](https://lolesportsprediction.streamlit.app/)**  
---

## Installation

1. Clone the repository  
   ```bash
   git clone https://github.com/alexzhangryan/LOLEsportsRandomForest
   cd LOLEsportsRandomForest

2. Install dependencies

   ```bash
   pip install -r requirements.txt

3. Run the Streamlit app
   
   ```bash
   streamlit run frontend.py

4. Open in browser: http://localhost:8502
