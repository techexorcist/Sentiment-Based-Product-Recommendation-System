# 🛒 Ebuss: Sentiment-Based Product Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.3.2-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-green.svg)](https://xgboost.ai/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-blueviolet.svg)](https://www.nltk.org/)
[![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5.svg)](https://spacy.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458.svg)](https://pandas.pydata.org/)

> **Author:** Vinodh Nagarajaiah  
> **Programme:** AI/ML Executive Programme (UpGrad & IIIT-B)

## ⏱️ Executive Summary (TL;DR)
* **The Goal:** Enhance the e-commerce shopping experience for 'Ebuss' by building a recommendation system that not only suggests relevant products but ensures those products have high positive user sentiment.
* **The Data:** Analysed an extensive dataset of 30,000 user reviews (`sample30.csv`) across various product categories.
* **The Process:** Engineered a dual-engine AI pipeline: First, a **Collaborative Filtering** recommendation model to identify top products for a user. Second, an **NLP Classification Model** (using XGBoost and SMOTE) to analyse review text and filter recommendations based on predicted positive sentiment.
* **The Result:** Successfully deployed a robust User-User collaborative filtering system integrated with a highly accurate sentiment classifier (F1-Score: 0.94), capable of delivering the top 5 highly-rated, contextually relevant products for any given user.

---

## 📖 Table of Contents
1. [Business Problem & Objective](#-business-problem--objective)
2. [Skills & AI Competencies](#-skills--ai-competencies)
3. [Methodology: The Dual-Engine Pipeline](#-methodology-the-dual-engine-pipeline)
4. [Key Insights & Model Evaluation](#-key-insights--model-evaluation)
5. [Steps to Run the Project](#-steps-to-run-the-project)
6. [Repository Structure](#-repository-structure)
7. [Acknowledgements & Contact](#-acknowledgements--contact)

---

## 💼 Business Problem & Objective
**Ebuss** is a rapidly growing e-commerce company selling a vast array of products, from household essentials to electronics. To compete with market leaders like Amazon and Flipkart, Ebuss must rapidly scale its customer engagement and retention strategies. 

**The Core Objective:** Traditional recommendation engines suggest products based purely on user or item similarities, completely ignoring whether the recommended product is actually *good* or *bad* based on user feedback. The goal of this capstone is to build a **Sentiment-Based Product Recommendation System**. The system must first recommend the top 20 products for a user, then predict the sentiment of the reviews for those products, and finally filter out the top 5 products with the highest percentage of positive reviews.

---

## 🛠️ Skills & AI Competencies
* **Natural Language Processing (NLP):** Text cleansing, tokenisation, lemmatisation (using `spaCy`), stop-word removal, and TF-IDF vectorisation.
* **Recommendation Engines:** Building and evaluating **User-User** and **Item-Item Collaborative Filtering** models.
* **Imbalanced Data Handling:** Resolving severe class imbalances in user reviews (89% Positive vs 11% Negative) using **SMOTE** (Synthetic Minority Over-sampling Technique).
* **Advanced Classification Modelling:** Training and hyperparameter tuning of Logistic Regression, Random Forest, LightGBM, and **XGBoost** models.
* **Statistical Diagnostics:** Evaluating models using RMSE, Precision, Recall, F1-Score, and AUC.

---

## 🧠 Methodology: The Dual-Engine Pipeline

### 1. Data Ingestion & NLP Pre-processing
* Loaded the raw `sample30.csv` dataset and conducted thorough Exploratory Data Analysis (EDA) using word clouds, radar charts, and tree maps.
* Executed advanced text pre-processing: converted text to lowercase, removed punctuation, stripped stop-words using `NLTK`, and applied lemmatisation via `spaCy` to standardise the vocabulary.
* Transformed the cleaned textual reviews into a numerical format using **TF-IDF** (Term Frequency-Inverse Document Frequency).

### 2. Handling Class Imbalance
* Identified a massive skew towards positive sentiments.
* Evaluated multiple techniques (ADASYN, Random Oversampling, NearMiss, etc.) and selected **SMOTE**. SMOTE perfectly balanced the dataset by synthesising minority class samples, achieving an optimal balance between Precision (0.90) and Recall (0.87) while preventing model overfitting.

### 3. Sentiment Classification Modelling
* Trained multiple machine learning algorithms on the TF-IDF vectors to classify sentiments.
* Applied Hyperparameter Tuning (Randomised Search & Grid Search) to optimise the models. 
* **XGBoost (Randomised Search)** emerged as the champion classifier for interpreting the sentiment of text reviews.

### 4. Collaborative Filtering Recommendation Engine
* Built baseline recommendation matrices for both **User-User** and **Item-Item** collaborative filtering.
* Evaluated both architectures using **Root Mean Square Error (RMSE)**. The User-User recommendation system yielded the lowest error margin and was selected as the foundational engine.

---

## 📊 Key Insights & Model Evaluation

### The Champion Sentiment Model: XGBoost (HPT RS)
The hyperparameter-tuned XGBoost model delivered exceptional performance in distinguishing between positive and negative reviews:
* **Accuracy:** 0.90
* **Recall (Sensitivity):** 0.94
* **Precision:** 0.94
* **F1-Score:** 0.94

### The Final Recommendation Pipeline
When a user (e.g., `"charlie"`) is queried, the system performs the following sequence automatically:
1. The **User-User Collaborative Filter** identifies the top 20 most relevant products based on the purchasing behaviour of similar users.
2. The **XGBoost Classifier** scans the historical text reviews for those 20 products and predicts the sentiment of each review.
3. The system calculates the percentage of positive sentiments for each product and outputs the **Top 5** items boasting the highest positive sentiment ratios (e.g., *Stargate Ultimate Edition*, *Cars Toon: Mater's Tall Tales*, etc.).

---

## 🚀 Steps to Run the Project
1. **Environment Setup:** Ensure Python 3.8+ is installed. Install all required dependencies by referencing the imports in the notebook (e.g., `pip install spacy xgboost imbalanced-learn`).
2. **Download Models:** Ensure the pre-trained `.pkl` files (located in the `models/` directory) are in your working path to bypass lengthy training times.
3. **Execute Notebook:** Open `capstone_sentiment_based_product_recommendation_system.ipynb` via Jupyter or Google Colab.
4. **Data Sourcing:** Ensure `sample30.csv` is in the correct root directory.
5. **Run Pipeline:** Execute the cells sequentially. You can test the final engine by inputting a valid username into the final recommendation function to see the top 5 sentiment-filtered products.

---

## 📁 Repository Structure

    ├── capstone_sentiment_based_product_recommendation_system.ipynb   # Main Notebook (EDA, NLP, Modelling, RS)
    ├── sample30.csv                                                   # Raw e-commerce reviews dataset
    ├── models/                                                        # Directory containing trained model weights
    │   ├── xgboost_base_model.pkl                                     
    │   ├── xg_boost_hpt_rs.pkl                                        # Champion Sentiment Model
    │   ├── lr_base_model_hpt_gs.pkl                                   
    │   └── lgbm_base_model.pkl                                        
    └── README.md                                                      # Project overview and insights

---

## 🎓 Acknowledgements & Contact
This project is an assessment capstone designed and integrated into the AI/ML Programme at **UpGrad**, in collaboration with **IIIT-B**. 

**Created by:** Vinodh Nagarajaiah  

* 💼 **LinkedIn:** [vinodh-nagarajaiah](https://www.linkedin.com/in/vinodh-nagarajaiah/)
* 🐙 **GitHub:** [@techexorcist](https://github.com/techexorcist)
* ✉️ **Email:** [vinodh.nagarajaiah@gmail.com](mailto:vinodh.nagarajaiah@gmail.com)

<br>

> **Disclaimer:** *The dataset used in this project is for educational purposes only. All personally identifiable information (PII) has been removed or anonymised.*

---

## 📜 Licence
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - see the LICENSE file for details.
