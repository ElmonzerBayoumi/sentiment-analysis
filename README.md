
#  Sentiment Analysis for Arabic and English App Reviews

A multilingual sentiment analysis project using Machine Learning to classify user comments (Arabic and English) from a mobile banking app into Positive, Negative categories.

---

##  Overview

This project applies Natural Language Processing (NLP) and supervised machine learning algorithms to analyze and predict user sentiment from real-world app reviews. The dataset contains mixed-language (Arabic and English) text data, making the project both linguistically and technically challenging.

We trained and compared three ML models:
-  **Support Vector Machine (SVM)** — Best performer
-  **Logistic Regression**
-  **Multinomial Naive Bayes**

---

##  Objectives

- Clean and preprocess multilingual text data
- Perform sentiment classification using multiple models
- Evaluate and compare model performance
- Visualize class distributions and important words
- Build a simple prediction system for custom input

---

##  Tools & Libraries

- Python, Google Colab
- NLTK, Scikit-learn, Pandas, Matplotlib, Seaborn
- Arabic NLP: ISRIStemmer, custom stopword filtering


---

##  Example Output

<img src="images/Sentiment Distribution.png" width="400"/>
<img src="images/Confusion Matrix Comparison for Models.png" width="400"/>

---
##  Workflow Summary

1. **Data Loading & Exploration**
2. **Text Cleaning & Preprocessing**
   - Remove noise, digits, emojis
   - Normalize Arabic text
   - Tokenize and remove stopwords
   - Apply stemming (Arabic & English)
3. **Feature Extraction**
   - `TfidfVectorizer` (max 5000 features)
4. **Model Training & Tuning**
   - Grid Search with Cross-Validation
5. **Model Evaluation**
   - Accuracy, F1-Score, Confusion Matrix
6. **Visualization**
   - Sentiment distribution
   - Word cloud per sentiment class
7. **Interactive Prediction**
   - Input a custom comment to predict its sentiment using SVM

---

##  Results Summary

| Model                | Accuracy | Macro F1 Score |
|--------------------- |----------|----------------|
| **SVM**              | 97%      | 0.97           |
| Logistic Regression  | ~97%     | 0.96           |
| Multinomial NB       | ~96%     | 0.95           |

>  **SVM outperformed all other models** in both accuracy and F1-score.

---

## Real-Time Sentiment Prediction (SVM)

Try it yourself at the bottom of the notebook!


**Input**: "خدمه تعبانه تعب الموت ومافي فايده"  
**Prediction**: `Positive `

**Input**: "The app is too slow and crashes a lot"  
**Prediction**: `Negative `

---



## How to Use

1. Clone this repo or open in Colab
2. Run all cells step-by-step
3. Scroll to the end to try real-time sentiment prediction
4. Enter your own Arabic or English comment and see the result!

---

##  Future Improvements

- Use deep learning (LSTM or BERT multilingual)
- Deploy as a web app (e.g., using Streamlit or Flask)
- Build dashboard for insights from feedback

---
## Acknowledgements
I would like to sincerely thank my colleagues Duaa Khalil and Tasabeeh Alabas from the University of Khartoum for their assistance in labeling the dataset and supporting various stages of the project development was instrumental to its success.

##  Author

**ُ Elmonzer Bayoumi **  
 www.linkedin.com/in/elmonzer-bayoumi  
Aspiring Data Scientist | Python & NLP Enthusiast

