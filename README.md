# Sentiment Analysis Using Text Embeddings

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

This repository contains a comprehensive machine learning pipeline designed to perform sentiment analysis on textual data. Instead of relying on traditional bag-of-words or TF-IDF approaches, this project leverages advanced **Large Language Model (LLM) text embeddings** to capture the deep semantic meaning and context of the text. 

By converting unstructured text into dense vector representations, the project successfully trains and evaluates classification models to categorize text into distinct sentiments (e.g., Positive, Negative, Neutral).

## 🚀 How It Works (Project Workflow)

1. **Data Ingestion & Preprocessing:** 
   Raw text data is loaded, cleaned, and explored using Exploratory Data Analysis (EDA) techniques to understand class distribution and text characteristics.
2. **Embedding Generation:** 
   Text data is passed through an embedding model (e.g., Google Gemini Text Embeddings) to transform sentences into high-dimensional numerical vectors. Custom batching and rate-limit handling are implemented to ensure smooth API interactions.
3. **Model Training:** 
   The generated embeddings serve as feature inputs for machine learning classifiers. The project establishes a baseline using linear models (like Logistic Regression) and compares it against more complex, non-linear models (like XGBoost).
4. **Evaluation & Inference:** 
   The models are evaluated using accuracy scores, classification reports, and confusion matrices. A final inference pipeline allows for testing new, unseen text inputs.

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Environment:** Jupyter Notebook / Google Colab
* **Text Embeddings / API:** Google GenAI SDK (`models/gemini-embedding-001`)
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`, `wordcloud`

## 📁 Repository Structure

```text
Sentiment_Analysis-_Using_Text_Embeddings/
│
├── Sentiment_Analysis__Using_Text_Embeddings_001.ipynb   # Main Jupyter Notebook containing the full pipeline
└── README.md                                             # Project documentation
```

⚙️ Setup and Installation
To run this project locally, follow these steps:

Clone the repository:
git clone [https://github.com/TharunRam08/Sentiment_Analysis-_Using_Text_Embeddings.git](https://github.com/TharunRam08/Sentiment_Analysis-_Using_Text_Embeddings.git)
cd Sentiment_Analysis-_Using_Text_Embeddings

Create a virtual environment (Optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

Install required dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud google-genai xgboost jupyter

Configure API Keys:
Open the Jupyter Notebook and locate the configuration section. Replace the placeholder with your actual API key for generating embeddings.
API_KEY = "your_api_key_here"


💻 Usage
Launch Jupyter Notebook in your terminal:
jupyter notebook
Open Sentiment_Analysis__Using_Text_Embeddings_001.ipynb.
Run the cells sequentially to execute the data processing, embedding generation, and model training phases.
Use the final inference cells to test the trained model on custom string inputs.

📊 Results & Key Findings

Semantic Understanding: Text embeddings successfully captured nuances in language that traditional keyword-based methods often miss.
Model Performance: The project compares the stability of linear classification against tree-based boosting, detailing the accuracy and trade-offs of each approach when working with high-dimensional embedding spaces.
