# Tweet Sentiment Analysis using BERT and VADER

## 📌 Project Overview

This repository contains a sentiment analysis project on tweets and social media product reviews using advanced natural language processing (NLP) techniques. The analysis involves preprocessing the data, performing sentiment classification using two distinct methods:

- **BERT (Bidirectional Encoder Representations from Transformers)**
- **VADER (Valence Aware Dictionary and sEntiment Reasoner)**

## 🎯 Objectives

- Develop a robust sentiment classification pipeline
- Evaluate and compare the performance of the BERT and VADER sentiment analyzers
- Achieve a high accuracy in predicting tweet sentiments

## 🗃️ Dataset

The dataset utilized in this project comprises tweets and social media texts, labeled with sentiment polarity:

- **Positive**
- **Negative**
- **Neutral**

The dataset is stored under the [`dataset/`](./dataset) directory.

## 🚀 Technologies & Tools

- Python 3.x
- Jupyter Notebook
- PyTorch
- Hugging Face Transformers
- NLTK
- Pandas
- Scikit-Learn
- Git & GitHub

## 🛠️ Methodology

### 1. **Data Cleaning and Preprocessing**

The text data is cleaned and preprocessed with:
- Lowercasing
- URL and HTML tag removal
- Removal of punctuation and special characters
- Stopword removal
- Tokenization and stemming

### 2. **Sentiment Analysis**

- **BERT-based Classifier:**
  - Utilizes Hugging Face's Transformers library
  - Fine-tuning a pre-trained BERT model for sentiment classification

- **VADER Sentiment Analyzer:**
  - Lexicon-based sentiment analysis
  - Calculation of compound polarity scores to classify tweets

### 3. **Evaluation Metrics**

- Accuracy Score
- Classification Report
- Compound Polarity Scores (VADER)

## 📊 Results

- **BERT Model Accuracy:** ~92%
- **VADER Model Accuracy:** ~72.7%

BERT provided superior performance due to its deep contextual understanding capabilities.

## 📂 Repository Structure

```
tweet-sentiment-analysis/
├── dataset/
│   └── sentimentdataset.csv
├── sentiment-analysis.ipynb         # Jupyter Notebook implementation
├── sentiment-analysis.py            # Python script (Exported)
├── requirements.txt                 # Project dependencies
└── README.md
```

## 🚩 How to Use

### Step 1: Clone the Repository

```bash
git clone https://github.com/abdulazeemsikander/tweet-sentiment-analysis.git
cd tweet-sentiment-analysis
```

### Step 2: Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Notebook

Open the `sentiment-analysis.ipynb` file in Jupyter Notebook to run the analysis interactively.

```bash
jupyter notebook
```

Alternatively, run the Python script directly:

```bash
python sentiment-analysis.py
```

## 📌 References

- [Original Kaggle Notebook](https://www.kaggle.com/code/alkidiarete/social-media-analysis-sentiment/notebook)
- [Hugging Face Transformers](https://huggingface.co/)
- [NLTK Documentation](https://www.nltk.org/)

## 📢 Contributing

Feel free to fork the repository, create pull requests, and suggest improvements or enhancements.

## 📜 License

This project is open-source and available under the MIT License.

