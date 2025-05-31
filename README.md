sd#  News Sentiment Analysis – Task 1 : Git and Github

This repository contains the setup and exploratory analysis for **Week 1 – Predicting Price Moves with News Sentiment** challenge. The focus is on configuring version control, setting up a Python development environment, automating CI/CD workflows, and performing Exploratory Data Analysis (EDA) on financial news headlines.


##  Project Structure
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows
│       ├── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
├── notebooks/
│   ├── __init__.py
│   └── README.md
├── tests/
│   ├── __init__.py
└── scripts/
    ├── __init__.py
    └── README.md

##  Objectives (Based on Task 1, completing the following activities)

 ✅ Setup Git and GitHub with proper branching.
 
 ✅ Establish a clean Python environment.
 
 ✅ Configure automated testing using GitHub Actions.
 
 ✅ Perform EDA on financial news data.


## Key Analysis Performed
### Descriptive Statistics
- Headline length distributions
- Article count per publisher
- Temporal trends in article frequency

### Text Analysis (Topic Modeling)
- Keyword extraction
- Topic identification using NLP

### Time Series Analysis
- News volume over time
- Publishing time patterns

### Publisher Analysis
- Top publishers by frequency
- Domain analysis of publisher emails

## Tech Stack : using the following Tools, Softwares, Libraries and Framework.
- Python 3.13
- Jupyter
- Pandas, NumPy, Seaborn, Matplotlib
- scikit-learn, NLTK
- Git, GitHub Actions

## Development Setup
# Create virtual environment : 
We have used Windows with Python 3.13

python -m venv .venv

.venv\Scripts\activate 

# Install dependencies
pip install -r requirements.txt