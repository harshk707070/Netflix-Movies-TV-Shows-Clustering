# 🎬 Netflix Movies and TV Shows Clustering

## 📌 Project Overview
This project applies **unsupervised machine learning** techniques to analyze and segment Netflix movies and TV shows based on their **textual descriptions and metadata**.  
The objective is to uncover hidden content patterns that can support **recommendation systems, personalization, catalog organization, and content strategy decisions**.

The project demonstrates a **complete end-to-end unsupervised ML pipeline**, starting from data understanding and EDA to feature engineering, NLP processing, clustering model implementation, and business-oriented insights.

---

## 🧠 Problem Statement
Netflix hosts a vast and diverse content catalog spanning multiple genres, countries, formats, and release periods.  
However, the absence of labeled categories makes it challenging to understand **natural similarities and groupings** within the content library.

This project aims to:
- Discover meaningful content clusters without predefined labels
- Identify patterns across movies and TV shows
- Support strategic decisions such as content recommendation, acquisition, and personalization

---

## 🛠️ Tech Stack & Tools Used
- **Programming Language:** Python  
- **Libraries:**
  - pandas, numpy
  - matplotlib, seaborn
  - scipy
  - scikit-learn
  - nltk
- **Techniques:**
  - Exploratory Data Analysis (EDA)
  - Feature Engineering
  - NLP (Text Cleaning, Lemmatization)
  - TF-IDF Vectorization
  - Unsupervised Clustering

---

## 📂 Dataset Information
- **Source:** Netflix Movies and TV Shows Dataset  
- **Rows:** 7,787  
- **Columns:** 12  
- **Key Features:**
  - Categorical: type, rating, country, listed_in
  - Textual: description, title
  - Numerical: release_year, duration

Missing metadata values were handled carefully to preserve dataset integrity for clustering.

---

## 🔍 Exploratory Data Analysis (EDA)
- Content distribution across Movies and TV Shows
- Year-wise content addition trends
- Country-wise content production
- Rating and duration patterns
- Relationship analysis using univariate, bivariate, and multivariate visualizations

More than **15 meaningful visualizations** were created following the **UBM rule**:
- Univariate
- Bivariate
- Multivariate

---

## ⚙️ Feature Engineering & NLP
- Text preprocessing on `description` column:
  - Lowercasing
  - Noise removal
  - Stopword removal
  - Lemmatization
- Text vectorization using **TF-IDF**
- Encoding categorical variables with business-aware strategies
- Feature scaling using **StandardScaler**
- Final feature matrix created by combining:
  - TF-IDF vectors
  - Scaled numerical and encoded features

---

## 🤖 Machine Learning Models Implemented

### ✅ Model 1: KMeans Clustering (Final Model)
- Produced **4 stable and interpretable clusters**
- Evaluation metric: **Silhouette Score (~0.25)**
- Captured:
  - Movies vs TV Shows
  - Modern vs classic content
  - Mainstream vs niche titles
- Best balance of scalability, interpretability, and business usability

### 🔍 Model 2: Hierarchical Clustering
- Validated KMeans results
- Produced similar cluster structures
- Useful for exploratory validation
- Less scalable for production use

### ⚠️ Model 3: DBSCAN
- Classified all points as noise
- Not suitable for high-dimensional sparse TF-IDF data
- Helped justify correct model selection

---

## 📊 Model Evaluation Strategy
Since this is an **unsupervised learning problem**, traditional accuracy-based metrics were not used.

Instead, evaluation focused on:
- Silhouette Score
- Cluster stability
- Cluster size distribution
- Business interpretability

These metrics ensure clusters are both **mathematically meaningful and practically actionable**.

---

## 📈 Key Insights
- Netflix content naturally segments into distinct groups based on format, duration, and temporal patterns
- TV Shows and Movies follow fundamentally different structural behaviors
- Certain clusters represent niche or archival content, which is valuable for targeted recommendations
- Textual features play a dominant role in defining content similarity

---

## 🏁 Conclusion
- **KMeans Clustering** is the most suitable model for this problem due to its stability, scalability, and interpretability
- Hierarchical Clustering confirms the robustness of the segmentation
- DBSCAN highlights the limitations of density-based methods on sparse NLP features
- The project successfully converts unstructured content data into **actionable business insights**

---

## 🚀 Future Scope
- Integrate user watch history for hybrid recommendation systems
- Apply advanced embeddings (Word2Vec / BERT)
- Deploy clusters via a web app (Streamlit / Flask)
- Combine with supervised learning for personalized recommendations

---

## ▶️ How to Run the Project
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
