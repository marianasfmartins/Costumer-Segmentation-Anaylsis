# Customer Segmentation and Market Basket Analysis 🛒🤖

This repository contains the final project for Machine Learning II. The project focuses on understanding customer purchasing behavior through advanced data mining techniques, specifically Clustering and Association Rules. By analyzing customer baskets and demographic info, we identified distinct customer personas and discovered hidden patterns in their buying habits.

## 🎯 Project Objectives

1. **Customer Segmentation (Clustering)**: Group customers into distinct segments based on their demographics, behavior, and purchasing history to allow for targeted marketing strategies.
2. **Market Basket Analysis (Association Rules)**: Discover associations between different product categories to optimize cross-selling, promotions, and store layout strategies.

## 📂 Repository Structure

The project is structured into three main Jupyter Notebooks and supporting Python modules:

### 📓 Notebooks

* **`NB1_DATAEXPLORATION.ipynb`**: Comprehensive Exploratory Data Analysis (EDA). Includes data cleaning, univariate/bivariate analysis, feature engineering and initial insights into customer behavior. Warning: This notebook was removed from GitHub because its file size exceeds the 25 MB limit.
* **`NB2_MODELSELECTION.ipynb`**: The experimental core. This notebook details the process of training and evaluating various clustering algorithms (K-Means, Hierarchical Clustering, MeanShift, DBSCAM and SOM).
* **`NB3_FINALNOTEBOOK.ipynb`**: The final polished pipeline. It presents the final chosen models, deep dives into the profiling of the generated customer segments, and extracts actionable business insights using Association Rules for each specific segment.

### 🐍 Python Scripts

* **`Preprocessing.py`**: Functions for data cleaning, treating outliers and scaling.
* **`Clustering.py`**: Encapsulates the logic for the final clustering models and dimensionality reduction visualizations (UMAP / t-SNE).
* **`AssociationRules.py`**: Contains the logic to run the Apriori algorithm, filter rules by lift/confidence, and generate scatter plots and top product visualizations.

## 📊 Data

* **`customer_info.csv`** / **`customer_info_engineered.csv`**: Demographic and behavioral data.
* **`customer_basket.csv`**: Transactional data showing what products were bought together.
* **`dataset_clusters.csv`**: The final dataset with the assigned cluster labels for each customer.

## 🧑‍🤝‍🧑 Customer Segments Identified

Through our clustering models, we successfully identified **6 unique customer personas**:

1. **🌱 The Plant-Based Consumer**: Highly focused on vegetarian and vegan alternatives.
2. **💰 The Family Budget Optimizer**: Price-sensitive shoppers buying in bulk for larger households.
3. **💻 The Demanding Premium Tech Consumer**: High spenders focused on premium brands and electronics.
4. **🚿 The Early Morning Hygiene Consumer**: Consistent buyers of personal care and hygiene products, usually early in the day.
5. **🧘 The Wellness Customer**: Focused on supplements, organic foods, and general well-being.
6. **🥂 The Affluent Health-Conscious Buyer**: Premium shoppers with a strong preference for high-quality, organic, and health-oriented products.


## 🔍 Folder Structure

```
ml-customer-segmentation/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── NB1_DATAEXPLORATION.ipynb
│   ├── NB2_MODELSELECTION.ipynb
│   └── NB3_FINALNOTEBOOK.ipynb
│
├── src/
│   ├── Preprocessing.py
│   ├── Clustering.py
│   └── AssociationRules.py
│
├── data/
│   ├── customer_info.csv
│   ├── customer_info_engineered.csv
│   ├── customer_basket.csv
│   └── dataset_clusters.csv
│
├── outputs/
│   ├── visualizations/
│   ├── models/
│   └── reports/
│
└── .gitignore
```

## 👥 Author

Developed as part of the **Machine Learning II** curriculum by Joana Martins, Maiara Almada and Mariana Martins.


## 📝 License

This project is provided for educational purposes as part of the Machine Learning II course.



