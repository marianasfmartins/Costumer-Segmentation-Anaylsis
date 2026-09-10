# Customer Segmentation & Market Basket Analysis

A reproducible project that segments customers and discovers product association rules from transactional and customer data to uncover business insights for targeted marketing, merchandising, and cross-sell opportunities.

Key results included in this repository:
- Unsupervised segmentation (K-Means, Hierarchical, DBSCAN, SOM, MeanShift) to produce customer personas.
- Market-basket analysis using Apriori / association rules to find frequent itemsets and product recommendations.
- Notebooks and helper modules to reproduce preprocessing, modeling, visualizations, and exports.

---

## Table of contents
- [What's inside](#whats-inside)
- [Quickstart](#quickstart)
- [How to run the analyses](#how-to-run-the-analyses)
- [Files & code overview](#files--code-overview)
- [Data description](#data-description)
- [Reproducible pipeline (recommended order)](#reproducible-pipeline-recommended-order)
- [Notable implementation details & tips](#notable-implementation-details--tips)
- [Authors](#authors)
- [License](#license)

---

## What's inside
- Notebooks for exploration, model selection, and the final pipeline:
  - `NB1_DATAEXPLORATION.ipynb` — EDA, cleaning, and feature engineering.
  - `NB2 MODELSELECTION.ipynb` — experiments and comparisons across clustering algorithms.
  - `NB3 FINALNOTEBOOK.ipynb` — final pipeline, profiling of clusters and outputs.
- Python modules (reusable code):
  - `Preprocessing.py` — imputation, log-transform of spend/prop features, and three scaler helpers (Standard, Robust, MinMax). Also `cluster_analysis()` to prepare unscaled data for profiling.
  - `Clustering.py` — utilities to find optimal K, fit KMeans/Hierarchical/DBSCAN/MeanShift, a simple SOM implementation, and UMAP/t-SNE plotting helpers.
  - `AssociationRules.py` — loads baskets, maps clusters to friendly names, runs apriori + association_rules, generates plots (top products and rules scatter) and saves outputs.
- Data files:
  - `customer_info.csv`, `customer_info_engineered.csv` — customer demographic/behavior features.
  - `customer_basket.csv` — transaction baskets (list of goods per customer).
  - `dataset_clusters.csv` — final dataset with assigned cluster labels (used to join to baskets).
- Visual outputs and models are expected to be saved under an `outputs/` directory (visualizations, models, reports).

---

## Quickstart

Recommended installation (create a virtual environment first):

```bash
# Clone
git clone https://github.com/marianasfmartins/Costumer-Segmentation-Anaylsis.git
cd Costumer-Segmentation-Anaylsis

# Create a virtualenv (example)
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate    # Windows

# Install core dependencies
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn umap-learn mlxtend scipy jupyterlab
```

Open the notebooks:
```bash
jupyter lab
# then open NB1_DATAEXPLORATION.ipynb and follow the notebooks in order
```

Run the association rules script directly (it has a CLI-style __main__):
```bash
python AssociationRules.py
```
This will read `customer_basket.csv` and `dataset_clusters.csv`, generate plots (saved as PNGs) and print summarized rule tables to the console.

---

## How to run the analyses

1. Prepare data:
   - If you plan to reproduce exactly, start with `NB1_DATAEXPLORATION.ipynb` to run the cleaning and feature-engineering steps that create `customer_info_engineered.csv`.
   - Alternatively, use functions in `Preprocessing.py` to run programmatically.

2. Clustering:
   - Use `Clustering.find_optimal_k()` on the preprocessed numeric dataset (scaled) to choose K.
   - Use `Clustering.fit_kmeans()` or `Clustering.fit_hierarchical()` to fit models and get cluster profiles.
   - Use `plot_umap()` or `plot_tsne()` to visualize clusters in 2D.

3. Market-basket (association rules):
   - Ensure `dataset_clusters.csv` includes `customer_id` and `cluster` (cluster labels).
   - Run `python AssociationRules.py`. The script maps cluster integers to friendly names (see `CLUSTER_NAMES_MAP`), saves top-products plots and rules scatterplots, and prints frequent itemsets and association rules.

---

## Files & code overview

Top-level files:
- `NB1_DATAEXPLORATION.ipynb` — exploratory analysis, cleaning, and feature engineering.
- `NB2 MODELSELECTION.ipynb` — model selection and evaluation notebooks.
- `NB3 FINALNOTEBOOK.ipynb` — final results and profiling.
- `Preprocessing.py` — preprocessing helper functions and scalers.
- `Clustering.py` — clustering algorithms, SOM implementation, and visualization helpers.
- `AssociationRules.py` — market-basket analysis pipeline and plotting utilities.
- `customer_info.csv`, `customer_info_engineered.csv`, `customer_basket.csv`, `dataset_clusters.csv` — datasets used throughout the notebooks and scripts.

How it fits together:
- The notebooks and scripts use the raw data files to produce engineered data (`customer_info_engineered.csv`) and a final dataset with cluster assignments (`dataset_clusters.csv`).
- Clustering modules operate on scaled numeric features to produce cluster labels and cluster profiles.
- Association rules are generated per-cluster by joining cluster assignments with transaction baskets; plots and rule summaries are saved to disk.

---

## Data description

- customer_info.csv: customer-level attributes (demographics, loyalty, aggregated spend/behavior features).
- customer_info_engineered.csv: engineered features produced during preprocessing (log transforms, proportions, imputed values).
- customer_basket.csv: transaction-level data with a column containing a list of goods per transaction or per customer.
- dataset_clusters.csv: dataset containing customers and their assigned cluster label (used to segment baskets in AssociationRules.py).

Note: Check dataset headers to confirm column names expected by the scripts:
- `customer_id` — used to join datasets.
- `list_of_goods` — in `customer_basket.csv`, expected to be a string representation of a list; `AssociationRules.py` parses it using ast.literal_eval.

---

## Reproducible pipeline (recommended order)

1. Run NB1_DATAEXPLORATION.ipynb to clean and produce engineered data.
2. Run NB2 MODELSELECTION.ipynb to find the best clustering approach and parameters.
3. Run NB3 FINALNOTEBOOK.ipynb to produce `dataset_clusters.csv`.
4. Run `python AssociationRules.py` to produce frequent itemsets, association rules, and visualization PNGs.

---

## Notable implementation details & tips

- Preprocessing:
  - `Preprocessing.py` uses KNN imputation (scikit-learn's KNNImputer) for numeric features and applies log1p to lifetime spend and proportion columns before scaling.
  - Three scaler variants are provided: StandardScaler, RobustScaler, and MinMaxScaler. Choose based on distribution and outlier sensitivity.

- Clustering:
  - `Clustering.py` includes helper functions to evaluate K (inertia and silhouette), fit KMeans and hierarchical clustering, find eps for DBSCAN using k-distance graphs, and fit MeanShift and a basic SOM implementation.
  - Use UMAP or t-SNE for visualization of cluster separations.

- Association Rules:
  - `AssociationRules.py` uses mlxtend (TransactionEncoder, apriori, association_rules).
  - The script automatically lowers support/confidence thresholds for small clusters if no rules are found (within limits).
  - Outputs: PNGs for top products per cluster and rules scatter plots; printed top itemsets and rules.

- Reproducibility:
  - Many functions set `random_state=42`; if you need different seeds, search for `random_state` and update as required.

---

## Try asking
- How do I modify `AssociationRules.py` to run rules only on purchases within the last 6 months?
- Which columns does `Preprocessing.cluster_analysis()` expect to exist before calling — can you list them from `customer_info_engineered.csv`?
- Can you add a small script to take `dataset_clusters.csv` and produce a single CSV with cluster-level aggregate metrics (avg spend, top product categories)?

---

## Authors

Developed as part of the Machine Learning II curriculum by Joana Martins, Maiara Almada and Mariana Martins.

---

## License

This project is provided for educational purposes as part of the Machine Learning II course.
