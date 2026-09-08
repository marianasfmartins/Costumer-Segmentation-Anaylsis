import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
import os
import ast
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder



warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CLUSTER_NAMES_MAP = {
    0: "The Plant-Based Consumer",
    1: "The Family Budget Optimizer",
    2: "The Demanding Premium Tech Consumer",
    3: "The Early Morning Hygiene Consumer",
    4: "The Wellness Customer",
    5: "The Affluent Health-Conscious Buyer"
}


def load_and_prepare_data(basket_path=os.path.join(SCRIPT_DIR, 'customer_basket.csv'), clusters_path=os.path.join(SCRIPT_DIR, 'dataset_clusters.csv')):
    basket = pd.read_csv(basket_path)
    clusters = pd.read_csv(clusters_path)
    
    # Check if 'customer_id' is missing in clusters
    if 'customer_id' not in clusters.columns:
        info_path = os.path.join(SCRIPT_DIR, 'customer_info.csv')
        if os.path.exists(info_path):
            customer_info = pd.read_csv(info_path, usecols=['customer_id'])
            clusters['customer_id'] = customer_info['customer_id']
            
    # Map cluster numbers to names only if they are not already mapped
    if clusters['cluster'].dropna().iloc[0] not in CLUSTER_NAMES_MAP.values():
        clusters['cluster'] = clusters['cluster'].map(CLUSTER_NAMES_MAP)
        
    return pd.merge(clusters, basket, on='customer_id', how='inner')


def convert_string_to_list(string):
    return ast.literal_eval(string)


def calculate_metric_range(rules_df, metric='lift'):
    if rules_df.empty:
        return f"{metric} Range: N/A (No rules generated)"
    min_val = rules_df[metric].min()
    max_val = rules_df[metric].max()
    return f"{metric} Range: {max_val - min_val:.4f} (Min: {min_val:.4f}, Max: {max_val:.4f})"


def plot_top_products(basket_list, title_suffix=""):
    te = TransactionEncoder()
    te_ary = te.fit(basket_list).transform(basket_list)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
    top_items = df_trans.sum().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 5))
    top_items.plot(kind='bar', color='#ff7f00', edgecolor='black')
    plt.title(f'Top 10 Most Frequent Products - {title_suffix}', fontweight='bold')
    plt.ylabel('Absolute Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filename = os.path.join(SCRIPT_DIR, f"top_products_{title_suffix.lower().replace(' ', '_')}.png")
    plt.savefig(filename)
    plt.close()


def plot_association_results(rules_df, title_suffix=""):
    if rules_df.empty:
        return
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(rules_df['support'], rules_df['confidence'], 
                          color='#ff7f00', alpha=0.6)
    plt.title(f'Rules Scatter Plot (Support vs Confidence) - {title_suffix}', fontweight='bold')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    filename = os.path.join(SCRIPT_DIR, f"rules_scatter_{title_suffix.lower().replace(' ', '_')}.png")
    plt.savefig(filename)
    plt.close()


def run_association_rules(basket_data, min_support=0.005, min_confidence=0.05, split_train=True):
    if split_train:
        train_size = int(len(basket_data) * 0.8)
        data_to_model = basket_data[:train_size]
    else:
        data_to_model = basket_data
        
    te = TransactionEncoder()
    te_fit = te.fit(data_to_model).transform(data_to_model)
    transactions_items = pd.DataFrame(te_fit, columns=te.columns_)
    
    frequent_itemsets = apriori(transactions_items, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame()
        
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(by='lift', ascending=False).reset_index(drop=True)
    return frequent_itemsets, rules


def clean_rules_df(df_rules):
    if df_rules.empty:
        return df_rules
    df_cool = df_rules.copy()
    df_cool['antecedents'] = df_cool['antecedents'].apply(lambda x: f"[{', '.join(list(x))}]")
    df_cool['consequents'] = df_cool['consequents'].apply(lambda x: f"[{', '.join(list(x))}]")
    metrics = ['support', 'confidence', 'lift', 'leverage', 'conviction', 'zhangs_metric']
    for m in metrics:
        if m in df_cool.columns:
            df_cool[m] = df_cool[m].round(4)
    return df_cool


def clean_items_df(df_items):
    if df_items.empty:
        return df_items
    df_cool = df_items.copy()
    df_cool['itemsets'] = df_cool['itemsets'].apply(lambda x: f"[{', '.join(list(x))}]")
    df_cool['support'] = df_cool['support'].round(4)
    return df_cool


if __name__ == "__main__":
    try:
        df_basket_clusters = load_and_prepare_data(
            basket_path=os.path.join(SCRIPT_DIR, 'customer_basket.csv'), 
            clusters_path=os.path.join(SCRIPT_DIR, 'dataset_clusters.csv')
        )
        df_basket_clusters['list_of_goods_parsed'] = df_basket_clusters['list_of_goods'].apply(convert_string_to_list)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', None)

        # PART 1: Global Association Rules Analysis (All Customers Combined)

        print("\n" + "="*80)
        print("Global Association Rules Analysis (All Customers Combined)")
        print("="*80)
        all_baskets = df_basket_clusters['list_of_goods_parsed'].tolist()
        plot_top_products(all_baskets, "Global Dataset")
        
        items_all, rules_all = run_association_rules(all_baskets, min_support=0.005, min_confidence=0.05, split_train=True)
        
        print(f"\n-> [Global] Frequent Itemsets (Top 10):")
        print((clean_items_df(items_all.head(10))))
        
        print(f"\n-> [Global] Association Rules Generated (Top 15 sorted by Lift):")
        if not rules_all.empty:
            df_all_view = rules_all[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15)
            print(clean_rules_df(df_all_view))
            print(f"\n{calculate_metric_range(rules_all, 'lift')}")
            plot_association_results(rules_all, "Global Dataset")
        else:
            print("No rules generated globally for the current threshold limits.")

        # PART 2: Association Rules Analysis by Cluster 

        print("\n" + "="*80)
        print("=== PART 2: ASSOCIATION RULES ANALYSIS BY CLUSTER ===")
        print("="*80)
        
        DEFAULT_SUPPORT = 0.005
        DEFAULT_CONFIDENCE = 0.05
        
        for cluster_id, cluster_name in CLUSTER_NAMES_MAP.items():
            print("\n" + "-"*60)
            print(f"ANALYSIS FOR CLUSTER {cluster_id}: {cluster_name}")
            print("-"*60)
            
            df_sub_cluster = df_basket_clusters[df_basket_clusters['cluster'] == cluster_name]
            cluster_baskets = df_sub_cluster['list_of_goods_parsed'].tolist()
            
            print(f"Total transactions within this segment: {len(cluster_baskets)}")
            
            if len(cluster_baskets) > 0:
                plot_top_products(cluster_baskets, cluster_name)
                
                items_cl, rules_cl = run_association_rules(
                    cluster_baskets, 
                    min_support=DEFAULT_SUPPORT, 
                    min_confidence=DEFAULT_CONFIDENCE, 
                    split_train=True
                )
                
                # Auto-adjust thresholds if no rules are found for this cluster
                curr_supp = DEFAULT_SUPPORT
                curr_conf = DEFAULT_CONFIDENCE
                while rules_cl.empty and curr_supp > 0.001 and curr_conf > 0.01:
                    curr_supp = round(curr_supp - 0.001, 4)
                    curr_conf = round(curr_conf - 0.01, 4)
                    
                    items_cl, rules_cl = run_association_rules(
                        cluster_baskets, 
                        min_support=curr_supp, 
                        min_confidence=curr_conf, 
                        split_train=True
                    )
                
                print(f"\n   -> Frequent Itemsets Found (Top 10 Sample out of {len(items_cl)}):")
                if not items_cl.empty:
                    print(clean_items_df(items_cl.head(10)))
                else:
                    print(f"      No frequent itemsets passed the {DEFAULT_SUPPORT*100}% support threshold.")
                
                print(f"\n   -> Association Rules Found (Top 10 sorted by Lift out of {len(rules_cl)}):")
                if not rules_cl.empty:
                    df_cl_view = rules_cl[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10)
                    print(clean_rules_df(df_cl_view))
                    print(f"\n      {calculate_metric_range(rules_cl, 'lift')}")
                    plot_association_results(rules_cl, cluster_name)
                else:
                    print(f"      No rules generated for the thresholds (Support={DEFAULT_SUPPORT*100}%, Confidence={DEFAULT_CONFIDENCE*100}%).\n"
                          f"      Indicates independent purchasing behavior for products within this segment.")
            else:
                print(f"   -> Alert: No records found in the dataset for this cluster.")

        print("\n" + "="*80)
        print("Data Processing and Association Rules Analysis Completed Successfully!")
        print("="*80)

    except Exception as e:
        print(f"\n[Execution Error]: {e}")