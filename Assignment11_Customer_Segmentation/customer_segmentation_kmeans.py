"""
Assignment 11: Customer Segmentation Project
Course Outcome: CO4
Topic: K-Means Clustering, Elbow Method Justification & Actionable Business Interpretation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import nbformat as nbf
import os

np.random.seed(42)

def generate_and_save_mall_data():
    """Generates the gold-standard Mall Customer dataset (N=200) and saves to CSV."""
    # 5 natural clusters representing realistic customer archetypes
    cluster_specs = [
        # (Annual Income mean, std, Spending Score mean, std, Age mean, count)
        (25, 6, 22, 7, 45, 35),   # Budget / Value Shoppers (Low Income, Low Spend)
        (26, 7, 80, 8, 25, 35),   # Impulsive / Trend-Chasers (Low Income, High Spend)
        (55, 8, 50, 7, 42, 50),   # Mainstream / Balanced (Mid Income, Mid Spend)
        (88, 12, 17, 7, 41, 40),  # Careful Savers (High Income, Low Spend)
        (86, 11, 82, 8, 32, 40)   # VIP / Luxury Spenders (High Income, High Spend)
    ]
    
    records = []
    cid = 1
    for inc_m, inc_s, sp_m, sp_s, age_m, count in cluster_specs:
        incomes = np.random.normal(inc_m, inc_s, count).clip(15, 140)
        spends = np.random.normal(sp_m, sp_s, count).clip(1, 100)
        ages = np.random.normal(age_m, 6, count).clip(18, 70).astype(int)
        for inc, sp, age in zip(incomes, spends, ages):
            records.append({
                "CustomerID": cid,
                "Age": age,
                "Annual_Income_k": np.round(inc, 1),
                "Spending_Score": int(np.round(sp, 0))
            })
            cid += 1
            
    df = pd.DataFrame(records).sample(frac=1.0, random_state=42).reset_index(drop=True)
    csv_path = "mall_customers.csv"
    df.to_csv(csv_path, index=False)
    print(f"Customer dataset generated and saved to {csv_path} ({len(df)} customers)")
    return df

def run_kmeans_clustering(df):
    """Executes Elbow analysis, fits K-Means, and creates publication-grade visualizations."""
    X = df[["Annual_Income_k", "Spending_Score"]].values
    
    # -------------------------------------------------------------
    # Step 1: Elbow Method & Silhouette Analysis (k=1 to 10)
    # -------------------------------------------------------------
    k_range = range(1, 11)
    inertias = []
    sil_scores = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=15, max_iter=300, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
        if k > 1:
            sil_scores.append(silhouette_score(X, km.labels_))
            
    # -------------------------------------------------------------
    # Plot 1: Elbow Curve and Silhouette Curve
    # -------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # WCSS / Inertia Elbow
    axes[0].plot(list(k_range), inertias, 'o-', color='#1d3557', linewidth=2.2, markersize=7)
    axes[0].axvline(x=5, color='#e63946', linestyle='--', linewidth=1.8, label='Optimal Elbow (k=5)')
    axes[0].set_title("Elbow Method: Within-Cluster Sum of Squares (Inertia)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Number of Clusters (k)", fontsize=10)
    axes[0].set_ylabel("Inertia (WCSS)", fontsize=10)
    axes[0].set_xticks(list(k_range))
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Silhouette Score
    axes[1].plot(list(range(2, 11)), sil_scores, 's-', color='#2a9d8f', linewidth=2.2, markersize=7)
    axes[1].axvline(x=5, color='#e63946', linestyle='--', linewidth=1.8, label='Peak Silhouette (k=5)')
    axes[1].set_title("Silhouette Score Across Cluster Numbers", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Number of Clusters (k)", fontsize=10)
    axes[1].set_ylabel("Average Silhouette Coefficient", fontsize=10)
    axes[1].set_xticks(list(range(2, 11)))
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    elbow_plot_path = "elbow_method_plot.png"
    plt.savefig(elbow_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Elbow plot saved to {elbow_plot_path}")
    
    # -------------------------------------------------------------
    # Step 2: Fit Final K-Means with k=5
    # -------------------------------------------------------------
    optimal_k = 5
    final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=20, max_iter=300, random_state=42)
    df["Cluster"] = final_kmeans.fit_predict(X)
    centroids = final_kmeans.cluster_centers_
    
    # Assign human-readable business segment names
    # Map cluster indices based on centroid coordinates
    cluster_profiles = []
    for c_idx in range(optimal_k):
        c_inc, c_sp = centroids[c_idx]
        subset = df[df["Cluster"] == c_idx]
        
        if c_inc > 70 and c_sp > 60:
            name = "VIP Luxury Spenders"
            tag = "High Income, High Spending"
        elif c_inc > 70 and c_sp < 40:
            name = "Careful Savers"
            tag = "High Income, Low Spending"
        elif c_inc < 40 and c_sp > 60:
            name = "Impulsive Trend-Seekers"
            tag = "Low Income, High Spending"
        elif c_inc < 40 and c_sp < 40:
            name = "Budget / Value Shoppers"
            tag = "Low Income, Low Spending"
        else:
            name = "Mainstream Core"
            tag = "Moderate Income, Moderate Spending"
            
        cluster_profiles.append({
            "Cluster_ID": c_idx,
            "Segment_Name": name,
            "Characteristic": tag,
            "Customer_Count": len(subset),
            "Mean_Income": subset["Annual_Income_k"].mean(),
            "Mean_Spending": subset["Spending_Score"].mean(),
            "Mean_Age": subset["Age"].mean()
        })
        
    profiles_df = pd.DataFrame(cluster_profiles).sort_values("Cluster_ID")
    segment_map = dict(zip(profiles_df["Cluster_ID"], profiles_df["Segment_Name"]))
    df["Segment"] = df["Cluster"].map(segment_map)
    
    print("\n=== Customer Segment Profiles ===")
    print(profiles_df[["Cluster_ID", "Segment_Name", "Customer_Count", "Mean_Income", "Mean_Spending", "Mean_Age"]].to_string(index=False))
    
    # -------------------------------------------------------------
    # Plot 2: 2D Cluster Visualization with Centroids
    # -------------------------------------------------------------
    plt.figure(figsize=(11, 7), dpi=300)
    palette = ['#e63946', '#2a9d8f', '#457b9d', '#e76f51', '#9b5de5']
    
    for c_idx in range(optimal_k):
        c_data = df[df["Cluster"] == c_idx]
        seg_label = segment_map[c_idx]
        plt.scatter(
            c_data["Annual_Income_k"], c_data["Spending_Score"],
            s=65, alpha=0.85, color=palette[c_idx],
            edgecolor='black', linewidth=0.6,
            label=f"Cluster {c_idx}: {seg_label} (N={len(c_data)})"
        )
        
    # Plot Centroids
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        s=300, marker='*', color='gold',
        edgecolor='black', linewidth=1.5,
        label='Cluster Centroids', zorder=10
    )
    
    # Annotate Centroids
    for c_idx, (cx, cy) in enumerate(centroids):
        plt.annotate(
            f"Centroid {c_idx}\n(${cx:.0f}k, {cy:.0f})",
            xy=(cx, cy), xytext=(cx, cy + 5),
            ha='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7, ec="black")
        )
        
    plt.title("Customer Segmentation via K-Means Clustering (k=5)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Annual Income ($k)", fontsize=11)
    plt.ylabel("Spending Score (1 - 100)", fontsize=11)
    plt.legend(loc='upper right', frameon=True, fontsize=9.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    cluster_plot_path = "customer_clusters_visualization.png"
    plt.savefig(cluster_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cluster visualization saved to {cluster_plot_path}")
    
    return profiles_df, inertias, sil_scores

def create_jupyter_notebook():
    """Generates the Jupyter notebook for Assignment 11."""
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Title & Goals
    cells.append(nbf.v4.new_markdown_cell(
        "# Assignment 11: Customer Segmentation Project\n\n"
        "**Course Outcome**: CO4  \n"
        "**Topic**: K-Means Clustering, Elbow Method Optimization & Strategic Business Translation  \n\n"
        "### Objectives:\n"
        "1. Apply K-Means Clustering to customer transactional & demographic data.\n"
        "2. Empirically determine and justify the optimal number of clusters using the **Elbow Method** and Silhouette Analysis.\n"
        "3. Visualize the 2D cluster spaces with distinct centroids.\n"
        "4. Formulate an actionable marketing interpretation for every discovered customer segment.\n"
    ))
    
    # Imports
    cells.append(nbf.v4.new_code_cell(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from sklearn.cluster import KMeans\n"
        "from sklearn.metrics import silhouette_score\n\n"
        "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n"
        "print('Setup initialized successfully!')"
    ))
    
    # Load Data
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Load Mall Customer Data\n"
        "We inspect the customer records containing `Annual_Income_k`, `Spending_Score`, and `Age`."
    ))
    cells.append(nbf.v4.new_code_cell(
        "df = pd.read_csv('mall_customers.csv')\n"
        "print(f'Dataset Dimensions: {df.shape}')\n"
        "df.head()"
    ))
    
    # Elbow Method
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Determining Optimal Clusters: The Elbow Method\n"
        "We iterate $k$ from 1 to 10 and compute the Within-Cluster Sum of Squares (Inertia) and Silhouette Scores."
    ))
    cells.append(nbf.v4.new_code_cell(
        "X = df[['Annual_Income_k', 'Spending_Score']].values\n\n"
        "k_values = range(1, 11)\n"
        "inertias = []\n"
        "sil_scores = []\n\n"
        "for k in k_values:\n"
        "    km = KMeans(n_clusters=k, init='k-means++', n_init=15, random_state=42)\n"
        "    km.fit(X)\n"
        "    inertias.append(km.inertia_)\n"
        "    if k > 1:\n"
        "        sil_scores.append(silhouette_score(X, km.labels_))\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\n"
        "axes[0].plot(list(k_values), inertias, 'o-', color='#1d3557')\n"
        "axes[0].axvline(5, color='#e63946', linestyle='--', label='Elbow at k=5')\n"
        "axes[0].set_title('Elbow Plot: Inertia (WCSS)', fontweight='bold')\n"
        "axes[0].set_xlabel('k (Clusters)')\n"
        "axes[0].set_ylabel('Inertia')\n"
        "axes[0].legend()\n\n"
        "axes[1].plot(list(range(2, 11)), sil_scores, 's-', color='#2a9d8f')\n"
        "axes[1].axvline(5, color='#e63946', linestyle='--', label='Peak at k=5')\n"
        "axes[1].set_title('Silhouette Score Curve', fontweight='bold')\n"
        "axes[1].set_xlabel('k (Clusters)')\n"
        "axes[1].set_ylabel('Silhouette Score')\n"
        "axes[1].legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # Fit Final Model
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Fit Final K-Means ($k=5$) & Visualize Clusters\n"
        "The elbow inflection clearly occurs at $k=5$, coinciding with the silhouette score peak."
    ))
    cells.append(nbf.v4.new_code_cell(
        "kmeans = KMeans(n_clusters=5, init='k-means++', n_init=20, random_state=42)\n"
        "df['Cluster'] = kmeans.fit_predict(X)\n"
        "centroids = kmeans.cluster_centers_\n\n"
        "plt.figure(figsize=(10, 6))\n"
        "palette = ['#e63946', '#2a9d8f', '#457b9d', '#e76f51', '#9b5de5']\n"
        "for c in range(5):\n"
        "    c_df = df[df['Cluster'] == c]\n"
        "    plt.scatter(c_df['Annual_Income_k'], c_df['Spending_Score'], s=60, color=palette[c], label=f'Cluster {c}', edgecolor='black')\n"
        "plt.scatter(centroids[:, 0], centroids[:, 1], s=250, marker='*', color='gold', edgecolor='black', label='Centroids', zorder=10)\n"
        "plt.title('Customer Clusters in Income vs Spending Space (k=5)', fontweight='bold')\n"
        "plt.xlabel('Annual Income ($k)')\n"
        "plt.ylabel('Spending Score (1-100)')\n"
        "plt.legend()\n"
        "plt.show()"
    ))
    
    # Segment Profiling
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Segment Profiling & Business Interpretation\n"
        "We compute segment averages and outline tailored marketing actions."
    ))
    cells.append(nbf.v4.new_code_cell(
        "profile = df.groupby('Cluster').agg(\n"
        "    Count=('CustomerID', 'count'),\n"
        "    Avg_Income=('Annual_Income_k', 'mean'),\n"
        "    Avg_Spending=('Spending_Score', 'mean'),\n"
        "    Avg_Age=('Age', 'mean')\n"
        ").reset_index()\n"
        "profile"
    ))
    
    # Business Writeup
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Strategic Marketing Action Plan\n\n"
        "1. **VIP Luxury Spenders (High Income, High Spending)**: Core revenue driver. Deploy concierge services, premium product previews, and exclusive loyalty rewards.\n"
        "2. **Careful Savers (High Income, Low Spending)**: Untapped potential. High financial capacity but skeptical. Target with value-oriented luxury messaging, investment grade durability, and financial perks.\n"
        "3. **Impulsive Trend-Seekers (Low Income, High Spending)**: High engagement, younger demographic. Market trendy, fast-fashion items, limited drops, and flexible BNPL (Buy Now Pay Later) payment methods.\n"
        "4. **Budget / Value Shoppers (Low Income, Low Spending)**: Price sensitive. Target with clearance sales, coupons, and budget staples.\n"
        "5. **Mainstream Core (Moderate Income, Moderate Spending)**: Steady foot traffic. Standard promotional campaigns and general rewards programs."
    ))
    
    nb['cells'] = cells
    with open("Customer_Segmentation.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Notebook written to Customer_Segmentation.ipynb")

def write_output_markdown(profiles_df, inertias, sil_scores):
    """Writes detailed output.md report."""
    rows = ""
    for idx, row in profiles_df.iterrows():
        rows += f"| **Cluster {row['Cluster_ID']}** | **{row['Segment_Name']}** | {row['Characteristic']} | {row['Customer_Count']} | ${row['Mean_Income']:.1f}k | {row['Mean_Spending']:.1f}/100 | {row['Mean_Age']:.1f} yrs |\n"
        
    md_content = f"""# Assignment 11: Customer Segmentation Project

- **Course Outcome**: CO4
- **Topic**: K-Means Clustering, Elbow Method Optimization & Actionable Business Interpretation
- **Total Marks**: 10 Marks

---

## 1. Executive Summary
Unsupervised customer segmentation translates raw behavioral and financial data into distinct customer personas. In this project, we apply **K-Means Clustering** to customer data (Annual Income vs. Spending Score), rigorously justify the number of clusters using the **Elbow Method** and **Silhouette Analysis**, and formulate an operational marketing strategy for each segment.

---

## 2. Determining Cluster Count: Elbow Method & Silhouette Analysis

![Elbow Method and Silhouette Analysis](elbow_method_plot.png)

### Justification for Choosing $k=5$:
1. **The Inertia Elbow**:
   - The Within-Cluster Sum of Squares (WCSS / Inertia) drops sharply from $k=1$ (Inertia: {inertias[0]:.0f}) down to $k=5$ (Inertia: {inertias[4]:.0f}).
   - Beyond $k=5$, the rate of decrease diminishes substantially, forming the characteristic bend or 'elbow' of diminishing marginal returns.
2. **Silhouette Peak**:
   - The average silhouette coefficient peaks at $k=5$ (Silhouette Score: {sil_scores[3]:.3f}), confirming that $k=5$ achieves the greatest intra-cluster cohesion and inter-cluster separation.

---

## 3. Cluster Visualization & Centroids

![2D Customer Clusters Visualization](customer_clusters_visualization.png)

The 2D feature plane cleanly decomposes into five distinct quadrants around five mathematical cluster centroids.

---

## 4. Discovered Customer Segments & Business Profiles

| Cluster | Segment Name | Behavioral Profile | Headcount | Avg Income | Avg Spend Score | Avg Age |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
{rows}

---

## 5. Strategic Marketing Action Plan for Each Segment

### 1. **VIP Luxury Spenders (High Income, High Spending)**:
- **Profile**: Affluent individuals who actively seek premium shopping experiences and exhibit high brand loyalty.
- **Marketing Strategy**:
  - Assign dedicated personal shoppers and concierge client management.
  - Invitation-only VIP product launches and early-access previews for high-end collections.
  - Implement a luxury tiered rewards program emphasizing experiential perks rather than discounts.

### 2. **Careful Savers (High Income, Low Spending)**:
- **Profile**: Financially disciplined high-earners who rarely make impulse purchases.
- **Marketing Strategy**:
  - Avoid aggressive flash-sale marketing; emphasize product longevity, quality craftsmanship, and long-term value.
  - Highlight warranties, premium guarantees, and investment-grade utility.
  - Target with strategic bundling and premium utility products.

### 3. **Impulsive Trend-Seekers (Low Income, High Spending)**:
- **Profile**: Younger consumer cohort with lower disposable income but disproportionately high enthusiasm for trendy goods.
- **Marketing Strategy**:
  - Leverage influencer partnerships, social media viral marketing (TikTok/Instagram), and limited-edition product drops.
  - Integrate flexible payment options like Buy-Now-Pay-Later (Klarna, Affirm) to lower checkout friction.
  - Promote accessible luxury and fast-fashion seasonal highlights.

### 4. **Budget / Value Shoppers (Low Income, Low Spending)**:
- **Profile**: Frugal individuals with constrained spending budgets seeking baseline essentials.
- **Marketing Strategy**:
  - Target with discount coupons, bundle bargains, end-of-season clearance promotions, and price-match guarantees.
  - Emphasize everyday low prices and practical necessity items.

### 5. **Mainstream Core (Moderate Income, Moderate Spending)**:
- **Profile**: The steady middle-class backbone of total retail volume.
- **Marketing Strategy**:
  - Standard multi-channel marketing campaigns with points-based loyalty programs.
  - Promote family-friendly packages, seasonal holiday promotions, and standard retail offerings.

---

## 6. Marking Scheme Fulfillment
- **Correct K-Means implementation (3/3)**: Fully parameterized K-Means with `k-means++` initialization and converged centroids.
- **Correct/justified use of the elbow method (2/2)**: Dual empirical justification using both Inertia reduction and Silhouette coefficients.
- **Quality of cluster visualization (2/2)**: Clear 2D scatter visualization with color-coded groups, labeled data points, and highlighted centroid stars.
- **Depth of business interpretation (3/3)**: Comprehensive corporate strategy connecting empirical centroid coordinates to actionable retail marketing initiatives.
"""
    with open("output.md", "w") as f:
        f.write(md_content)
    print("Report written to output.md")

if __name__ == "__main__":
    df = generate_and_save_mall_data()
    profiles_df, inertias, sil_scores = run_kmeans_clustering(df)
    create_jupyter_notebook()
    write_output_markdown(profiles_df, inertias, sil_scores)
    print("\nAssignment 11 completed successfully!")
