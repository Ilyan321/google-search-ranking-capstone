import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, brier_score_loss, roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

# Style settings
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'content_refresh_anonymized.csv'))
FIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))
DOCS_FIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures'))
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DOCS_FIG_DIR, exist_ok=True)

def precision_at_k(y_true, y_scores, k=50):
    idx = np.argsort(y_scores)[::-1][:k]
    return np.mean(y_true.iloc[idx] if hasattr(y_true, 'iloc') else y_true[idx])

def recall_at_k(y_true, y_scores, k=50):
    idx = np.argsort(y_scores)[::-1][:k]
    total_pos = np.sum(y_true)
    if total_pos == 0:
        return 0.0
    return np.sum(y_true.iloc[idx] if hasattr(y_true, 'iloc') else y_true[idx]) / total_pos

def run_pipeline():
    print("Loading FlyRank dataset from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape[0]:,} rows, {df.shape[1]} columns across {df['client_id'].nunique()} clients.")

    # -------------------------------------------------------------
    # 1. SIGNAL ANALYSIS & MYTH-BUSTING
    # -------------------------------------------------------------
    print("\n--- 1. Signal Analysis & Myth-Busting ---")
    
    # Myth 1: Search Volume vs Impressions
    corr_pearson = df[['search_volume', 'impressions_90d']].corr().iloc[0, 1]
    corr_spearman = df[['search_volume', 'impressions_90d']].corr(method='spearman').iloc[0, 1]
    print(f"Search Volume vs Impressions 90d Pearson Correlation: {corr_pearson:.5f}")
    print(f"Search Volume vs Impressions 90d Spearman Correlation: {corr_spearman:.5f}")

    # Myth 2: Length vs Decline/Growth
    growing_df = df[df['trend_direction'] == 'up']
    declining_df = df[df['trend_direction'] == 'down']
    avg_words_growing = growing_df['word_count'].median()
    avg_words_declining = declining_df['word_count'].median()
    print(f"Median Word Count - Growing Content: {avg_words_growing:.1f} words")
    print(f"Median Word Count - Declining Content: {avg_words_declining:.1f} words")

    # Position vs CTR Decay
    ctr_by_tier = df.groupby('position_tier')['ctr'].mean().reset_index()
    print("Average CTR by Position Tier:\n", ctr_by_tier)

    # Plot 1: Myth-Busting Correlations & Distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1A: Search volume vs Impressions
    sns.scatterplot(
        data=df.sample(2000, random_state=42),
        x='search_volume', y='impressions_90d',
        alpha=0.4, color='#2563eb', ax=axes[0]
    )
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_title(f'Search Volume vs Impressions (r = {corr_pearson:.4f})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Third-Party Search Volume (log)')
    axes[0].set_ylabel('Actual GSC Impressions 90d (log)')

    # 1B: CTR vs Position Curve
    pos_summary = df.groupby('position_tier').agg(
        mean_ctr=('ctr', 'mean'),
        median_ctr=('ctr', 'median')
    ).reindex(['top_3', 'striking', 'page_1', 'page_2', 'page_3_5', 'deep']).dropna()
    
    bars = axes[1].bar(pos_summary.index, pos_summary['mean_ctr'], color='#0ea5e9', edgecolor='#0369a1')
    axes[1].set_title('Empirical CTR Decay Across Position Tiers', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Search Position Tier')
    axes[1].set_ylabel('Mean Observed CTR (%)')
    axes[1].tick_params(axis='x', rotation=30)
    for bar in bars:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}%", ha='center', va='bottom', fontsize=9)

    # 1C: Word Count by Performance Trend
    sns.boxplot(
        data=df[df['trend_direction'].isin(['up', 'down', 'stable'])],
        x='trend_direction', y='word_count',
        palette={'up': '#10b981', 'down': '#ef4444', 'stable': '#64748b'},
        showfliers=False, ax=axes[2]
    )
    axes[2].set_title('Word Count Distribution by Ranking Trend', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Performance Trend')
    axes[2].set_ylabel('Article Word Count')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '01_signal_analysis_myths.png'))
    plt.savefig(os.path.join(DOCS_FIG_DIR, '01_signal_analysis_myths.png'))
    plt.close()

    # -------------------------------------------------------------
    # 2. CONTENT ARCHETYPE CLUSTERING
    # -------------------------------------------------------------
    print("\n--- 2. Content Archetype Clustering ---")
    cluster_features = ['impressions_90d', 'clicks_90d', 'ctr', 'avg_position', 'content_age_days', 'days_since_last_update']
    X_cluster = df[cluster_features].copy()
    X_cluster_imputed = X_cluster.fillna(X_cluster.median())
    
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(np.log1p(np.maximum(X_cluster_imputed, 0)))

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['archetype_cluster'] = kmeans.fit_predict(X_cluster_scaled)

    # Map archetypes based on centroid characteristics
    centroids = df.groupby('archetype_cluster')[cluster_features].mean()
    print("Cluster Centroids (raw scale):\n", centroids)

    # Robust cluster mapping based on relative ranking of cluster centroids
    # Rank clusters by impressions and ctr
    sorted_by_imp = centroids['impressions_90d'].sort_values(ascending=False).index.tolist()
    
    cluster_mapping = {
        sorted_by_imp[0]: 'Powerhouse Core (Protect)',
        sorted_by_imp[1]: 'Decaying Traffic Stars (Refresh)',
        sorted_by_imp[2]: 'High-Impression CTR Gap (Metadata Fix)',
        sorted_by_imp[3]: 'Zombie Longtail (Prune / Consolidate)'
    }

    df['archetype_name'] = df['archetype_cluster'].map(cluster_mapping)

    # PCA 2D Visualization
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_cluster_scaled)
    df['pca_1'] = pca_coords[:, 0]
    df['pca_2'] = pca_coords[:, 1]

    plt.figure(figsize=(10, 7))
    palette = {'Powerhouse Core (Protect)': '#2563eb', 
               'Decaying Traffic Stars (Refresh)': '#f59e0b', 
               'High-Impression CTR Gap (Metadata Fix)': '#10b981', 
               'Zombie Longtail (Prune / Consolidate)': '#ef4444'}
    sns.scatterplot(
        data=df.sample(3000, random_state=42),
        x='pca_1', y='pca_2', hue='archetype_name',
        alpha=0.6, s=30, palette=palette
    )
    plt.title('Content Performance Archetypes (PCA 2D Space)', fontsize=14, fontweight='bold')
    plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    plt.legend(title='Content Archetype', loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '02_archetype_clusters.png'))
    plt.savefig(os.path.join(DOCS_FIG_DIR, '02_archetype_clusters.png'))
    plt.close()

    # -------------------------------------------------------------
    # 3. FEATURE ENGINEERING & LEAKAGE PREVENTION
    # -------------------------------------------------------------
    print("\n--- 3. Feature Engineering & Strict Leakage Auditing ---")
    
    # Ground truth label: is_initial_refresh_candidate (fill NaN with False)
    y = df['is_initial_refresh_candidate'].fillna(False).astype(int)
    print(f"Target Distribution: {y.value_counts(normalize=True).to_dict()} (Positive count: {y.sum():,})")

    # Exclude all target leakage features (internal FlyRank diagnostic flags)
    leakage_columns = [
        'health_score', 'needs_indexing', 'is_quick_win', 'needs_ctr_fix',
        'needs_engagement_fix', 'ai_opportunity', 'is_underperformer',
        'is_declining', 'is_initial_refresh_candidate', 'trend_direction', 'trend_pct',
        'archetype_cluster', 'archetype_name', 'pca_1', 'pca_2'
    ]

    # Clean feature engineering
    df_feat = pd.DataFrame(index=df.index)
    
    # Raw engagement and search metrics
    df_feat['log_search_volume'] = np.log1p(df['search_volume'].fillna(0))
    df_feat['competition'] = df['competition'].fillna(df['competition'].median())
    df_feat['cpc'] = df['cpc'].fillna(0)
    df_feat['word_count'] = df['word_count'].fillna(df['word_count'].median())
    df_feat['char_count'] = df['char_count'].fillna(df['char_count'].median())
    
    df_feat['log_impressions_90d'] = np.log1p(df['impressions_90d'].fillna(0))
    df_feat['log_clicks_90d'] = np.log1p(df['clicks_90d'].fillna(0))
    df_feat['log_sessions_90d'] = np.log1p(df['sessions_90d'].fillna(0))
    df_feat['log_pageviews_90d'] = np.log1p(df['pageviews_90d'].fillna(0))
    df_feat['ctr'] = df['ctr'].fillna(0)
    df_feat['avg_position'] = df['avg_position'].fillna(df['avg_position'].median())
    df_feat['content_age_days'] = df['content_age_days'].fillna(df['content_age_days'].median())
    df_feat['days_since_last_update'] = df['days_since_last_update'].fillna(df['days_since_last_update'].median())
    
    # Velocity & Ratio Features (Comparing last 30d vs prev 30d)
    df_feat['impression_velocity'] = (df['impressions_last_30d'] + 1) / (df['impressions_prev_30d'] + 1)
    df_feat['click_velocity'] = (df['clicks_last_30d'] + 1) / (df['clicks_prev_30d'] + 1)
    df_feat['session_velocity'] = (df['sessions_last_30d'] + 1) / (df['sessions_prev_30d'] + 1)
    df_feat['decay_ratio_30d'] = (df['impressions_prev_30d'] - df['impressions_last_30d']) / (df['impressions_90d'] + 1)

    # Efficiency Ratios
    df_feat['engagement_density'] = df['engaged_sessions_90d'].fillna(0) / (df['sessions_90d'] + 1)
    df_feat['scroll_density'] = df['scroll_events_90d'].fillna(0) / (df['pageviews_90d'] + 1)
    df_feat['capture_efficiency'] = df['clicks_90d'] / (df['search_volume'] + 1)
    df_feat['active_days_ratio'] = df['days_with_impressions'].fillna(0) / 90.0

    # Categorical Encoded
    for col in ['competition_level', 'main_intent', 'content_type', 'age_tier', 'position_tier']:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df_feat = pd.concat([df_feat, dummies], axis=1)

    print(f"Engineered {df_feat.shape[1]} clean, leakage-free features.")

    # -------------------------------------------------------------
    # 4. GROUPED CROSS-VALIDATION & MODEL EVALUATION
    # -------------------------------------------------------------
    print("\n--- 4. Grouped 5-Fold Cross-Validation Across Clients ---")
    groups = df['client_id']
    gkf = GroupKFold(n_splits=5)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=0.1, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10, n_jobs=-1, random_state=42),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08, random_state=42)
    }

    # Storage for out-of-fold predictions
    oof_predictions = {name: np.zeros(len(df)) for name in models}
    oof_predictions['Rule Heuristic Baseline'] = np.zeros(len(df))

    # Compute Rule Baseline (Age > 180 & Impressions > 1000 & Position > 10)
    rule_score = (
        (df['content_age_days'] > 180).astype(float) * 0.3 +
        (df['impressions_90d'] > 1000).astype(float) * 0.4 +
        (df['avg_position'] > 10).astype(float) * 0.3
    )
    oof_predictions['Rule Heuristic Baseline'] = rule_score.values

    # Run GroupKFold
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_feat, y, groups=groups)):
        X_train, y_train = df_feat.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = df_feat.iloc[val_idx], y.iloc[val_idx]

        # Fit models
        for name, model in models.items():
            if name == 'Logistic Regression':
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_train.fillna(0))
                X_va_s = scaler.transform(X_val.fillna(0))
                model.fit(X_tr_s, y_train)
                oof_predictions[name][val_idx] = model.predict_proba(X_va_s)[:, 1]
            else:
                model.fit(X_train, y_train)
                oof_predictions[name][val_idx] = model.predict_proba(X_val)[:, 1]

    # Calculate overall metrics
    results_summary = []
    for name, preds in oof_predictions.items():
        roc_auc = roc_auc_score(y, preds)
        pr_auc = average_precision_score(y, preds)
        p_10 = precision_at_k(y, preds, k=10)
        p_20 = precision_at_k(y, preds, k=20)
        p_50 = precision_at_k(y, preds, k=50)
        p_100 = precision_at_k(y, preds, k=100)
        r_50 = recall_at_k(y, preds, k=50)
        brier = brier_score_loss(y, preds)

        results_summary.append({
            'Model': name,
            'ROC-AUC': round(roc_auc, 4),
            'PR-AUC': round(pr_auc, 4),
            'Precision@10': round(p_10, 4),
            'Precision@20': round(p_20, 4),
            'Precision@50': round(p_50, 4),
            'Precision@100': round(p_100, 4),
            'Recall@50': round(r_50, 4),
            'Brier Score': round(brier, 4)
        })

    results_df = pd.DataFrame(results_summary)
    print("\n--- Empirical Validation Results ---")
    print(results_df.to_markdown(index=False))

    # Save metrics JSON for charts / paper
    with open(os.path.join(FIG_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

    # -------------------------------------------------------------
    # 5. GENERATE EVALUATION CHARTS
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 5A: Precision@K Comparison
    k_vals = [10, 20, 50, 100, 200, 500]
    for name, color in [('Rule Heuristic Baseline', '#64748b'), ('Logistic Regression', '#0ea5e9'),
                        ('Random Forest', '#2563eb'), ('HistGradientBoosting', '#10b981')]:
        prec_curves = [precision_at_k(y, oof_predictions[name], k=k) for k in k_vals]
        axes[0].plot(k_vals, prec_curves, marker='o', label=name, color=color, linewidth=2.2)
    
    axes[0].set_title('Precision@K Across Action Ranks (Decision Priority)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Top K Recommended Candidates (Bandwidth Capacity)')
    axes[0].set_ylabel('Precision (True Opportunity Ratio)')
    axes[0].legend(loc='lower left', frameon=True)
    axes[0].set_ylim(0, 1.05)

    # 5B: ROC Curves
    for name, color in [('Rule Heuristic Baseline', '#64748b'), ('Logistic Regression', '#0ea5e9'),
                        ('Random Forest', '#2563eb'), ('HistGradientBoosting', '#10b981')]:
        fpr, tpr, _ = roc_curve(y, oof_predictions[name])
        auc = roc_auc_score(y, oof_predictions[name])
        axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2.0)
    
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1].set_title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend(loc='lower right', frameon=True)

    # 5C: PR Curves
    for name, color in [('Rule Heuristic Baseline', '#64748b'), ('Logistic Regression', '#0ea5e9'),
                        ('Random Forest', '#2563eb'), ('HistGradientBoosting', '#10b981')]:
        precision, recall, _ = precision_recall_curve(y, oof_predictions[name])
        ap = average_precision_score(y, oof_predictions[name])
        axes[2].plot(recall, precision, label=f"{name} (AP={ap:.3f})", color=color, linewidth=2.0)
    
    axes[2].set_title('Precision-Recall Curve (PR-AUC)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].legend(loc='lower left', frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '03_model_evaluation_curves.png'))
    plt.savefig(os.path.join(DOCS_FIG_DIR, '03_model_evaluation_curves.png'))
    plt.close()

    # -------------------------------------------------------------
    # 6. FEATURE IMPORTANCE & EXPLAINABILITY
    # -------------------------------------------------------------
    print("\n--- 6. Feature Importance & Model Explainability ---")
    rf_full = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42)
    rf_full.fit(df_feat, y)
    
    feat_importances = pd.Series(rf_full.feature_importances_, index=df_feat.columns).sort_values(ascending=False).head(15)
    print("Top 15 Most Influential Features:\n", feat_importances)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(feat_importances.index[::-1], feat_importances.values[::-1], color='#2563eb', edgecolor='#1d4ed8')
    plt.title('Top 15 Signal Importances (Random Forest MDI)', fontsize=13, fontweight='bold')
    plt.xlabel('Relative Feature Importance (Mean Decrease in Impurity)')
    plt.ylabel('Engineered Signal')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '04_feature_importance.png'))
    plt.savefig(os.path.join(DOCS_FIG_DIR, '04_feature_importance.png'))
    plt.close()

    # -------------------------------------------------------------
    # 7. PRODUCTION DECISION-SUPPORT ENGINE & ACTION PLAYBOOK
    # -------------------------------------------------------------
    print("\n--- 7. Building Production Opportunity Scoring Engine ---")
    df['model_opportunity_score'] = np.round(oof_predictions['Random Forest'] * 100, 1)

    # Generate Explainable Reason Codes
    def generate_reason_codes(row):
        reasons = []
        if row['impressions_last_30d'] < row['impressions_prev_30d'] * 0.65:
            reasons.append('CRITICAL_DECAY_VELOCITY')
        if row['avg_position'] >= 8.0 and row['avg_position'] <= 20.0 and row['ctr'] < 0.35:
            reasons.append('STRIKING_POSITION_CTR_GAP')
        if row['content_age_days'] > 180 and row['days_since_last_update'] > 60:
            reasons.append('STALE_CONTENT_LIFECYCLE')
        if row['impressions_90d'] > 3000 and (row['impressions_prev_30d'] - row['impressions_last_30d']) > 500:
            reasons.append('HIGH_EQUITY_DRIFT')
        if row['engagement_rate'] < 10.0 and row['sessions_90d'] > 15:
            reasons.append('ENGAGEMENT_RETENTION_BOUNCE')
        
        if not reasons:
            reasons.append('MONITOR_BASELINE')
        return ' | '.join(reasons)

    df['reason_codes'] = df.apply(generate_reason_codes, axis=1)

    # Ranked recommendations table
    ranked_candidates = df.sort_values(by='model_opportunity_score', ascending=False)[
        ['content_id', 'client_id', 'model_opportunity_score', 'impressions_90d', 'clicks_90d', 
         'avg_position', 'ctr', 'content_age_days', 'archetype_name', 'reason_codes']
    ].head(50)

    print("\nTop 10 High-Priority Content Refresh Opportunities:\n")
    print(ranked_candidates.head(10).to_markdown(index=False))

    # Save Ranked Recommendations Sample
    ranked_candidates.to_csv(os.path.join(FIG_DIR, 'top_50_ranked_recommendations.csv'), index=False)
    ranked_candidates.to_csv(os.path.join(DOCS_FIG_DIR, 'top_50_ranked_recommendations.csv'), index=False)
    
    # Save as JSON as well for the interactive paper table
    ranked_candidates.to_json(os.path.join(DOCS_FIG_DIR, 'top_50_ranked_recommendations.json'), orient='records', indent=2)
    ranked_candidates.to_json(os.path.join(FIG_DIR, 'top_50_ranked_recommendations.json'), orient='records', indent=2)

    print("\nPipeline execution complete! All figures and data saved successfully.")

if __name__ == '__main__':
    run_pipeline()
