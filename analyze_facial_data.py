#!/usr/bin/env python3
"""
Analyze facial expression data from RecruitView dataset
to understand distribution and create appropriate categories.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def load_data(metadata_path):
    """Load metadata from JSONL file"""
    data = []
    with open(metadata_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def analyze_facial_expression_distribution(df):
    """Analyze the distribution of facial expression scores"""
    facial_scores = df['facial_expression']

    print("Facial Expression Score Statistics:")
    print(facial_scores.describe())
    print("\n" + "="*50)

    # Create histogram
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(facial_scores, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Facial Expression Scores')
    plt.xlabel('Facial Expression Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.boxplot(facial_scores)
    plt.title('Box Plot of Facial Expression Scores')
    plt.ylabel('Facial Expression Score')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('facial_expression_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return facial_scores

def create_expressiveness_categories(scores):
    """
    Create expressiveness categories based on statistical analysis
    Instead of arbitrary thresholds, use percentiles for balanced categories
    """
    # Calculate percentiles for balanced distribution
    percentiles = np.percentile(scores, [33.33, 66.67])

    print(f"33rd percentile: {percentiles[0]:.3f}")
    print(f"66th percentile: {percentiles[1]:.3f}")

    def categorize(score):
        if score <= percentiles[0]:
            return 'Reserved Expression'  # Low expressiveness
        elif score <= percentiles[1]:
            return 'Balanced Expression'  # Neutral expressiveness
        else:
            return 'Expressive'  # High expressiveness

    return categorize, percentiles

def analyze_category_distribution(df, categorize_func):
    """Analyze the distribution of expressiveness categories"""
    df_copy = df.copy()
    df_copy['expressiveness_category'] = df_copy['facial_expression'].apply(categorize_func)

    # Display distribution
    category_counts = df_copy['expressiveness_category'].value_counts()
    category_percentages = df_copy['expressiveness_category'].value_counts(normalize=True) * 100

    print("\nExpressiveness Category Distribution:")
    print("-" * 40)
    for category in ['Reserved Expression', 'Balanced Expression', 'Expressive']:
        if category in category_counts:
            count = category_counts[category]
            percentage = category_percentages[category]
            print(f"{category}: {count} samples ({percentage:.1f}%)")
        else:
            print(f"{category}: 0 samples (0.0%)")

    # Visualize category distribution
    plt.figure(figsize=(10, 6))
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    plt.bar(category_counts.index, category_counts.values, color=colors[:len(category_counts)])
    plt.title('Distribution of Expressiveness Categories')
    plt.xlabel('Expressiveness Category')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)

    # Add percentage labels
    for i, (category, count) in enumerate(category_counts.items()):
        percentage = category_percentages[category]
        plt.text(i, count + 5, f'{percentage:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('expressiveness_categories.png', dpi=300, bbox_inches='tight')
    plt.show()

    return df_copy

def analyze_correlations(df):
    """Analyze correlations between facial expression and other personality traits"""
    personality_cols = ['openness', 'conscientiousness', 'extraversion',
                       'agreeableness', 'neuroticism', 'facial_expression']

    # Calculate correlation matrix
    corr_matrix = df[personality_cols].corr()

    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix: Personality Traits and Facial Expression')
    plt.tight_layout()
    plt.savefig('personality_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print strongest correlations with facial expression
    facial_corr = corr_matrix['facial_expression'].drop('facial_expression')
    print("\nCorrelations with Facial Expression:")
    print("-" * 35)
    for trait, corr in facial_corr.sort_values(ascending=False).items():
        print(f"{trait.capitalize()}: {corr:.3f}")

def main():
    """Main analysis function"""
    print("Facial Expression Data Analysis")
    print("=" * 40)

    # Load data
    metadata_path = 'FYP/RecruitView_Data/metadata.jsonl'
    df = load_data(metadata_path)
    print(f"Loaded {len(df)} samples from dataset")

    # Analyze facial expression distribution
    facial_scores = analyze_facial_expression_distribution(df)

    # Create expressiveness categories based on percentiles
    categorize_func, percentiles = create_expressiveness_categories(facial_scores)

    # Analyze category distribution
    df_with_categories = analyze_category_distribution(df, categorize_func)

    # Analyze correlations with personality traits
    analyze_correlations(df)

    print("\nAnalysis complete!")
    print("Generated files:")
    print("- facial_expression_analysis.png")
    print("- expressiveness_categories.png")
    print("- personality_correlations.png")

    return df_with_categories, percentiles

if __name__ == "__main__":
    df_analyzed, category_thresholds = main()

    # Save the categorized data for use in the model
    output_path = 'FYP/RecruitView_Data/analyzed_data_with_categories.csv'
    df_analyzed.to_csv(output_path, index=False)
    print(f"\nCategorized data saved to: {output_path}")

    # Save category thresholds for model use
    thresholds_path = 'FYP/RecruitView_Data/category_thresholds.npy'
    np.save(thresholds_path, category_thresholds)
    print(f"Category thresholds saved to: {thresholds_path}")