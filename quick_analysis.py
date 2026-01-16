#!/usr/bin/env python3
"""
Quick analysis of facial expression data to determine category thresholds
"""

import json
import pandas as pd
import numpy as np

def main():
    # Load first 500 samples for analysis
    data = []
    count = 0
    with open('FYP/RecruitView_Data/metadata.jsonl', 'r') as f:
        for line in f:
            if count >= 500:  # Limit to 500 samples for quick analysis
                break
            data.append(json.loads(line))
            count += 1

    df = pd.DataFrame(data)
    facial_scores = df['facial_expression']

    print(f"Analyzed {len(facial_scores)} samples")
    print("Facial Expression Score Statistics:")
    print(facial_scores.describe())

    # Calculate percentiles for balanced categories
    percentiles = np.percentile(facial_scores, [33.33, 66.67])
    print(f"\n33rd percentile: {percentiles[0]:.3f}")
    print(f"66th percentile: {percentiles[1]:.3f}")

    # Create categories and show distribution
    def categorize(score):
        if score <= percentiles[0]:
            return 'Reserved Expression'
        elif score <= percentiles[1]:
            return 'Balanced Expression'
        else:
            return 'Expressive'

    categories = facial_scores.apply(categorize)
    category_counts = categories.value_counts()
    category_percentages = categories.value_counts(normalize=True) * 100

    print("\nExpressiveness Category Distribution:")
    for category in ['Reserved Expression', 'Balanced Expression', 'Expressive']:
        if category in category_counts:
            count = category_counts[category]
            percentage = category_percentages[category]
            print(f"{category}: {count} samples ({percentage:.1f}%)")

    # Save thresholds
    np.save('FYP/RecruitView_Data/category_thresholds.npy', percentiles)
    print(f"\nThresholds saved: {percentiles}")

    return percentiles

if __name__ == "__main__":
    thresholds = main()