import pandas as pd

def compare_datasets(dataset1, dataset2):
    # Read datasets
    df1 = pd.read_csv(dataset1)
    df2 = pd.read_csv(dataset2)
    
    # Check if 'reqLabel' column is present in both datasets
    if 'reqLabel' not in df1.columns or 'reqLabel' not in df2.columns:
        print("Error: 'reqLabel' column not found in both datasets.")
        return
    
    # Extract 'reqLabel' values from both datasets
    reqLabels1 = set(df1['reqLabel'].unique())
    reqLabels2 = set(df2['reqLabel'].unique())
    
    # Compare 'reqLabel' values between datasets
    common_reqLabels = reqLabels1.intersection(reqLabels2)
    unique_reqLabels_df1 = reqLabels1 - common_reqLabels
    unique_reqLabels_df2 = reqLabels2 - common_reqLabels
    
    # Calculate similarity metrics
    total_reqLabels_df1 = len(reqLabels1)
    total_reqLabels_df2 = len(reqLabels2)
    common_count = len(common_reqLabels)
    unique_count_df1 = len(unique_reqLabels_df1)
    unique_count_df2 = len(unique_reqLabels_df2)
    
    # Calculate similarity percentage
    similarity_percentage_df1 = (common_count / total_reqLabels_df1) * 100
    similarity_percentage_df2 = (common_count / total_reqLabels_df2) * 100
    
    # Print comparison results
    print("Comparison Results:")
    print("===================")
    print(f"Total reqLabels in Dataset 1: {total_reqLabels_df1}")
    print(f"Total reqLabels in Dataset 2: {total_reqLabels_df2}")
    print(f"Common reqLabels: {common_count}")
    print(f"Unique reqLabels in Dataset 1: {unique_count_df1}")
    print(f"Unique reqLabels in Dataset 2: {unique_count_df2}")
    print(f"Similarity Percentage (Dataset 1): {similarity_percentage_df1:.2f}%")
    print(f"Similarity Percentage (Dataset 2): {similarity_percentage_df2:.2f}%")

# Example usage
dataset1_path = 'dataset1.csv'
dataset2_path = 'dataset2.csv'
compare_datasets(dataset1_path, dataset2_path)