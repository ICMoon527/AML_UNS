# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import randint

# # Example Data: Replace with your actual data
# # Assume your data is in a DataFrame with columns: 'Chromosome', 'Position', 'p-value'
# # 'Chromosome' can represent clusters or groups in your data
# # data = pd.DataFrame({
# #     'Chromosome': np.repeat(n_clusters),
# #     'Position': np.tile(np.arange(1, 101), 10),
# #     'p-value': np.random.uniform(0.0001, 1, 300)  # Random p-values
# # })

# # some sample data
# #significant_features = np.random.uniform(0, 1, 550)  # Example, replace with your actual data
# # significant_features =filtered_data
# # print(significant_features)

# df = pd.DataFrame({
#     'chromosome': ['Indices-%i' % i for i in np.arange(555)],
#     'pvalue': filtered_data[:,1],
#     'Position': ['ch-%i' % i for i in np.random.randint(0, 10, size=555)]
# })

# print((df))

# # Calculate -log10(p-value)
# # Calculate -log10(p-value)
# df['-log10(p-value)'] = -np.log10(df['pvalue'])

# # Sort data by chromosome and position
# data = df.sort_values(['Chromosome', 'Position'])

# # Create a new column for plotting purposes that combines chromosome and position
# data['pos'] = range(len(data))

# # Manhattan Plot
# plt.figure(figsize=(12, 6))

# # Define colors for alternating chromosomes (clusters in this case)
# colors = sns.color_palette("husl", len(data['Chromosome'].unique()))
# sns.scatterplot(
#     x='pos',
#     y='-log10(p-value)',
#     hue='Chromosome',
#     palette=colors,
#     data=data,
#     legend='full',
#     edgecolor=None,
#     s=50
# )

# # Add axis labels and title
# plt.xlabel('Position in Chromosome (Cluster)')
# plt.ylabel('-log10(p-value)')
# plt.title('Manhattan Plot of p-values across Clusters')

# # Add chromosome ticks
# chr_ticks = []
# for i, chrom in enumerate(data['Chromosome'].unique()):
#     tick_pos = data[data['Chromosome'] == chrom]['pos'].median()
#     chr_ticks.append(tick_pos)
# plt.xticks(chr_ticks, data['Ch

#     chr_ticks.append(tick_pos)
# plt.xticks(chr_ticks, data['Chromosome'].unique(), rotation=45, ha='right')

# # Horizontal line for genome-wide significance level (e.g., 5e-8)
# plt.axhline(y=-np.log10(5e-8), color='grey', linestyle='--', lw=1)

# plt.grid(True)
# plt.show()