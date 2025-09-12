import matplotlib.pyplot as plt
import seaborn as sns
import features

df = features.df.head(10000)

plt.figure(figsize=(10, 6))

sns.scatterplot(x='X', y='Y', data=df, hue='result', alpha=0.5)
plt.savefig('shot_plot.png', dpi=300, bbox_inches='tight')