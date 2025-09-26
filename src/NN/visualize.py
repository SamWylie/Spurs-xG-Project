import matplotlib.pyplot as plt
import seaborn as sns
import tottenham_analysis
import matplotlib
import features
matplotlib.use("Agg")  # non-interactive backend, avoids Qt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#shots scatterplot
df = tottenham_analysis.tottenham_df
dfX, dfY = features.trim(df)
df['result'] = dfY
plt.figure(figsize=(10, 6))
sns.scatterplot(x='X', y='Y', data=df, hue='result', alpha=0.5)
plt.savefig('reports/figures/shots.png', dpi=300, bbox_inches='tight')
plt.clf() 

result1 = tottenham_analysis.result1
#comparing yearly performance 
sns.scatterplot(x='xG', y='result', data=result1)
for i in range(result1.shape[0]):
    plt.text(result1["xG"][i]+0.1, result1["result"][i]+0.1, result1["year"][i], fontsize=9)
sns.regplot(x="xG", y="result", data=result1, scatter=False, ci=None, color="red")
plt.savefig('reports/figures/xG-Goals-per-Year.png', dpi=300, bbox_inches='tight')

result2 = tottenham_analysis.result2
sns.scatterplot(x='xG', y='result', data=result2)
for i in range(result2.shape[0]):
    plt.text(result2["xG"][i]+0.1, result2["result"][i]+0.1, result2["player"][i], fontsize=9)
sns.regplot(x="xG", y="result", data=result2, scatter=False, ci=None, color="red")
plt.savefig('reports/figures/PlayerComparison.png', dpi=300, bbox_inches='tight')