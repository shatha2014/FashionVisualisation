import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xlrd import open_workbook
data = pd.read_excel('RankCorr.XLSX', header = None)
print(data)
print(data.shape)
#data = open_workbook('RankCorr.XLSX')
class9_IWF = data.iloc[1, 1:]
print(class9_IWF.shape)
class9_F = data.iloc[2, 1:]
coef, p = spearmanr(class9_IWF, class9_F)
print('Spearmans correlation coefficient: %.3f' % coef)
alpha = 0.05
if p > alpha:
	print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('Samples are correlated (reject H0) p=%.3f' % p)

# plot
plt.scatter(class9_IWF, class9_F)
plt.savefig("rankplot.png")
plt.title("Rank correlation")
plt.xlabel("Rank for images without features")
plt.ylabel("Rank for images with only features")
plt.legend(loc="lower left")
plt.show()



