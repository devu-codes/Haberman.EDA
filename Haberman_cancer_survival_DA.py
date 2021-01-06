# ------------------------------ EDA on Haberman cancer survival rate ---------------
# 1. objective : check survival rate of patient die after surgery or not
#Title: Haberman’s Survival Data
#Description: The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago’s Billings Hospital
# on the survival of patients who had undergone surgery for breast cancer.
#Attribute Information:
#Age of patient at the time of operation (numerical)
#Patient’s year of operation (year — 1900, numerical)
#Number of positive axillary nodes detected (numerical)
#Survival status (class attribute) :
#1 = the patient survived 5 years or longer


# 2. import important libraries
import numpy as np
#2 = the patient died within 5 years
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read the csv file
haberman = pd.read_csv('haberman.csv')

# 3. understand the data
print("Data view:\n",haberman.head())
# no of rows and columns
print(haberman.shape)
# printing columns
print(haberman.columns)
# info
print(haberman.info())
# No. of points
print(haberman['status'].value_counts()) # 1: Dead , 2: Not Dead

haberman['status'] = haberman['status'].map({1:'yes',2:'No'})

print(haberman.describe())

status_yes = haberman[haberman['status'] == 'yes']
print(status_yes.describe())
status_no = haberman[haberman['status'] == 'No']
print(status_no.describe())

# bivariate analysis
haberman.plot(kind='scatter',x='age',y='year')
plt.show()

#plt.show()
sns.set_style('whitegrid')
sns.pairplot(haberman,hue='status',height=3)
plt.show()

# univariate analysis
haber_yes = haberman.loc[haberman['status'] == 'yes']
haber_No = haberman.loc[haberman['status'] == 'No']

# 1. Age , 2. year 3. node
sns.FacetGrid(haberman,hue='status',height=5)\
    .map(sns.distplot,'age')\
    .add_legend()
plt.show()
# age(29 to 40) : more chances of survival,(40 to 60): less chances
sns.FacetGrid(haberman,hue='status',height=5)\
    .map(sns.distplot,'year')\
    .add_legend()
plt.show()
# nothing observed , too much overlapping.
sns.FacetGrid(haberman,hue='status',height=5)\
    .map(sns.distplot,'nodes')\
    .add_legend()
plt.show()
# nodes(=1):more chances of survival.
# nodes(>25): low chance of survival

# Observations : nodes pdf is useful more than age & year pdf's
counts,bin_edges = np.histogram(status_yes['nodes'],bins=10,density=True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
# 83.55% in the node range 0-4.6 : survival
# CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='yes')
plt.plot(bin_edges[1:],cdf,label='yes')
plt.xlabel('nodes')

print("---------------------------------")
counts1,bin_edges1 = np.histogram(status_no['nodes'], bins=10, density=True)
pdf1 = counts1/(sum(counts1))
print(pdf1)
print(bin_edges1)

cdf1 = np.cumsum(pdf1)
plt.plot(bin_edges1[1:],pdf1,label='No')
plt.plot(bin_edges1[1:],cdf1,label='No')
plt.xlabel('nodes')
plt.legend()
# 56.25% in the node range 0-5.2 : No survival
plt.show()

sns.boxplot(x='status',y='nodes',data=haberman)
plt.show()
sns.boxplot(x='status',y='age',data=haberman)
plt.show()
sns.boxplot(x='status',y='year',data=haberman)
plt.show()

sns.violinplot(x='status',y='nodes',data=haberman,size =8)
plt.show()
sns.violinplot(x='status',y='age',data=haberman,size=8)
plt.show()
sns.violinplot(x='status',y='year',data=haberman,size=8)

plt.show()