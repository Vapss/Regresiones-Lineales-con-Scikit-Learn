import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
df = pd.read_csv('cal_housing.csv', sep=',', engine='python')

for i in len(df):
    print(df.columns[i])

X = df[i].values
y = df['price'].values
plt.scatter(X,y)
plt.xlabel('num_bedrooms')
plt.ylabel('price')

#~ plt.show()

#~ #Correlación 
#~ X = df.drop(['price'],axis=1).values
#~ print (X)

correlation_matrix = df.corr().round(2)
print(correlation_matrix)
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()
