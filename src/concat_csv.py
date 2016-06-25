import glob
import pandas as pd

files = glob.glob('./csvfiles/concat/*.csv')
print files

result = []
for csv in files:
    df = pd.read_csv(csv, index_col=0, header=0)
    #df.drop('index', axis=1, inplace=True)
    #df = df.drop(df.columns[[0]], axis=1)

    result.append(df)

df = pd.concat(result, ignore_index=True)
df.to_csv('concat.csv', columns=['seriesuid','coordX','coordY','coordZ','probability'])


quit()

df = pd.concat(pd.read_csv(f, index_col=0, header=0,names=['index','seriesuid', 'coordX','coordY','coordZ','probability']).drop('index', axis=1, inplace=True) for f in files)
df.drop('index', axis=1, inplace=True)
print df
df.to_csv('concat.csv', columns=['seriesuid','coordX','coordY','coordZ','probability'])
