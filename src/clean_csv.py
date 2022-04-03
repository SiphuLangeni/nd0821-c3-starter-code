import pandas as pd


df = pd.read_csv('data/census.csv', skipinitialspace=True)
df.to_csv('data/census_clean.csv', index=False)