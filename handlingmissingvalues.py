import numpy as np
import pandas as pd

dict={'First Score':[100,90,np.nan,95],
      'Second Score':[30,45,56,np.nan],
      'Third Score':[np.nan,40,80,98]}
df=pd.DataFrame(dict)
print(df)

print(df.isnull())
print(df.notnull())
print(df.fillna(0))
print(df.fillna(method='pad'))
print(df.fillna(method='bfill'))

# Print interpolation
print(df.interpolate(method='linear', limit_direction='forward'))

# New DataFrame
dict={'First Score':[100,90,np.nan,95],
      'Second Score':[30,np.nan,45,56],
      'Third Score':[52,40,80,98],
      'Fourth Score':[np.nan,np.nan,np.nan,65]}
df=pd.DataFrame(dict)

# Drop rows containing any NaN
print(df.dropna())
