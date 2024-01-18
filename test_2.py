import numpy as np
import pandas as pd


x = np.random.randn(5)
print(type(x))
y = np.sin(x)
print(type(y))
print(y)
df = pd.DataFrame({'x':x, 'y':y})
df.plot('x', 'y', kind='scatter')
