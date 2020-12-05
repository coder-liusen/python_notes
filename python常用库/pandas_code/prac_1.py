
'''
pandas练习
'''

import numpy as np
import pandas as pd


pd.set_option('display.max_columns', None) #显示所有列
pd.set_option('display.max_rows', None) #显示所有行

## 生成DataFrame数据
dates = pd.date_range('20190101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

##
