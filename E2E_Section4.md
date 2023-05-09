```python
import numpy as np
import pandas as pd
import os
import warnings
from tqdm import tqdm
import math
import seaborn as sns

from scipy.stats import gamma
import datetime as dt
```


```python
# df_test = pd.read_pickle("df_test.pkl")
df_test = pd.read_csv("df_test.csv")
len(df_test)
```




    3000




```python
#Correct data type
df_test['sf_rnn'] = df_test['sf_rnn'].apply(lambda x: x.strip('][').split(', '))
df_test['sf_rnn'] = df_test['sf_rnn'].apply(lambda x: pd.to_numeric(x))
df_test['demand_hist'] = df_test['demand_hist'].apply(lambda x: pd.to_numeric(x.strip('][').split()))
```

#### PTO benchmark1 and benchmark 2


```python
df_test['Bm1_pred'] = df_test['sf_rnn'].apply(lambda x: np.mean(x))
df_test['Bm1_pred'] = df_test['Bm1_pred'] * (df_test['review_period'] + df_test['vlt']).astype(int)

b = 9
h = 1
def get_bm2(x):
    rl = x['review_period'] + x['vlt']
    if rl <= b:
        days = int(rl)
    else:
        days = int(rl) - rl//(b+h)
    return x['Bm2_pred'] * days

df_test['Bm2_pred'] = df_test['sf_rnn'].apply(lambda x: np.mean(x))
df_test['Bm2_pred'] = df_test.apply(get_bm2, axis=1)
```

#### Normal benchmark


```python
Z90 = 1.64
df_test['Normal_pred'] = df_test.apply(lambda x: int(x['demand_mean']*(x['review_period']+x['vendor_vlt_mean'])
                                       +Z90*np.sqrt((x['review_period']+x['vendor_vlt_mean'])*x['demand_std']**2
                                                    + x['demand_std']**2 * x['vendor_vlt_std'])), axis=1)
```

#### Gamma benchmark


```python
def gamma_base(x):
    mean = x['demand_mean']
    var = x['demand_std']**2
    theta = var/(mean+1e-4)
    k = mean/(theta+1e-4)
    k_sum = int(x['review_period']+x['vendor_vlt_mean'])*k
    gamma_stock = gamma.ppf(0.9, a=k_sum, scale = theta)
    if(np.isnan(gamma_stock)):
        return 0
    else:
        return int(gamma_stock)
df_test['Gamma_pred'] = df_test.apply(gamma_base, axis=1)
```


```python
df_test = df_test.groupby('SKU').agg(lambda x: x.tolist())
```

### Sequential test model


```python
def get_agginv(x, name):
    inv1, inv2 = [x['initial_stock'][0]], []
    rd = len(x['OPT_pred'])
 
    for r in range(rd):
        if r < rd - 1:
            len_day = len(x['demand_hist'][r])-1
        else:
            len_day = len(x['demand_hist'][r])
        for t in range(len_day):
            if t == 0:
                if r == 0:
                    replen = int(round(x[name+'_pred'][r] - inv1[0]))
                else:
                    try:
                        replen = int(round(x[name+'_pred'][r] - inv1[-int(round(x['vlt'][r]))-1]))
                    except:
                        replen = int(round(x[name+'_pred'][r] - inv1[1]))
            if t < int(round(x['vlt'][r])):
                if r == 0:
                    inv1.append(inv1[-1] - x['demand_hist'][r][t])
            elif t == int(round(x['vlt'][r])):
                if inv1[-1] >= 0:
                    inv_ = inv1[-1] + replen - x['demand_hist'][r][t]
                else:
                    inv_ = replen - x['demand_hist'][r][t]
                inv1.append(inv_)
                inv2.append(inv_)
            else:
                inv_ = inv1[-1] - x['demand_hist'][r][t]
                inv1.append(inv_)
                inv2.append(inv_)
    
    inv1 = inv1[1:]
    return [inv1, inv2]
```


```python
df_test['OPT_agginv_f'], df_test['OPT_agginv'] = zip(*df_test.apply(get_agginv, name='OPT',  axis=1))
df_test['E2E_RNN_agginv_f'], df_test['E2E_RNN_agginv'] = zip(*df_test.apply(get_agginv, name='E2E_RNN',  axis=1))
df_test['Bm1_agginv_f'], df_test['Bm1_agginv'] = zip(*df_test.apply(get_agginv, name='Bm1',  axis=1))
df_test['Bm2_agginv_f'], df_test['Bm2_agginv'] = zip(*df_test.apply(get_agginv, name='Bm2',  axis=1))
df_test['Normal_agginv_f'], df_test['Normal_agginv'] = zip(*df_test.apply(get_agginv, name='Normal',  axis=1))
df_test['Gamma_agginv_f'], df_test['Gamma_agginv'] = zip(*df_test.apply(get_agginv, name='Gamma',  axis=1))
df_test['gbm_agginv_f'], df_test['gbm_agginv'] = zip(*df_test.apply(get_agginv, name='gbm',  axis=1))
```

### Calculate cost


```python
h = 1
b = 9
str_list = ['OPT', 'E2E_RNN', 'Bm1', 'Bm2', 'Normal', 'Gamma', 'gbm']
numberOfRows = len(df_test)
df_cost_agg = pd.DataFrame(index=np.arange(0, numberOfRows), columns=str_list)
df_holding_agg = pd.DataFrame(index=np.arange(0, numberOfRows), columns=str_list)
df_back_agg = pd.DataFrame(index=np.arange(0, numberOfRows), columns=str_list)
df_stockout_agg = pd.DataFrame(index=np.arange(0, numberOfRows), columns=str_list)
df_turnover_agg = pd.DataFrame(index=np.arange(0, numberOfRows), columns=str_list)


df_test_ = df_test.reset_index(drop=True)
for str1 in str_list:
    str2 = str1 + '_agginv'
    df_holding_agg[str1] = df_test_[str2].apply(lambda x: h * sum([inv for inv in x if inv>0]) )
    df_back_agg[str1] = df_test_[str2].apply(lambda x: b * -sum([inv for inv in x if inv<0]) )
    df_cost_agg[str1] = df_holding_agg[str1] + df_back_agg[str1] 
```


```python
df_aggcom = pd.DataFrame({'Total cost': df_cost_agg[str_list].mean(),
             'Holding cost': df_holding_agg[str_list].mean(),
             'Stockout cost': df_back_agg[str_list].mean()}).T
df_aggcom
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OPT</th>
      <th>E2E_RNN</th>
      <th>Bm1</th>
      <th>Bm2</th>
      <th>Normal</th>
      <th>Gamma</th>
      <th>gbm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Total cost</th>
      <td>2572.964327</td>
      <td>3036.758343</td>
      <td>3842.305332</td>
      <td>3837.828539</td>
      <td>3961.691600</td>
      <td>4099.949751</td>
      <td>3424.869965</td>
    </tr>
    <tr>
      <th>Holding cost</th>
      <td>1363.218642</td>
      <td>1883.591484</td>
      <td>1709.364020</td>
      <td>1541.630610</td>
      <td>2317.097814</td>
      <td>1947.838128</td>
      <td>1424.929804</td>
    </tr>
    <tr>
      <th>Stockout cost</th>
      <td>1209.745685</td>
      <td>1153.166858</td>
      <td>2132.941312</td>
      <td>2296.197929</td>
      <td>1644.593786</td>
      <td>2152.111623</td>
      <td>1999.940161</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df_aggcom.to_latex(float_format=lambda x: '%.2f' % x))
```

    \begin{tabular}{lrrrrrrr}
    \toprule
    {} &     OPT &  E2E\_RNN &     Bm1 &     Bm2 &  Normal &   Gamma &     gbm \\
    \midrule
    Total cost    & 2572.96 &  3036.76 & 3842.31 & 3837.83 & 3961.69 & 4099.95 & 3424.87 \\
    Holding cost  & 1363.22 &  1883.59 & 1709.36 & 1541.63 & 2317.10 & 1947.84 & 1424.93 \\
    Stockout cost & 1209.75 &  1153.17 & 2132.94 & 2296.20 & 1644.59 & 2152.11 & 1999.94 \\
    \bottomrule
    \end{tabular}
    


    /var/folders/8c/bv99bkt927d5cb60c1q8n6b00000gp/T/ipykernel_47485/1135118302.py:1: FutureWarning: In future versions `DataFrame.to_latex` is expected to utilise the base implementation of `Styler.to_latex` for formatting and rendering. The arguments signature may therefore change. It is recommended instead to use `DataFrame.style.to_latex` which also contains additional functionality.
      print(df_aggcom.to_latex(float_format=lambda x: '%.2f' % x))



```python

```


```python

```
