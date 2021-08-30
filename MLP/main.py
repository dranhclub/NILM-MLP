import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MyMLPRegressor import MyMLPRegressor

plt.rcParams['figure.figsize'] = (20, 4)

# ----------READ DATA-------------
# Power dataframe
p_df = pd.read_csv('../data/W.csv',
                   names=['time', 'light', 'socket', 'heater', 'aircond1', 'aircond2', 'aircond3', 'indcooker'],
                   header=0)
# Voltage dataframe
u_df = pd.read_csv('../data/V.csv',
                   names=['time', 'u'],
                   header=0)
# Current dataframe
i_df = pd.read_csv('../data/A.csv',
                   names=['time', 'light', 'socket', 'heater', 'aircond1', 'aircond2', 'aircond3', 'indcooker'],
                   header=0)
# Power factor dataframe
pf_df = pd.read_csv('../data/cosphi.csv',
                    names=['time', 'light', 'socket', 'heater', 'aircond1', 'aircond2', 'aircond3', 'indcooker'],
                    header=0)

# Reactive power dataframe
q_df = pd.DataFrame()
q_df['time'] = p_df['time']
column_names = ['light', 'socket', 'heater', 'aircond1', 'aircond2', 'aircond3', 'indcooker']

# Calculate reactive power using P and cosphi
for col_name in column_names:
    q_df[col_name] = np.tan(np.arccos(pf_df[col_name])) * p_df[col_name]

select_device = ['heater', 'indcooker', 'aircond1']

p_sum = p_df[select_device].sum(axis=1).to_numpy()
q_sum = q_df[select_device].sum(axis=1).to_numpy()
u_sum = u_df['u'].to_numpy()  # không cần tính tổng U, nhưng cứ đặt là u_sum cho đồng nhất
i_sum = i_df[select_device].sum(axis=1).to_numpy() * 100  # Nhân với 100 để I không quá nhỏ

# plt.title("Công suất P theo thời gian")
# plt.plot(p_sum)
# plt.show()

data = []
label = []
for t in range(0, len(p_sum)):
    data.append([u_sum[t], i_sum[t], p_sum[t], q_sum[t]])
    percent = []
    for j, device_name in enumerate(select_device):
        if p_sum[t] == 0:
            percent.append(0)
        else:
            percent.append(p_df[device_name].iloc[t] / p_sum[t])
    label.append(percent)

print("num data point =", len(data))

X = np.array(data)
y = np.array(label)
print(f'{X.shape=}')
print(f'{y.shape=}')
print(f'{X[0:10]=}')
print(f'{y[0:10]=}')

model = MyMLPRegressor(
                    hidden_layer_sizes=(30,),
                    learning_rate=0.01,
                    random_state=0,
                    max_iter=100000,
                    n_iter_no_change=50,
                    tol=0.001,
                    hidden_activation='tanh',
                    output_activation='sigmoid',
                    solver='adam',
                    loss_func='mse',
                    batch_size=2000)
model.fit(X, y)
score = model.score(X, y)
print("Score=", score)

yhat = model.predict(X)
a = 8000
b = 9000
for i in range(y.shape[1]):
    plt.plot(y[a:b, i] * p_sum[a:b])
    plt.plot(yhat[a:b, i] * p_sum[a:b])
    plt.title(select_device[i])
    plt.show()