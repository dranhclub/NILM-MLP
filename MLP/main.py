import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MLPClassifier import MLPClassifier

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

select_device = ['light', 'socket', 'heater', 'aircond1', 'aircond2', 'aircond3', 'indcooker']

p_sum = p_df[select_device].sum(axis=1).to_numpy()
q_sum = q_df[select_device].sum(axis=1).to_numpy()
u_sum = u_df['u'].to_numpy()  # không cần tính tổng U, nhưng cứ đặt là u_sum cho đồng nhất
i_sum = i_df[select_device].sum(axis=1).to_numpy() * 100  # Nhân với 100 để I không quá nhỏ

plt.title("Công suất P theo thời gian")
plt.plot(p_sum)
plt.show()

# -----------------PREPARE DATA-------------------
data = []
label = []
threshold = 20  # ngưỡng xác định thiết bị là bật > 20W
for i in range(1, len(p_sum)):
    delta_p = p_sum[i] - p_sum[i - 1]
    delta_q = q_sum[i] - q_sum[i - 1]
    data.append([u_sum[i], i_sum[i], p_sum[i], q_sum[i], delta_p, delta_q])
    x = 0
    for j, device_name in enumerate(select_device):
        if p_df[device_name].iloc[i] > threshold:
            x += 2 ** j

    label.append(x)

print("num data point =", len(data))


def tobase2(n):
    length = len(select_device)
    ret = [0 for i in range(length)]
    i = length - 1
    while n > 0:
        ret[i] = n % 2
        n = n // 2
        i -= 1
    return ret


def train_test_split(X, y, test_size=0.3):
    id = int(len(X) * (1 - test_size))
    return X[:id], X[id:], y[:id], y[id:]


X = np.array(data)
y = np.array([tobase2(label[i]) for i in range(len(label))])

model = MLPClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model.fit(X_train, y_train)
