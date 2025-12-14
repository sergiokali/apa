import control
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
s = control.TransferFunction.s

k = 0.1
T1 = 0.1
T2 = 0.1
#G = k / ((T1*s +1)(T2*s +1))
#(G)
# step should be every 5ms 30s / 0.005 s
NUM_SAMPLES = 1000
DURATION = 60
TIME_STEPS  = int(DURATION / 0.1)
FEATURES = 3
print(TIME_STEPS)

t = np.linspace(start=-1,stop=DURATION-1, num=TIME_STEPS)
print(t)

K = 0 
T1 = 0
T2 = 0
#data shape (1000, 300, 3)
# 1000 experiments with different parameters
# 300 time steps per experiment 
# 3 features (x_e, x_a, t)

X_data = np.zeros((NUM_SAMPLES, TIME_STEPS, FEATURES))
# 3 outputs (K, T1, T2)
Y_data = np.zeros((NUM_SAMPLES, 3))
features_list = []
labels_list = []
for i in tqdm(range(NUM_SAMPLES)):
    K = np.round(np.random.uniform(1, 50),3)
    val_1 = np.round(np.random.uniform(1, 15),3)
    val_2 = np.round(np.random.uniform(1, 15),3)
    # T1 always bigger to help NN learn consistently
    T1 = max(val_1, val_2)
    T2 = min(val_1, val_2)

    num = [K]
    den = [T1*T2, T1 + T2, 1]
    print(f"K= {K}, T1={T1}, T2= {T2}")
    features = pd.DataFrame(index=t)
    features['t'] = t
    features['x_e'] = np.zeros(TIME_STEPS)
    features.loc[features.index > 0, 'x_e'] = 1

    sys = control.tf(num, den)
    _, x_a = control.forced_response(sys, T=t, U=features['x_e'])
    features['x_a'] = x_a

    features = features[['x_e', 'x_a', 't']]
    features_list.append(features)
    labels_list.append([K, T1, T2])

# Assuming X_final and Y_final are your arrays from the generation script
X_final = np.array([df.values for df in features_list]) 
print("Saving data...")

# Save features
np.save('pt2/X_data.npy', X_final)

# Save labels
np.save('pt2/Y_data.npy', labels_list)

print("Saved successfully to 'X_data.npy' and 'Y_data.npy'")


