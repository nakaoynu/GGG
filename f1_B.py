# 5種類の温度に対するスペクトルの表示(B)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = "Meiryo"
from scipy.signal import find_peaks

# 行列の定義
X = np.array([
    [7, 0, 0, 0, 0, 0, 0, 0],
    [0, -13, 0, 0, 0, 0, 0, 0],
    [0, 0, -3, 0, 0, 0, 0, 0],
    [0, 0, 0, 9, 0, 0, 0, 0],
    [0, 0, 0, 0, 9, 0, 0, 0],
    [0, 0, 0, 0, 0, -3, 0, 0],
    [0, 0, 0, 0, 0, 0, -13, 0],
    [0, 0, 0, 0, 0, 0, 0, 7]
])

factor_04 = 60
O04 = factor_04 * X

X = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 5 * np.sqrt(3), 0],
    [0, 0, 0, 0, 0, 0, 0, np.sqrt(35)],
    [np.sqrt(35), 0, 0, 0, 0, 0, 0, 0],
    [0, 5 * np.sqrt(3), 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

factor_44 = 12
O44 = factor_44 * (X + X.T)

X = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, -5, 0, 0, 0, 0, 0, 0],
    [0, 0, 9, 0, 0, 0, 0, 0],
    [0, 0, 0, -5, 0, 0, 0, 0],
    [0, 0, 0, 0, -5, 0, 0, 0],
    [0, 0, 0, 0, 0, 9, 0, 0],
    [0, 0, 0, 0, 0, 0, -5, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
])

factor_06 = 1260
O06 = factor_06 * X

X = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -7 * np.sqrt(3), 0],
    [0, 0, 0, 0, 0, 0, 0, 3 * np.sqrt(35)],
    [3 * np.sqrt(35), 0, 0, 0, 0, 0, 0, 0],
    [0, -7 * np.sqrt(3), 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

factor_46 = 60
O46 = factor_46 * (X + X.T)

# スピン演算子
Sz = np.diag([3.5, 2.5, 1.5, 0.5, -0.5, -1.5, -2.5, -3.5])

# Onnに与えられた係数
B4 = 0.8 / 240  # [K]
B6 = 0.04 / 5040  # [K]

# 係数調整のための倍率(手動設定)
factor_b4 = 0.6060886
factor_b6 = -1.513

B4 *= factor_b4
B6 *= factor_b6

g = 1.95  # ｇ因子
eps_BG = 11.5  # GGGの背景誘電率
c = 299792458  # 光速 [m/s]
kB = 1.381e-23  # ボルツマン定数 [J/K]
muB = 9.274e-24  # ボーア磁子 [J/T]
hbar = 1.055e-34  # ディラック定数 [J・s]
T_array = [35, 50, 75, 95, 200, 250]  # 温度 [K]
B_array = [0.0, 7.8]  # 磁場 [T]
g0_const = 6.0e11  
gamma = 2.5e11  # 緩和 [Hz]
d = 0.1578e-3  # サンプルの厚み [m]

# ω配列と値の初期化
omega_array = []
value_vector = [[] for _ in range(6)]

# ポピュレーション関数
def Pm(Em, Z, T):
    return np.exp(-Em / (kB * T)) / Z

for i, T in enumerate(T_array):
    value_array = []
    for omega in np.arange(0.1, 7e12, 0.1e11):
        # 背景透過係数の計算
        B = B_array[0]
        g0 = 0.0
        H_CF = (B4 * kB) * (O04 + 5.0 * O44) + (B6 * kB) * (O06 - 21.0 * O46)
        H_Zee = (g * muB * B) * Sz
        H = H_CF + H_Zee

        eigenvalues, eigenvectors = np.linalg.eigh(H)
        Z = np.sum(np.exp(-eigenvalues / (kB * T)))

        chi = 0.0
        s = 3.5

        for num, m in enumerate(np.arange(-3.5, 4.0, 1.0)):
            if num + 1 < len(eigenvalues):
                delta_E = eigenvalues[num + 1] - eigenvalues[num]
                omega_0 = delta_E / hbar
                chi +=  (4 * g0**2 / (2 * np.pi* 2.5e11) * (s + m) * (s - m + 1) * (Pm(eigenvalues[num + 1], Z, T) - Pm(eigenvalues[num], Z, T))
                            / ((omega_0) - omega - (1j * gamma / 2)))
        chi = -chi     
        mu = 1 / (1 - chi)
        n = np.sqrt(eps_BG * mu)
        impe = np.sqrt(mu / eps_BG)
        lamda = c / (omega / (2 * np.pi))
        delta = 2 * np.pi * n * d / lamda
        t = np.exp(delta * 1j) * 4 * impe / (1 + impe)**2
        t_BG = t
        

        # 実際の透過率の計算
        B = B_array[1]
        g0 = np.sqrt(B) * g0_const / np.sqrt(21.5)
        H_Zee = (g * muB * B) * Sz
        H = H_CF + H_Zee

        eigenvalues, eigenvectors = np.linalg.eigh(H)
        Z = np.sum(np.exp(-eigenvalues / (kB * T)))

        chi = 0.0
        for num, m in enumerate(np.arange(-3.5, 4.0, 1.0)):
            if num + 1 < len(eigenvalues):
                delta_E = eigenvalues[num + 1] - eigenvalues[num]
                omega_0 = delta_E / hbar
                chi +=  (4 * g0**2 / (2 * np.pi * 2.5e11) * (s + m) * (s - m + 1) * (Pm(eigenvalues[num + 1], Z, T) - Pm(eigenvalues[num], Z, T))
                            / ((omega_0) - omega - (1j * gamma / 2)))
        chi = -chi        
        mu = 1 / (1 - chi)
        n = np.sqrt(eps_BG * mu)
        impe = np.sqrt(mu / eps_BG)
        lamda = c / (omega / (2 * np.pi))
        delta = 2 * np.pi * n * d / lamda
        t = np.exp(delta * 1j ) * 4 * impe / (1 + impe)**2
        y = abs(t - t_BG)**2
        value_array.append(y)

        if i == 0:
            omega_array.append(omega / (2 * np.pi * 1e12))

    value_vector[i] = (
        (np.array(value_array) - np.min(value_array)) / (np.max(value_array) - np.min(value_array)) + i
    )

# グラフ描画
for i, values in enumerate(value_vector):
    plt.plot(omega_array, values, label=f"T={T_array[i]} K")
    
    # ピーク検出
    peaks, _ = find_peaks(values, width=6)
    peak_frequencies = [omega_array[p] for p in peaks]
    peak_values = [values[p] for p in peaks]
    
    # ピークをプロット
    plt.scatter(peak_frequencies, peak_values, color='red')

#plt.title(f"B={B_array[1]} T (B)")
plt.xlabel("周波数 (THz)")
plt.ylabel("$|t-t_{BG}|^2$")
plt.legend()
plt.savefig('f_B_spectrum.png') 

