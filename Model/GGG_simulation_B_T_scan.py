import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.signal import find_peaks

# --- 0. プロット設定 ---
plt.rcParams['figure.dpi'] = 100

# --- 1. 物理定数とパラメータ定義 ---
kB = 1.380649e-23; muB = 9.274010e-24; hbar = 1.054571e-34; c = 299792458
s = 3.5; g_factor = 1.95
mu0 = 4.0 * np.pi * 1e-7; N_spin = 24/1.238 * 1e27
G0 = mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)

# 結晶場定数
B4_0 = 0.8 / 240 ; B6_0 = 0.04 / 5040
B4_param = 0.606 ; B6_param = -1.513 
B4 = B4_0 * B4_param; B6 = B6_0 * B6_param
O04 = 60 * np.diag([7,-13,-3,9,9,-3,-13,7])
X_O44 = np.zeros((8,8)); X_O44[3,7],X_O44[4,0]=np.sqrt(35),np.sqrt(35); X_O44[2,6],X_O44[5,1]=5*np.sqrt(3),5*np.sqrt(3); O44=12*(X_O44+X_O44.T)
O06 = 1260 * np.diag([1,-5,9,-5,-5,9,-5,1]); X_O46 = np.zeros((8,8)); X_O46[3,7],X_O46[4,0]=3*np.sqrt(35),3*np.sqrt(35); X_O46[2,6],X_O46[5,1]=-7*np.sqrt(3),-7*np.sqrt(3); O46=60*(X_O46+X_O46.T)

# ★★★モデルを調整するためのパラメータ★★★
d = 0.1578e-3 * 1.05 # Elijahが設計した膜厚
eps_bg = 13 # Elijahが設計した屈折率に基づく背景誘電率(eps_bg = 3.8^2)
gamma0 = np.ones(7) * 0.11e12  # 初期値としてのgamma値（Hz単位）
gamma_H = gamma0 * np.array([0.9, 0.9, 0.9, 1, 1, 0.9, 0.9])  # H_formのgamma値（遷移依存あり）
gamma_B = gamma0 * np.array([1, 1, 1, 1, 1, 1, 1])  # B_formのgamma値(遷移依存あり)
print(f"B形式のgamma値: {gamma_B}")
print(f"H形式のgamma値: {gamma_H}")
# モデルごとのパラメータ定義
params = {
        'H_form': {'a': 1.5805e+00, 'gamma': gamma_H, 'color': 'blue', 'line_style': '--'},
        'B_form': {'a': 1.5805e+00, 'gamma': gamma_B, 'color': 'orange', 'line_style': '-'},
        }

# Sz演算子の定義 
m_values = np.arange(s, -s - 1, -1)
Sz = np.diag(m_values)

# --- 2. 汎用化された物理モデル関数 ---

def get_hamiltonian(B_ext_z):
    H_cf = (B4 * kB) * (O04 + 5 * O44) + (B6 * kB) * (O06 - 21 * O46)
    H_zee = g_factor * muB * B_ext_z * Sz
    return H_cf + H_zee

def calculate_susceptibility(omega_array, H, T, gamma, a_param):
    eigenvalues, _ = np.linalg.eigh(H)
    eigenvalues -= np.min(eigenvalues)
    Z = np.sum(np.exp(-eigenvalues / (kB * T)))
    populations = np.exp(-eigenvalues / (kB * T)) / Z
    delta_E = eigenvalues[1:] - eigenvalues[:-1]
    delta_pop = populations[1:] - populations[:-1]
    omega_0 = delta_E / hbar
    m_vals = np.arange(s, -s, -1)
    transition_strength = (s + m_vals) * (s - m_vals + 1)
    
    numerator = G0 * delta_pop * transition_strength
    denominator = (omega_0[:, np.newaxis] - omega_array) - (1j * gamma[:, np.newaxis])
    chi_array = np.sum(numerator[:, np.newaxis] / denominator, axis=0)
    return -a_param * chi_array

def calculate_transmission_intensity(omega_array, mu_r_array):
    """透過係数tを計算するヘルパー関数"""
    n_complex = np.sqrt(eps_bg * mu_r_array + 0j)
    impe = np.sqrt(mu_r_array / eps_bg + 0j)
    lambda_0 = np.full_like(omega_array, np.inf, dtype=float)
    nonzero_mask = omega_array != 0; lambda_0[nonzero_mask] = (2 * np.pi * c) / omega_array[nonzero_mask]
    delta = 2 * np.pi * n_complex * d / lambda_0
    
    r = (impe - 1) / (impe + 1)
    numerator = 4 * impe * np.exp(1j * delta) / (1 + impe)**2
    denominator = 1 - r**2 * np.exp(2j * delta)
    t = np.divide(numerator, denominator, where=np.abs(denominator)>1e-12, out=np.zeros_like(denominator, dtype=complex))
    return t

def get_delta_T_spectrum(omega_array, T, B, gamma, a_param, model_type):
    """aとgammaを外部から受け取り、計算にそのまま使う"""
    H_B = get_hamiltonian(B)

    if model_type == 'H_form':
        chi_B = calculate_susceptibility(omega_array, H_B, T, gamma, a_param)
        mu_r_B = 1 + chi_B
    elif model_type == 'B_form':
        chi_B = calculate_susceptibility(omega_array, H_B, T, gamma, a_param)
        mu_r_B = np.divide(1, 1 - chi_B, where=(1 - chi_B)!=0, out=np.full_like(chi_B, np.inf, dtype=complex))
    else:
        raise ValueError("model_type must be 'H_form' or 'B_form'")
    print(f"\n 計算中: {model_type} モデル, a = {a_param}, T = {T}, B = {B}, 膜 厚d = {float(d)*1e6:.2f} μm, eps_bg = {eps_bg:.3f}")
    # print(gamma_B if model_type == 'B_form' else gamma_H)
    T_B = calculate_transmission_intensity(omega_array, mu_r_B)

    return np.abs(T_B)**2

# --- 3. メイン実行ブロック ---
if __name__ == '__main__':
    # --- 3.0 実験データの読み込み ---
    print("実験データを読み込みます...")
    file_path = "Circular_Polarization_B_Field.xlsx"; sheet_name = 'Sheet2'

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
        exp_freq_thz = df['Frequency (THz)'].to_numpy(dtype=float)
        exp_transmittance_5 = df['Transmittance (5T)'].to_numpy(dtype=float)
        exp_transmittance_7_7 = df['Transmittance (7.7T)'].to_numpy(dtype=float)
        exp_transmittance_9 = df['Transmittance (9T)'].to_numpy(dtype=float)
        print(f"データの読み込みに成功しました。読み込み件数: {len(df)}件")

        min_exp_transmittance = np.min(exp_transmittance_7_7)
        max_exp_transmittance = np.max(exp_transmittance_7_7)
        exp_transmittance_7_7_normalized = (exp_transmittance_7_7 - min_exp_transmittance) / (max_exp_transmittance - min_exp_transmittance)

    except FileNotFoundError:
        print(f"ファイル '{file_path}' が見つかりません。正しいパスを指定してください。")
        exit()

    except Exception as e:
    # その他のエラー（例：シート名が違うなど）もキャッチ
        print(f"データの読み込み中にエラーが発生しました: {e}")
        exit()

    print(df)
    print('-' * 30)

    # 周波数範囲を定義
    omega_hz = np.linspace((np.min(exp_freq_thz) - 0.03)*1e12, np.max(exp_freq_thz)*1e12, 500)
    omega_rad_s = omega_hz * 2 * np.pi
    freq_thz = omega_hz / 1e12

    # --- 3.1 磁場依存性のシミュレーション ---
    print(f"磁場依存性を計算・プロットします...")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    # まず実験データをプロット
    #ax1.plot(exp_freq_thz, exp_transmittance_5, 'o', color='black', markersize=4, label='実験データ (5T)')
    ax1.plot(exp_freq_thz, exp_transmittance_7_7_normalized, 'o', color='black', markersize=4, label='正規化された実験データ (7.7T)')
    #ax1.plot(exp_freq_thz, exp_transmittance_9, 'o', color='blue', markersize=4, label='実験データ (9T)')

    # 実験データのピーク位置を検出
    peak_indices_exp, _ = find_peaks(exp_transmittance_7_7_normalized, height=0.3)
    peak_freq_exp = exp_freq_thz[peak_indices_exp]
    peak_trans_exp = exp_transmittance_7_7_normalized[peak_indices_exp]
    ax1.scatter(peak_freq_exp, peak_trans_exp, marker='o', color='red', label='実験データのピーク (7.7T)', zorder=5)
    print("\n--- 実験データのピーク位置 ---")
    for f, t in zip(peak_freq_exp, peak_trans_exp):
        print(f"  周波数: {f:.4f} THz")

    T_fixed = 35.0  # 固定温度
    B_scan_values = [7.7]    

    # 次にシミュレーションデータをプロット
    for b_val in B_scan_values:
        for model_type, param in params.items():
            a_sim = param['a']
            gamma_sim = param['gamma'] 
            delta_T = get_delta_T_spectrum(omega_rad_s, T_fixed, b_val, gamma_sim, a_sim, model_type=model_type)

            min_th = np.min(delta_T)
            max_th = np.max(delta_T)
            if(max_th - min_th) > 1e-9:
                delta_T_normalized = (delta_T - min_th) / (max_th - min_th)
            else:
                delta_T_normalized = np.zeros_like(delta_T)
            ax1.plot(freq_thz, delta_T_normalized, label=f'理論値(B = {b_val:.1f} T, {model_type})')

            # ピーク位置を検出
            peak_indices_sim, _ = find_peaks(delta_T_normalized, height=0.3)
            peak_freq_sim = freq_thz[peak_indices_sim]
            peak_trans_sim = delta_T_normalized[peak_indices_sim]
            ax1.scatter(peak_freq_sim, peak_trans_sim, marker='x', color=param['color'], label=f'シミュレーションデータのピーク（{model_type}）', zorder=5)
            print(f"\n--- {model_type} モデルのピーク位置 (B = {b_val:.1f} T) ---")
            for f, t in zip(peak_freq_sim, peak_trans_sim):
                print(f"  周波数: {f:.4f} THz")

    ax1.set_title(f'透過スペクトル (T = {T_fixed} K, d = {d*1e6:.2f} μm)')
    ax1.set_xlabel('周波数 (THz)')
    ax1.set_ylabel('透過率 $T(B)$')
    ax1.legend()
    ax1.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()
