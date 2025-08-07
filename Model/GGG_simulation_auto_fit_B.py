import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.signal import find_peaks
from scipy.optimize import minimize
import time

# --- 1. グローバル物理定数と定数行列 ---
# これらは不変なので、グローバルスコープに配置
kB = 1.380649e-23; muB = 9.274010e-24; hbar = 1.054571e-34; c = 299792458
mu0 = 4.0 * np.pi * 1e-7
s = 3.5
g_factor = 1.95
N_spin = 24 / 1.238 * 1e27  # スピン数密度
G0 = mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)

# 結晶場定数
B4_0 = 0.8 / 240 ; B6_0 = 0.04 / 5040
B4_param = 0.606 ; B6_param = -1.513 
B4 = B4_0 * B4_param; B6 = B6_0 * B6_param
O04 = 60 * np.diag([7,-13,-3,9,9,-3,-13,7])
X_O44 = np.zeros((8,8)); X_O44[3,7],X_O44[4,0]=np.sqrt(35),np.sqrt(35); X_O44[2,6],X_O44[5,1]=5*np.sqrt(3),5*np.sqrt(3); O44=12*(X_O44+X_O44.T)
O06 = 1260 * np.diag([1,-5,9,-5,-5,9,-5,1]); X_O46 = np.zeros((8,8)); X_O46[3,7],X_O46[4,0]=3*np.sqrt(35),3*np.sqrt(35); X_O46[2,6],X_O46[5,1]=-7*np.sqrt(3),-7*np.sqrt(3); O46=60*(X_O46+X_O46.T)

#Sz演算子の定義
m_values = np.arange(s, -s - 1, -1)
Sz = np.diag(m_values)

# ★★★モデルを調整するためのパラメータ★★★
d_init = 0.1578e-3  # Elijahが設計した膜厚
eps_bg_init = 12 # Elijahが設計した屈折率に基づく背景誘電率(eps_bg = 3.8^2)
# ---------------------------------------------------------

# --- 2. 物理モデルクラスの定義 ---
class PhysicsModel:
    """
    物理パラメータを保持し、透過スペクトルを計算するクラス
    """
    def __init__(self, d, eps_bg, gamma, a_param):
        # 💡 パラメータをクラスの属性として保持
        self.d = d
        self.eps_bg = eps_bg
        self.gamma = gamma
        self.a_param = a_param

    def get_hamiltonian(self, B_ext_z):
        H_cf = (B4 * kB) * (O04 + 5 * O44) + (B6 * kB) * (O06 - 21 * O46)
        H_zee = g_factor * muB * B_ext_z * Sz
        return H_cf + H_zee

    def calculate_susceptibility(self, omega_array, H, T):
        eigenvalues, _ = np.linalg.eigh(H)
        eigenvalues -= np.min(eigenvalues)
        Z = np.sum(np.exp(-eigenvalues / (kB * T)))
        populations = np.exp(-eigenvalues / (kB * T)) / Z
        delta_E = eigenvalues[1:] - eigenvalues[:-1]
        delta_pop = populations[1:] - populations[:-1]
        omega_0 = delta_E / hbar
        m_vals_trans = np.arange(s, -s, -1)
        transition_strength = (s + m_vals_trans) * (s - m_vals_trans + 1)
        numerator = G0 * delta_pop * transition_strength
        denominator = (omega_0[:, np.newaxis] - omega_array) - (1j * self.gamma)
        chi_array = np.sum(numerator[:, np.newaxis] / denominator, axis=0)
        return -self.a_param * chi_array

    def calculate_transmission_intensity(self, omega_array, mu_r_array):
        n_complex = np.sqrt(self.eps_bg * mu_r_array + 0j)
        impe = np.sqrt(mu_r_array / self.eps_bg + 0j)
        lambda_0 = np.full_like(omega_array, np.inf, dtype=float)
        nonzero_mask = omega_array != 0
        lambda_0[nonzero_mask] = (2 * np.pi * c) / omega_array[nonzero_mask]
        delta = 2 * np.pi * n_complex * self.d / lambda_0
        r = (impe - 1) / (impe + 1)
        numerator = 4 * impe * np.exp(1j * delta) 
        denominator = (1 + impe)**2 - (impe - 1)**2 * np.exp(2j * delta)
        t = np.divide(numerator, denominator, where=np.abs(denominator)>1e-12, out=np.ones_like(denominator, dtype=complex)) #分母が0に近い場合は1
        return t

    def get_spectrum(self, omega_array, T, B, model_type):
        """指定された条件でスペクトルを計算するメインメソッド"""
        H_B = self.get_hamiltonian(B)
        chi_B = self.calculate_susceptibility(omega_array, H_B, T)
        
        if model_type == 'H_form':
            mu_r_B = 1 + chi_B
        elif model_type == 'B_form':
            mu_r_B = np.divide(1, 1 - chi_B, where=(1 - chi_B)!=0, out=np.full_like(chi_B, np.inf, dtype=complex))
        else:
            raise ValueError("Unknown model_type")
            
        T_B = self.calculate_transmission_intensity(omega_array, mu_r_B)
        return np.abs(T_B)**2

# --- 3. メイン実行ブロック ---
if __name__ == '__main__':
    # --- 3.1 実験データの読み込み ---
    print("実験データを読み込みます...")
    file_path = "Circular_Polarization_B_Field.xlsx"; sheet_name = 'Sheet2'
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    exp_freq_thz = df['Frequency (THz)'].to_numpy(dtype=float)
    exp_transmittance_7_7 = df['Transmittance (7.7T)'].to_numpy(dtype=float)
    min_exp, max_exp = np.min(exp_transmittance_7_7), np.max(exp_transmittance_7_7)
    exp_transmittance_7_7_normalized = (exp_transmittance_7_7 - min_exp) / (max_exp - min_exp)
    print(f"データの読み込みに成功しました。")

    # --- 3.2 計算用周波数グリッドの準備 ---
    omega_hz = np.linspace(0.1*1e12, np.max(exp_freq_thz)*1e12, 500)
    omega_rad_s = omega_hz * 2 * np.pi
    freq_thz = omega_hz / 1e12
    T_fixed = 35.0

    # --- 4. 自動最適化 ---
    param_keys = ['d', 'eps_bg', 'gamma', 'a_param']
    p_initial = [d_init, eps_bg_init, 0.11e12, 1.0]
    bounds = [(0.10e-3, 0.20e-3), (12.0, 15.0), (0.5e12, 1e12), (0.8, 2.0)]

    def cost_function(p_array, model_type):
        """最適化用のコスト関数。クラスを利用して計算を簡潔化"""
        # 💡 minimizeから渡される配列を、クラスで使いやすい辞書に変換
        p_dict = dict(zip(param_keys, p_array))
        
        # 💡 クラスをインスタンス化
        model = PhysicsModel(**p_dict)
        
        # 💡 クラスのメソッドを呼び出してスペクトルを計算
        delta_T_fit = model.get_spectrum(omega_rad_s, T_fixed, 7.7, model_type)
        
        min_th, max_th = np.min(delta_T_fit), np.max(delta_T_fit)
        delta_T_normalized = (delta_T_fit - min_th) / (max_th - min_th) if (max_th - min_th) > 1e-9 else np.zeros_like(delta_T_fit)
        
        theory_interp = np.interp(exp_freq_thz, freq_thz, delta_T_normalized)
        error = np.sum((theory_interp - exp_transmittance_7_7_normalized)**2)
        # 各p_arrayでの計算結果
        # print(f"Model: {model_type}, Error: {error:.6f}")
        return error

    results_dict = {}
    for model_name in ['H_form', 'B_form']:
        print(f"\n--- [{model_name}]モデルの自動最適化を開始します ---")
        result = minimize(cost_function, p_initial, args=(model_name,), method='L-BFGS-B', bounds=bounds)
        results_dict[model_name] = result

    # --- 5. 結果の表示と最終プロット ---
    print("\n\n=== 全モデルの最適化結果まとめ ===")
    # (結果のターミナル表示部分は変更なし)
    for model_name, result in results_dict.items():
        if result.success:
            print(f"\n🎉 [{model_name}] モデル 最適化成功！ 最小二乗誤差: {result.fun:.4f}")
            for key, val in zip(param_keys, result.x):
                print(f"  {key:<10} = {val:.4e}")
        else:
            print(f"\n⚠️ [{model_name}] モデル 最適化失敗: {result.message}")

    # グラフ描画
    print("\n実験値、手動設定、自動最適化の結果を比較するグラフを作成します...")
    fig_final, ax_final = plt.subplots(figsize=(12, 8))

    # 5.1. 実験データをプロット (ベースライン)
    ax_final.plot(exp_freq_thz, exp_transmittance_7_7_normalized, 'o', color='black', markersize=5, label='実験データ')

    # 5.2. 手動設定時の理論値を計算してプロット (H/B両形式)
    manual_params_dict = dict(zip(param_keys, p_initial))
    manual_model = PhysicsModel(**manual_params_dict)
    manual_colors = {'H_form': 'cyan', 'B_form': 'limegreen'}

    for model_name in ['H_form', 'B_form']:
        manual_spectrum = manual_model.get_spectrum(omega_rad_s, T_fixed, 7.7, model_type=model_name)
        min_man, max_man = np.min(manual_spectrum), np.max(manual_spectrum)
        manual_spectrum_normalized = (manual_spectrum - min_man) / (max_man - min_man)
        ax_final.plot(freq_thz, manual_spectrum_normalized, color=manual_colors[model_name], linestyle='--', linewidth=2.5, label=f'手動設定の理論値 ({model_name})')

    # 5.3. 自動最適化後の理論値をプロット
    auto_colors = {'H_form': 'blue', 'B_form': 'darkorange'}
    for model_name, result in results_dict.items():
        if result.success:
            p_opt_dict = dict(zip(param_keys, result.x))
            final_model = PhysicsModel(**p_opt_dict)
            final_spectrum = final_model.get_spectrum(omega_rad_s, T_fixed, 7.7, model_name)
            min_opt, max_opt = np.min(final_spectrum), np.max(final_spectrum)
            final_spectrum_normalized = (final_spectrum - min_opt) / (max_opt - min_opt)
            ax_final.plot(freq_thz, final_spectrum_normalized, color=auto_colors[model_name], linestyle='-', linewidth=2.5, label=f'自動最適化後の理論値 ({model_name})')

    # グラフの体裁を整える
    ax_final.set_title('実験値、手動設定、自動最適化の比較プロット', fontsize=16)
    ax_final.set_xlabel('周波数 (THz)', fontsize=12)
    ax_final.set_ylabel('正規化透過率 $T(B)$', fontsize=12)
    ax_final.legend(fontsize=11)
    ax_final.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.close()
