import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.signal import find_peaks
from scipy.optimize import minimize
import time

# --- 1. グローバル物理定数と定数行列 ---
kB = 1.380649e-23; muB = 9.274010e-24; hbar = 1.054571e-34; c = 299792458
mu0 = 4.0 * np.pi * 1e-7
s = 3.5
g_factor = 1.95
N_spin = 24 / 1.238 * 1e27
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

# ★★★ 限定された最適化範囲の設定 ★★★
# デフォルト値（固定値として使用）
d_fixed = 0.1578e-3      # 膜厚を固定
eps_bg_fixed = 12.0      # 背景誘電率を固定

# 自動化設定オプション
OPTIMIZATION_MODE = {
    'limited_params': True,     # パラメータを限定するかどうか
    'limited_frequency': True,  # 周波数範囲を限定するかどうか
    'single_model': False,      # 単一モデルのみ最適化するかどうか
    'target_model': 'B_form'    # single_model=Trueの場合の対象モデル
}

class PhysicsModel:
    """
    物理パラメータを保持し、透過スペクトルを計算するクラス
    """
    def __init__(self, d, eps_bg, gamma, a_param):
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
        t = np.divide(numerator, denominator, where=np.abs(denominator)>1e-12, out=np.full_like(denominator, np.inf, dtype=complex))
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

def setup_optimization_parameters():
    """
    最適化設定に基づいてパラメータと範囲を設定
    """
    if OPTIMIZATION_MODE['limited_params']:
        # 限定モード: gamma と a_param のみ最適化
        param_keys = ['gamma', 'a_param']
        p_initial = [0.11e12, 1.0]
        bounds = [(1e11, 5e12), (0.5, 5.0)]  # より狭い範囲
        
        def create_model(p_array):
            p_dict = dict(zip(param_keys, p_array))
            p_dict['d'] = d_fixed
            p_dict['eps_bg'] = eps_bg_fixed
            return PhysicsModel(**p_dict)
            
        print("🎯 限定最適化モード: gamma と a_param のみを最適化します")
        print(f"   固定パラメータ: d={d_fixed:.2e}, eps_bg={eps_bg_fixed}")
        
    else:
        # フルモード: 全パラメータを最適化
        param_keys = ['d', 'eps_bg', 'gamma', 'a_param']
        p_initial = [d_fixed, eps_bg_fixed, 0.11e12, 1.0]
        bounds = [(0.10e-3, 1.e-3), (12.0, 15.0), (1e10, 1e13), (0.01, 10.0)]
        
        def create_model(p_array):
            p_dict = dict(zip(param_keys, p_array))
            return PhysicsModel(**p_dict)
            
        print("🔄 フル最適化モード: 全パラメータを最適化します")
    
    return param_keys, p_initial, bounds, create_model

def setup_frequency_range(exp_freq_thz):
    """
    周波数範囲を設定
    """
    if OPTIMIZATION_MODE['limited_frequency']:
        # 限定周波数範囲（例：0.5-2.0 THz）
        freq_min, freq_max = 0.5, 2.0
        mask = (exp_freq_thz >= freq_min) & (exp_freq_thz <= freq_max)
        
        omega_hz = np.linspace(freq_min*1e12, freq_max*1e12, 300)
        omega_rad_s = omega_hz * 2 * np.pi
        freq_thz = omega_hz / 1e12
        
        print(f"🎯 限定周波数範囲: {freq_min}-{freq_max} THz")
        return omega_rad_s, freq_thz, mask
    else:
        # フル周波数範囲
        omega_hz = np.linspace(0.1*1e12, np.max(exp_freq_thz)*1e12, 500)
        omega_rad_s = omega_hz * 2 * np.pi
        freq_thz = omega_hz / 1e12
        mask = np.ones(len(exp_freq_thz), dtype=bool)
        
        print("🔄 フル周波数範囲を使用します")
        return omega_rad_s, freq_thz, mask

def setup_model_list():
    """
    最適化対象のモデルリストを設定
    """
    if OPTIMIZATION_MODE['single_model']:
        model_list = [OPTIMIZATION_MODE['target_model']]
        print(f"🎯 単一モデル最適化: {OPTIMIZATION_MODE['target_model']} のみ")
    else:
        model_list = ['H_form', 'B_form']
        print("🔄 両モデル（H形式・B形式）を最適化します")
    
    return model_list

# --- 3. メイン実行ブロック ---
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 GGG透過スペクトル自動フィッティング（限定版）")
    print("=" * 60)
    
    # 設定の表示
    print("\n📋 現在の最適化設定:")
    for key, value in OPTIMIZATION_MODE.items():
        print(f"   {key}: {value}")
    
    # --- 3.1 実験データの読み込み ---
    print("\n📂 実験データを読み込みます...")
    file_path = "Circular_Polarization_B_Field.xlsx"; sheet_name = 'Sheet2'
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    exp_freq_thz = df['Frequency (THz)'].to_numpy(dtype=float)
    exp_transmittance_7_7 = df['Transmittance (7.7T)'].to_numpy(dtype=float)
    
    # データの正規化
    min_exp, max_exp = np.min(exp_transmittance_7_7), np.max(exp_transmittance_7_7)
    exp_transmittance_7_7_normalized = (exp_transmittance_7_7 - min_exp) / (max_exp - min_exp)
    print(f"✅ データの読み込み完了: {len(exp_freq_thz)} データポイント")

    # --- 3.2 最適化パラメータと範囲の設定 ---
    param_keys, p_initial, bounds, create_model = setup_optimization_parameters()
    
    # --- 3.3 周波数範囲の設定 ---
    omega_rad_s, freq_thz, freq_mask = setup_frequency_range(exp_freq_thz)
    
    # フィルタリングされた実験データ
    exp_freq_filtered = exp_freq_thz[freq_mask]
    exp_data_filtered = exp_transmittance_7_7_normalized[freq_mask]
    
    T_fixed = 35.0

    # --- 3.4 最適化対象モデルの設定 ---
    model_list = setup_model_list()

    # --- 4. 自動最適化 ---
    def cost_function(p_array, model_type):
        """最適化用のコスト関数"""
        model = create_model(p_array)
        delta_T_fit = model.get_spectrum(omega_rad_s, T_fixed, 7.7, model_type)
        
        min_th, max_th = np.min(delta_T_fit), np.max(delta_T_fit)
        delta_T_normalized = (delta_T_fit - min_th) / (max_th - min_th) if (max_th - min_th) > 1e-9 else np.zeros_like(delta_T_fit)
        
        # 実験データと対応する理論値を補間
        theory_interp = np.interp(exp_freq_filtered, freq_thz, delta_T_normalized)
        
        # 最小二乗誤差を計算
        residuals = exp_data_filtered - theory_interp
        return np.sum(residuals**2)

    print(f"\n⚙️ 自動最適化を開始します...")
    print(f"   最適化パラメータ: {param_keys}")
    print(f"   対象モデル: {model_list}")
    
    results_dict = {}
    for model_name in model_list:
        print(f"\n--- [{model_name}]モデルの最適化中 ---")
        start_time = time.time()
        
        result = minimize(cost_function, p_initial, args=(model_name,), method='L-BFGS-B', bounds=bounds)
        
        end_time = time.time()
        results_dict[model_name] = result
        
        if result.success:
            print(f"✅ 最適化成功！ 実行時間: {end_time - start_time:.2f}秒")
            print(f"   最小二乗誤差: {result.fun:.6f}")
        else:
            print(f"❌ 最適化失敗: {result.message}")

    # --- 5. 結果の表示と最終プロット ---
    print("\n" + "=" * 60)
    print("📊 最適化結果まとめ")
    print("=" * 60)
    
    fig_final, ax_final = plt.subplots(figsize=(12, 8))
    
    # 実験データをプロット
    ax_final.plot(exp_freq_filtered, exp_data_filtered, 'o', color='black', markersize=6, label='実験データ（フィルタリング済み）')
    
    # 最適化結果をプロット
    colors = {'H_form': 'blue', 'B_form': 'darkorange'}
    
    for model_name, result in results_dict.items():
        if result.success:
            print(f"\n🎉 [{model_name}] モデル最適化成功！")
            print(f"   最小二乗誤差: {result.fun:.6f}")
            
            for key, val in zip(param_keys, result.x):
                print(f"   {key:<10} = {val:.4e}")
            
            # 最適化されたモデルで理論値を計算
            final_model = create_model(result.x)
            final_spectrum = final_model.get_spectrum(omega_rad_s, T_fixed, 7.7, model_name)
            min_opt, max_opt = np.min(final_spectrum), np.max(final_spectrum)
            final_spectrum_normalized = (final_spectrum - min_opt) / (max_opt - min_opt)
            
            ax_final.plot(freq_thz, final_spectrum_normalized, color=colors[model_name], 
                         linewidth=2.5, label=f'最適化後理論値 ({model_name})')
        else:
            print(f"\n⚠️ [{model_name}] モデル最適化失敗: {result.message}")
    
    # グラフの体裁
    ax_final.set_title('限定領域における自動最適化結果', fontsize=16)
    ax_final.set_xlabel('周波数 (THz)', fontsize=12)
    ax_final.set_ylabel('正規化透過率 $T(B)$', fontsize=12)
    ax_final.legend(fontsize=11)
    ax_final.grid(True, linestyle='--', alpha=0.7)
    
    # 最適化範囲を強調表示
    if OPTIMIZATION_MODE['limited_frequency']:
        ax_final.axvspan(0.5, 2.0, alpha=0.1, color='red', label='最適化対象範囲')
        ax_final.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('limited_fitting_result.png', dpi=300)
    plt.show()
    
    print(f"\n🎯 限定領域での最適化が完了しました。")
    print(f"   設定: {OPTIMIZATION_MODE}")
