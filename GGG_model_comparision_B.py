import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.graph import Op, Apply
import japanize_matplotlib # 日本語表示のため

# --- 0. プロット設定 ---
plt.rcParams['figure.dpi'] = 100

# --- 1. 物理定数とパラメータ定義 ---
kB = 1.380649e-23; muB = 9.274010e-24; hbar = 1.054571e-34; c = 299792458
eps_bg = 14.44; s = 3.5; g_factor = 1.95
d = 0.150466e-3 
B4_param = 0.8 / 240 * 0.606; B6_param = 0.04 / 5040 * -1.513; B4 = B4_param; B6 = B6_param
mu0 = 4.0 * np.pi * 1e-7; N_spin_exp = 24/1.238 * 1e27; N_spin = N_spin_exp * 10
G0 = mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)

# --- 2. 汎用化された物理モデル関数 ---
# (get_hamiltonian, calculate_susceptibility, calculate_transmission_intensityは変更なし)
def get_hamiltonian(B_ext_z):
    m_values = np.arange(s, -s - 1, -1); Sz = np.diag(m_values)
    O04 = 60*np.diag([7,-13,-3,9,9,-3,-13,7]); X_O44 = np.zeros((8,8)); X_O44[3,7],X_O44[4,0]=np.sqrt(35),np.sqrt(35); X_O44[2,6],X_O44[5,1]=5*np.sqrt(3),5*np.sqrt(3); O44=12*(X_O44+X_O44.T)
    O06 = 1260*np.diag([1,-5,9,-5,-5,9,-5,1]); X_O46 = np.zeros((8,8)); X_O46[3,7],X_O46[4,0]=3*np.sqrt(35),3*np.sqrt(35); X_O46[2,6],X_O46[5,1]=-7*np.sqrt(3),-7*np.sqrt(3); O46=60*(X_O46+X_O46.T)
    H_cf = (B4 * kB) * (O04 + 5 * O44) + (B6 * kB) * (O06 - 21 * O46)
    H_zee = g_factor * muB * B_ext_z * Sz
    return H_cf + H_zee

def calculate_susceptibility(omega_array, H, T, gamma):
    eigenvalues, _ = np.linalg.eigh(H); eigenvalues -= np.min(eigenvalues)
    Z = np.sum(np.exp(-eigenvalues / (kB * T))); populations = np.exp(-eigenvalues / (kB * T)) / Z
    delta_E = eigenvalues[1:] - eigenvalues[:-1]; delta_pop = populations[1:] - populations[:-1]
    omega_0 = delta_E / hbar
    m_vals = np.arange(s, -s, -1); transition_strength = (s + m_vals) * (s - m_vals + 1)
    numerator = G0 * delta_pop * transition_strength
    denominator = (omega_0[:, np.newaxis] - omega_array) - (1j * gamma)
    chi_array = np.sum(numerator[:, np.newaxis] / denominator, axis=0)
    return -chi_array # 吸収を負で表現

def calculate_transmission_intensity(omega_array, mu_r_array):
    n_complex = np.sqrt(eps_bg * mu_r_array + 0j); impe = np.sqrt(mu_r_array / eps_bg + 0j)
    lambda_0 = np.full_like(omega_array, np.inf, dtype=float)
    nonzero_mask = omega_array != 0; lambda_0[nonzero_mask] = (2 * np.pi * c) / omega_array[nonzero_mask]
    delta = 2 * np.pi * n_complex * d / lambda_0
    r = (impe - 1) / (impe + 1)
    t_amp_denominator = 1 - r**2 * np.exp(2j * delta)
    t = np.divide((1-r**2) * np.exp(1j * delta), t_amp_denominator, where=np.abs(t_amp_denominator)>1e-9, out=np.zeros_like(t_amp_denominator, dtype=complex))
    return np.abs(t)**2

# --- 3. PyMCと連携するためのOpクラス (ハミルトニアンの計算を__init__に移動) ---
class PhysicsModelOp(Op):
    itypes = [pt.dscalar, pt.dscalar] # a, gamma
    otypes = [pt.dvector]

    def __init__(self, omega_array, T_val, B_val, model_type):
        self.omega_array = omega_array; self.T = T_val; self.B = B_val
        self.model_type = model_type
        
        # ★★★【改善点】ハミルトニアンをここで一度だけ計算 ★★★
        self.H_B = get_hamiltonian(self.B)

    def perform(self, node, inputs, output_storage):
        a, gamma = inputs
        
        # 事前計算済みのハミルトニアンを使って、効率的に感受率を計算
        chi_B_raw = calculate_susceptibility(self.omega_array, self.H_B, self.T, gamma)
        chi_B = a * chi_B_raw
        
        if self.model_type == 'H_form':
            mu_r_B = 1 + chi_B
        elif self.model_type == 'B_form':
            mu_r_B = np.divide(1, 1 - chi_B, where=(1 - chi_B)!=0, out=np.ones_like(chi_B, dtype=complex))
        
        T_B = calculate_transmission_intensity(self.omega_array, mu_r_B)
        output_storage[0][0] = T_B

# --- 4. メイン実行ブロック ---
if __name__ == '__main__':
    # --- データの読み込み ---
    print("実験データを読み込みます...")
    file_path = "Circular_Polarization_B_Field.xlsx"; sheet_name = 'Sheet2'
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
        exp_freq_thz = df['Frequency (THz)'].to_numpy(dtype=float)
        # 磁場ごとの実験データを辞書で管理
        exp_data = {
            7.7: df['Transmittance (7.7T)'].to_numpy(dtype=float)
            # 5.0: df['Transmittance (5T)'].to_numpy(dtype=float), # 必要に応じてコメントを外す
            # 9.0: df['Transmittance (9T)'].to_numpy(dtype=float),
        }
        print(f"データの読み込みに成功しました。")
    except Exception as e: print(f"データ読み込み中にエラー: {e}"); exit()

    # --- シミュレーションと推定の実行 ---
    model_types = ['H_form', 'B_form']
    B_scan_values = [7.7] # ★★★ ここで解析したい磁場を指定 ★★★
    
    traces = {}
    ppcs = {}
    
    # 磁場ごと、モデルごとにサンプリングを実行
    for b_val in B_scan_values:
        print(f"\n========== 磁場 B = {b_val} T の解析を開始 ==========")
        # 対応する実験データを取得
        current_exp_data = exp_data[b_val]
        current_exp_omega = exp_freq_thz[~np.isnan(current_exp_data)] * 1e12 * 2 * np.pi
        current_exp_transmittance = current_exp_data[~np.isnan(current_exp_data)]

        for mt in model_types:
            print(f"\n--- [{mt}] モデルのサンプリングを開始します ---")
            
            # ★★★【改善点】各ループで、対応するBの値を使ってOpをインスタンス化 ★★★
            physics_model = PhysicsModelOp(current_exp_omega, T_val=35.0, B_val=b_val, model_type=mt)
            
            with pm.Model() as model:
                a = pm.HalfNormal('a', sigma=0.1)
                gamma_param = pm.HalfNormal('gamma', sigma=50e9)
                sigma_obs = pm.HalfCauchy('sigma', beta=0.1)
                mu = physics_model(a, gamma_param)
                Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma_obs, observed=current_exp_transmittance)
                
                # サンプリング結果を、(磁場, モデル名)のタプルをキーとして保存
                traces[(b_val, mt)] = pm.sample(2000, tune=2000, chains=4, cores=4, random_seed=42, idata_kwargs={"log_likelihood": True})
                ppcs[(b_val, mt)] = pm.sample_posterior_predictive(traces[(b_val, mt)], random_seed=42)

            print(f"--- [{mt}] モデルのサンプリング完了 ---")
            print(az.summary(traces[(b_val, mt)], var_names=['a', 'gamma','sigma']))

    # --- モデル比較と結果の可視化 ---
    # (ここでは7.7Tの結果のみをプロットする例)
    target_B = 7.7
    traces_to_compare = {mt: traces[(target_B, mt)] for mt in model_types}
    
    print(f"\n--- B = {target_B} T でのモデル比較 (LOO-CV) ---")
    compare_df = az.compare(traces_to_compare)
    print(compare_df)
    
    # 最終的な比較グラフ
    print("\nフィッティング結果をプロットします...")
    fig, ax = plt.subplots(figsize=(10, 6))
    target_exp_data = exp_data[target_B]
    ax.plot(exp_freq_thz, target_exp_data, 'o', color='black', markersize=4, label='実験データ')
    
    colors = {'H_form': 'blue', 'B_form': 'red'}
    for mt in model_types:
        trace = traces[(target_B, mt)]
        # ... (ベストフィット曲線の描画ロジックは以前のままでOK) ...
        a_mean = trace.posterior['a'].mean().item(); gamma_mean = trace.posterior['gamma'].mean().item()
        H_best = get_hamiltonian(B_ext_z=target_B); chi_best_raw = calculate_susceptibility(exp_omega_rad_s, H_best, T=35.0, gamma=gamma_mean)
        chi_best = a_mean * chi_best_raw
        if mt == 'H_form': mu_r_best = 1 + chi_best
        else: mu_r_best = 1 / (1-chi_best)
        best_fit_prediction = calculate_transmission_intensity(exp_omega_rad_s, mu_r_best)
        ax.plot(exp_freq_thz, best_fit_prediction, color=colors[mt], lw=2, label=f'ベストフィット ({mt})')

    ax.set_xlabel('周波数 (THz)'); ax.set_ylabel('透過率 $T(B)$'); ax.legend(); ax.grid(True)
    plt.savefig(f'final_fitting_comparison_B{target_B}.png', dpi=300)
    plt.show()

    print("\n全ての処理が完了しました。")