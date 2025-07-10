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
# 最終的にフィットが良かった膜厚dに更新
d = 0.150466e-3 
B4_param = 0.8 / 240 * 0.606; B6_param = 0.04 / 5040 * -1.513; B4 = B4_param; B6 = B6_param
mu0 = 4.0 * np.pi * 1e-7; N_spin_exp = 24/1.238 * 1e27; N_spin = N_spin_exp * 10
G0 = mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)

# --- 2. 汎用化された物理モデル関数 ---

def get_hamiltonian(B_ext_z):
    m_values = np.arange(s, -s - 1, -1)
    Sz = np.diag(m_values)
    O04 = 60*np.diag([7,-13,-3,9,9,-3,-13,7]) 
    X_O44 = np.zeros((8,8)); X_O44[3,7],X_O44[4,0]=np.sqrt(35),np.sqrt(35); X_O44[2,6],X_O44[5,1]=5*np.sqrt(3),5*np.sqrt(3); O44=12*(X_O44+X_O44.T)
    O06 = 1260*np.diag([1,-5,9,-5,-5,9,-5,1])
    X_O46 = np.zeros((8,8)); X_O46[3,7],X_O46[4,0]=3*np.sqrt(35),3*np.sqrt(35); X_O46[2,6],X_O46[5,1]=-7*np.sqrt(3),-7*np.sqrt(3); O46=60*(X_O46+X_O46.T)
    H_cf = (B4 * kB) * (O04 + 5 * O44) + (B6 * kB) * (O06 - 21 * O46)
    H_zee = g_factor * muB * B_ext_z * Sz
    return H_cf + H_zee

def calculate_susceptibility(omega_array, H, T, gamma):
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
    denominator = (omega_0[:, np.newaxis] - omega_array) - (1j * gamma)
    chi_array = np.sum(numerator[:, np.newaxis] / denominator, axis=0)
    return -chi_array

def calculate_transmission_intensity(omega_array, mu_r_array):
    """強度|t|^2を計算するヘルパー関数"""
    n_complex = np.sqrt(eps_bg * mu_r_array + 0j)
    impe = np.sqrt(mu_r_array / eps_bg + 0j)
    lambda_0 = np.full_like(omega_array, np.inf, dtype=float)
    nonzero_mask = omega_array != 0
    lambda_0[nonzero_mask] = (2 * np.pi * c) / omega_array[nonzero_mask]
    delta = 2 * np.pi * n_complex * d / lambda_0
    r = (impe - 1) / (impe + 1)
    numerator = 4 * impe * np.exp(1j * delta) / (1 + impe)**2
    denominator = 1 - r**2 * np.exp(2j * delta)
    t = np.divide(numerator, denominator, where=np.abs(denominator)>1e-9, out=np.zeros_like(denominator, dtype=complex))
    return np.abs(t)**2

# --- 3. PyMCと連携するためのOpクラス (絶対透過率T(B)を返すように修正) ---
class PhysicsModelOp(Op):
    itypes = [pt.dscalar, pt.dscalar] # a, gamma
    otypes = [pt.dvector]

    def __init__(self, omega_array, T_val, B_val, model_type):
        self.omega_array = omega_array
        self.T = T_val
        self.B = B_val
        self.model_type = model_type
        self.H_B = get_hamiltonian(self.B)
        # ハミルトニアンはBに依存するため、perform内で都度計算

    def perform(self, node, inputs, output_storage):
        a, gamma = inputs
        
        # ハミルトニアンと感受率を計算
        chi_B_raw = calculate_susceptibility(self.omega_array, self.H_B, self.T, gamma)

        # スケーリング係数aを適用
        chi_B = a * chi_B_raw
        
        # モデルタイプに応じてmu_rを計算
        if self.model_type == 'H_form':
            mu_r_B = 1 + chi_B
        elif self.model_type == 'B_form':
            mu_r_B = np.divide(1, 1 - chi_B, where=(1 - chi_B)!=0, out=np.ones_like(chi_B, dtype=complex))
        else:
            raise ValueError("Unknown model_type")
            
        # 絶対透過率 T(B) を計算
        T_B = calculate_transmission_intensity(self.omega_array, mu_r_B)
        output_storage[0][0] = T_B
        
# --- 4. メイン実行ブロック ---
if __name__ == '__main__':
    # --- データの読み込み ---
    print("実験データを読み込みます...")
    file_path = "Circular_Polarization_B_Field.xlsx"; sheet_name = 'Sheet2'
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
        print(f"データの読み込みに成功しました。読み込み件数: {len(df)}件")
        # 周波数が0.376 THz以下のデータのみをフィルタリング
        df_filtered = df[df['Frequency (THz)'] <= 0.376].copy()
        print(f"フィルタリング後のデータ件数: {len(df_filtered)}件")
        exp_freq_thz = df_filtered['Frequency (THz)'].to_numpy(dtype=float)
        exp_transmittance = df_filtered['Transmittance (7.7T)'].to_numpy(dtype=float)
    except Exception as e: print(f"データ読み込み中にエラー: {e}")
    exit()
    exp_omega_rad_s = exp_freq_thz * 1e12 * 2 * np.pi

    # --- モデル比較の準備 ---
    model_types = ['H_form', 'B_form']
    traces = {}
    ppcs = {}

    # --- 各モデルでサンプリングを実行 ---
    for mt in model_types:
        print(f"\n--- [{mt}] モデルのサンプリングを開始します ---")
        physics_model = PhysicsModelOp(exp_omega_rad_s, T_val=35.0, B_val=7.8, model_type=mt)
        
        with pm.Model() as model:
            # 事前分布
            a = pm.TruncatedNormal('a', mu=0.5, sigma=0.25, lower=0.0, upper=1.0)
            gamma_param = pm.HalfNormal('gamma', sigma=50e9)
            sigma_obs = pm.HalfCauchy('sigma', beta=1.0)
            
            mu = physics_model(a, gamma_param)

            # 尤度 (絶対透過率を直接比較)
            Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma_obs, observed=exp_transmittance)
            
            traces[mt] = pm.sample(2000, tune=2000, chains=4, cores=4, random_seed=42, idata_kwargs={"log_likelihood": True})
            ppcs[mt] = pm.sample_posterior_predictive(traces[mt], random_seed=42)
        print(f"--- [{mt}] モデルのサンプリング完了 ---")
        print(az.summary(traces[mt], var_names=['a', 'gamma','sigma']))

    # --- 5. モデル比較の結果表示 ---
    print("\n--- ベイズ的モデル比較 (LOO-CV) ---")
    # ArviZのcompare関数でモデルの予測性能を比較
    # traces辞書をInferenceDataの辞書に変換
    # idata_dict = {k: v if isinstance(v, az.InferenceData) else az.from_pymc3(trace=v) for k, v in traces.items()}
    compare_df = az.compare(idata_dict)
    print(compare_df)
    axes = az.plot_compare(compare_df, figsize=(8, 4))
    # Figureオブジェクトを取得
    fig = axes.ravel()[0].figure if hasattr(axes, "ravel") else axes.figure
    fig.suptitle('モデル比較', fontsize=16)
    fig.tight_layout()
    plt.savefig('final_model_comparison.png', dpi=300)
    plt.close()

    # --- 6. ベストフィット曲線のプロット ---    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(exp_freq_thz, exp_transmittance, 'o', color='black', markersize=4, label='実験データ')
    colors = {'H_form': 'blue', 'B_form': 'red'}
    freq_thz_plot = np.linspace(np.min(exp_freq_thz), np.max(exp_freq_thz), 500)
    omega_rad_s_plot = freq_thz_plot * 1e12 * 2 * np.pi

    for mt in model_types:
        trace = traces[mt]; ppc = ppcs[mt]
        az.plot_hdi(exp_freq_thz, ppc.posterior_predictive['Y_obs'], ax=ax, color=colors[mt], hdi_prob=0.94, fill_kwargs={'alpha': 0.2})
        # ベストフィット曲線の再計算ロジック（事後分布の平均値を採用）
        a_mean = trace.posterior['a'].mean().item()
        gamma_mean = trace.posterior['gamma'].mean().item()
        # PhysicsModelOpを再利用せず、グローバル関数で計算するのがより安全
        H_best = get_hamiltonian(B_ext_z=7.7)
        chi_best_raw = calculate_susceptibility(omega_rad_s_plot, H_best, T=35.0, gamma=gamma_mean)
        chi_best = a_mean * chi_best_raw
        if mt == 'H_form':
            mu_r_best = 1 + chi_best
        else: 
            mu_r_best = 1 / (1-chi_best)
        best_fit_prediction = calculate_transmission_intensity(omega_rad_s_plot, mu_r_best)

        ax.plot(freq_thz_plot, best_fit_prediction, color=colors[mt], lw=2, label=f'ベストフィット ({mt})')

    ax.set_xlabel('周波数 (THz)')
    ax.set_ylabel('透過率 $T(B)$')
    ax.legend()
    ax.grid(True)
    plt.savefig('final_fitting_comparison.png', dpi=300)
    plt.close()
    print("全ての処理が完了しました。結果を確認してください。")
