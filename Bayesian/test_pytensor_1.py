import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pytensor.tensor as pt # NumPyの代わりにPyTensorをインポート
import os
import psutil

try:
    import japanize_matplotlib # 日本語表示のため
except ImportError:
    print("注意: japanize_matplotlib がインストールされていません。")

# --- 0. プロット設定 ---
plt.rcParams['figure.dpi'] = 120

# --- 1. 物理定数とパラメータ初期値 ---
kB = 1.380649e-23; muB = 9.274010e-24; hbar = 1.054571e-34; c = 299792458; mu0 = 4.0 * np.pi * 1e-7
s = 3.5
N_spin = 24/1.238 * 1e27 #GGGのスピン数密度
d_init = 157.8e-6
eps_bg_init = 13.14
g_factor_init = 1.95
B4_init = 0.8 / 240 * 0.606
B6_init = 0.04 / 5040 * -1.513
gamma_init = 0.11e12

# --- 2. PyTensorネイティブな物理モデル関数 (pm.Model内で使用) ---
def get_hamiltonian_pt(B_ext_z, B4, B6, g_factor):
    """PyTensorのテンソルでハミルトニアンを計算する関数"""
    m_values = pt.arange(s, -s - 1, -1)
    Sz = pt.diag(m_values)
    # 結晶場演算子は定数なのでNumPyで定義してOK
    O04 = 60*np.diag([7,-13,-3,9,9,-3,-13,7])
    X_O44 = np.zeros((8,8)); X_O44[3,7],X_O44[4,0]=np.sqrt(35),np.sqrt(35); X_O44[2,6],X_O44[5,1]=5*np.sqrt(3),5*np.sqrt(3); O44=12*(X_O44+X_O44.T)
    O06 = 1260*np.diag([1,-5,9,-5,-5,9,-5,1])
    X_O46 = np.zeros((8,8)); X_O46[3,7],X_O46[4,0]=3*np.sqrt(35),3*np.sqrt(35); X_O46[2,6],X_O46[5,1]=-7*np.sqrt(3),-7*np.sqrt(3); O46=60*(X_O46+X_O46.T)
    H_cf = (B4 * kB) * (O04 + 5 * O44) + (B6 * kB) * (O06 - 21 * O46)
    H_zee = g_factor * muB * B_ext_z * Sz
    return H_cf + H_zee

def calculate_transmission_pt(params, omega_array, T_val, B_val, model_type):
    """PyTensor関数のみを使用して透過スペクトルを計算する統合関数"""
    a, log_gamma, eps_bg, d, B4, B6, g_factor = params
    gamma = pt.exp(log_gamma)
    H = get_hamiltonian_pt(B_val, B4, B6, g_factor)
    eigenvalues, _ = pt.linalg.eigh(H)
    eigenvalues -= eigenvalues.min()
    Z = pt.sum(pt.exp(-eigenvalues / (kB * T_val)))
    populations = pt.exp(-eigenvalues / (kB * T_val)) / Z
    delta_E = eigenvalues[1:] - eigenvalues[:-1]
    delta_pop = populations[1:] - populations[:-1]
    omega_0 = delta_E / hbar
    m_vals_trans = pt.arange(s, -s, -1)
    transition_strength = (s + m_vals_trans) * (s - m_vals_trans + 1)
    G0 = mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
    numerator = a * G0 * delta_pop * transition_strength
    denominator = (omega_0.dimshuffle(0, 'x') - omega_array) - (1j * gamma.dimshuffle(0, 'x'))
    gamma_term = gamma.dimshuffle(0, 'x')

    chi_B_raw = pt.sum(numerator.dimshuffle(0, 'x') / denominator, axis=0)
    chi_B = -chi_B_raw
    if model_type == 'H_form':
        mu_r_B = 1 + chi_B
    else: # B_form
        mu_r_B = 1 / (1 - chi_B)
    n_complex = pt.sqrt(eps_bg * mu_r_B)
    impe = pt.sqrt(mu_r_B / eps_bg)
    lambda_0 = (2 * np.pi * c) / omega_array
    delta = (2 * np.pi * n_complex * d) / lambda_0
    exp_j_delta = pt.exp(1j * delta)
    num_t = 4 * impe * exp_j_delta
    den_t = (1 + impe)**2 - (1 - impe)**2 * pt.exp(2j * delta)
    t = num_t / den_t
    
    # === 🚨 TypeError 修正箇所 ===
    # pt.abs(t)**2 の代わりに、実部と虚部の2乗和を計算する
    transmission = pt.real(t)**2 + pt.imag(t)**2
    
    min_trans = pt.min(transmission)
    max_trans = pt.max(transmission)
    normalized_transmission = (transmission - min_trans) / (max_trans - min_trans)
    return normalized_transmission

# --- 3. NumPyベースの物理モデル関数 (グラフ描画用) ---
def get_hamiltonian_np(B_ext_z, B4, B6, g_factor):
    m_values = np.arange(s, -s - 1, -1)
    Sz = np.diag(m_values)
    O04 = 60*np.diag([7,-13,-3,9,9,-3,-13,7])
    X_O44 = np.zeros((8,8)); X_O44[3,7],X_O44[4,0]=np.sqrt(35),np.sqrt(35); X_O44[2,6],X_O44[5,1]=5*np.sqrt(3),5*np.sqrt(3); O44=12*(X_O44+X_O44.T)
    O06 = 1260*np.diag([1,-5,9,-5,-5,9,-5,1])
    X_O46 = np.zeros((8,8)); X_O46[3,7],X_O46[4,0]=3*np.sqrt(35),3*np.sqrt(35); X_O46[2,6],X_O46[5,1]=-7*np.sqrt(3),-7*np.sqrt(3); O46=60*(X_O46+X_O46.T)
    H_cf = (B4 * kB) * (O04 + 5 * O44) + (B6 * kB) * (O06 - 21 * O46)
    H_zee = g_factor * muB * B_ext_z * Sz
    return H_cf + H_zee

def calculate_transmission_np(params, omega_array, T_val, B_val, model_type):
    a, gamma, eps_bg, d, B4, B6, g_factor = params
    H = get_hamiltonian_np(B_val, B4, B6, g_factor)
    eigenvalues, _ = np.linalg.eigh(H)
    eigenvalues -= eigenvalues.min()
    Z = np.sum(np.exp(-eigenvalues / (kB * T_val)))
    populations = np.exp(-eigenvalues / (kB * T_val)) / Z
    delta_E = eigenvalues[1:] - eigenvalues[:-1]
    delta_pop = populations[1:] - populations[:-1]
    omega_0 = delta_E / hbar
    m_vals_trans = np.arange(s, -s, -1)
    transition_strength = (s + m_vals_trans) * (s - m_vals_trans + 1)
    G0 = mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
    numerator = a * G0 * delta_pop * transition_strength
    denominator = (omega_0[:, np.newaxis] - omega_array) - (1j * gamma[:, np.newaxis])
    chi_B_raw = np.sum(numerator[:, np.newaxis] / denominator, axis=0)
    chi_B = -chi_B_raw
    if model_type == 'H_form':
        mu_r_B = 1 + chi_B
    else: # B_form
        mu_r_B = 1 / (1 - chi_B)
    n_complex = np.sqrt(eps_bg * mu_r_B)
    impe = np.sqrt(mu_r_B / eps_bg)
    lambda_0 = (2 * np.pi * c) / omega_array
    delta = (2 * np.pi * n_complex * d) / lambda_0
    exp_j_delta = np.exp(1j * delta)
    num_t = 4 * impe * exp_j_delta
    den_t = (1 + impe)**2 - (1 - impe)**2 * np.exp(2j * delta)
    t = num_t / den_t
    transmission = np.abs(t)**2
    min_trans = np.min(transmission)
    max_trans = np.max(transmission)
    normalized_transmission = (transmission - min_trans) / (max_trans - min_trans)
    return normalized_transmission


# --- 4. メイン実行ブロック ---
if __name__ == '__main__':
    # --- データの読み込み ---
    print("実験データを読み込みます...")
    file_path = "C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\Circular_Polarization_B_Field.xlsx"
    sheet_name = 'Sheet2'
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    exp_freq_thz_full = df['Frequency (THz)'].to_numpy(dtype=float)
    exp_transmittance_full = df['Transmittance (7.7T)'].to_numpy(dtype=float)
    df_filtered = df[df['Frequency (THz)'] <= 0.376].copy()
    exp_freq_thz_fit = df_filtered['Frequency (THz)'].to_numpy(dtype=float)
    exp_transmittance_fit = df_filtered['Transmittance (7.7T)'].to_numpy(dtype=float)
    min_exp, max_exp = np.min(exp_transmittance_fit), np.max(exp_transmittance_fit)
    exp_transmittance_norm_fit = (exp_transmittance_fit - min_exp) / (max_exp - min_exp)
    exp_transmittance_norm_full = (exp_transmittance_full - min_exp) / (max_exp - min_exp)
    exp_omega_rad_s_fit = exp_freq_thz_fit * 1e12 * 2 * np.pi

    # --- モデル比較の準備 ---
    model_types = ['H_form', 'B_form']
    traces = {}
    ppcs = {}
    n_transitions = 7

    # --- 各モデルでサンプリングを実行 ---
    for mt in model_types:
        print(f"\n--- [{mt}] PyTensorネイティブモデルのサンプリングを開始します ---")
        with pm.Model() as model:
            # --- 事前分布 ---
            a = pm.Normal('a', mu=1.0, sigma=0.5)
            log_gamma_sigma = pm.HalfNormal('log_gamma_sigma', sigma=1.0)
            log_gamma_array = pm.Normal('log_gamma', mu=np.log(gamma_init), sigma=log_gamma_sigma, shape=n_transitions)
            gamma_array = pm.Deterministic('gamma', pt.exp(log_gamma_array))
            eps_bg = pm.Normal('eps_bg', mu=eps_bg_init, sigma=1.0)
            d = pm.Normal('d', mu=d_init, sigma=10e-6)
            B4 = pm.Normal('B4', mu=B4_init, sigma=abs(B4_init) * 1.0)
            B6 = pm.Normal('B6', mu=B6_init, sigma=abs(B6_init) * 1.0)
            g_factor = pm.Normal('g_factor', mu=g_factor_init, sigma=0.05)
            nu = pm.Gamma('nu', alpha=2, beta=0.1)
            sigma_obs = pm.HalfCauchy('sigma', beta=0.1)
            
            # --- 物理モデルの呼び出し ---
            params_list = [a, log_gamma_array, eps_bg, d, B4, B6, g_factor]
            mu = calculate_transmission_pt(params_list, exp_omega_rad_s_fit, T_val=35.0, B_val=7.7, model_type=mt)
            
            # --- 尤度 ---
            Y_obs = pm.StudentT('Y_obs', nu=nu, mu=mu, sigma=sigma_obs, observed=exp_transmittance_norm_fit)
            
            # --- サンプリング ---
            traces[mt] = pm.sample(2000, tune=2000, target_accept=0.95, chains=4, cores=4, random_seed=42, idata_kwargs={"log_likelihood": True})
            ppcs[mt] = pm.sample_posterior_predictive(traces[mt], random_seed=42)
            
        print(f"--- [{mt}] モデルのサンプリング完了 ---")
        print(az.summary(traces[mt], var_names=['a', 'eps_bg', 'd', 'B4', 'B6', 'g_factor', 'nu', 'sigma']))

    # --- 5. モデル比較とPareto k値の診断 ---
    print("\n--- ベイズ的モデル比較 (LOO-CV) ---")
    idata_dict = {k: v for k, v in traces.items()}
    compare_df = az.compare(idata_dict)
    print(compare_df)
    az.plot_compare(compare_df)
    plt.savefig('model_comparison.png')
    plt.show()

    print("\n--- Pareto k 診断 ---")
    for mt in model_types:
        loo_result = az.loo(traces[mt], pointwise=True)
        k_values = loo_result.pareto_k.values
        k_problem_count = np.sum(k_values > 0.7)
        print(f"[{mt}モデル] Pareto k > 0.7 のデータ点数: {k_problem_count} / {len(k_values)}")
        if k_problem_count > 0:
            print("  -> LOO-CVの信頼性に懸念があります。モデルや尤度の見直しを検討してください。")

    # --- 6. プロット1: フィッティング領域の詳細 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'H_form': 'blue', 'B_form': 'red'}
    ax.plot(exp_freq_thz_fit, exp_transmittance_norm_fit, 'o', color='black', markersize=4, label='実験データ (フィッティング領域)')
    for mt in model_types:
        trace = traces[mt]
        az.plot_hdi(exp_freq_thz_fit, ppcs[mt].posterior_predictive['Y_obs'], ax=ax, color=colors[mt], hdi_prob=0.94, fill_kwargs={'alpha': 0.2})
        # ベストフィット曲線の計算
        params_mean = [
            trace.posterior['a'].mean().item(),
            trace.posterior['gamma'].mean(dim=('chain', 'draw')).values,
            trace.posterior['eps_bg'].mean().item(),
            trace.posterior['d'].mean().item(),
            trace.posterior['B4'].mean().item(),
            trace.posterior['B6'].mean().item(),
            trace.posterior['g_factor'].mean().item()
        ]
        freq_plot = np.linspace(exp_freq_thz_fit.min(), exp_freq_thz_fit.max(), 500)
        omega_plot = freq_plot * 1e12 * 2 * np.pi
        prediction = calculate_transmission_np(params_mean, omega_plot, 35.0, 7.7, mt)
        ax.plot(freq_plot, prediction, color=colors[mt], lw=2, label=f'ベストフィット ({mt})')
    ax.set_xlabel('周波数 (THz)')
    ax.set_ylabel('正規化透過率')
    ax.set_title('ベイズ最適化結果: フィッティング領域')
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig('fitting_plot_fit_region.png')
    plt.show()

    # --- 7. プロット2: 全領域への予測（外挿）性能 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(exp_freq_thz_full, exp_transmittance_norm_full, 'o', color='lightgray', markersize=3, label='実験データ (全領域)')
    ax.plot(exp_freq_thz_fit, exp_transmittance_norm_fit, 'o', color='black', markersize=4, label='実験データ (フィッティング領域)')
    for mt in model_types:
        trace = traces[mt]
        params_mean = [
            trace.posterior['a'].mean().item(),
            trace.posterior['gamma'].mean(dim=('chain', 'draw')).values,
            trace.posterior['eps_bg'].mean().item(),
            trace.posterior['d'].mean().item(),
            trace.posterior['B4'].mean().item(),
            trace.posterior['B6'].mean().item(),
            trace.posterior['g_factor'].mean().item()
        ]
        freq_plot = np.linspace(exp_freq_thz_full.min(), exp_freq_thz_full.max(), 1000)
        omega_plot = freq_plot * 1e12 * 2 * np.pi
        prediction = calculate_transmission_np(params_mean, omega_plot, 35.0, 7.7, mt)
        ax.plot(freq_plot, prediction, color=colors[mt], lw=2, linestyle='--', label=f'全領域予測 ({mt})')
    ax.axvline(x=exp_freq_thz_fit.max(), color='gray', linestyle=':', label='フィッティング領域上限')
    ax.set_xlabel('周波数 (THz)')
    ax.set_ylabel('正規化透過率')
    ax.set_title('ベイズ最適化結果: 全領域への予測性能')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig('fitting_plot_full_range.png')
    plt.show()

    print("\n全ての処理が完了しました。結果を確認してください。")
