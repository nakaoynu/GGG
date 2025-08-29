# bayesian_with_background_mu.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
import os
import pathlib
import re
from typing import List, Dict, Any

# --- 0. 環境設定 ---
print("--- 0. 環境設定を開始します ---")
os.environ['PYTENSOR_FLAGS'] = 'optimizer=fast_compile,floatX=float64'
try:
    import japanize_matplotlib
except ImportError:
    print("警告: japanize_matplotlib が見つかりません。")
plt.rcParams['figure.dpi'] = 120
IMAGE_DIR = pathlib.Path(__file__).parent / "pymc_background_mu_results"
IMAGE_DIR.mkdir(exist_ok=True)
print(f"画像は '{IMAGE_DIR.resolve()}' に保存されます。")

# --- 1. 物理定数とシミュレーション条件 ---
print("--- 1. 物理定数と初期値を設定します ---")
kB = 1.380649e-23; muB = 9.274010e-24; hbar = 1.054571e-34
c = 299792458; mu0 = 4.0 * np.pi * 1e-7
s = 3.5; N_spin = 24 / 1.238 * 1e27

# 固定パラメータ
d = 157.8e-6 * 0.99
eps_bg = 13.1404
TEMPERATURE = 1.5 # 温度 (K)

# パラメータの初期値（事前分布の中心として使用）
B4_init = 0.8 / 240 * 0.606
B6_init = 0.04 / 5040 * -1.513
gamma_init = 0.11e12
a_scale_init = 1.5
g_factor_init = 2.02
# 背景透磁率モデルのパラメータ初期値
mu0_a_init = 1.0; mu0_b_init = 0.0; mu0_c_init = 0.0

# --- 2. 物理モデル関数 ---

def get_hamiltonian(B_ext_z: float, g_factor: float, B4: float, B6: float) -> np.ndarray:
    """与えられたパラメータからハミルトニアンを計算する。"""
    m_values = np.arange(s, -s - 1, -1)
    Sz = np.diag(m_values)
    O40 = 60 * np.diag([7, -13, -3, 9, 9, -3, -13, 7])
    X_O44 = np.zeros((8, 8)); X_O44[3, 7], X_O44[4, 0] = np.sqrt(35), np.sqrt(35); X_O44[2, 6], X_O44[5, 1] = 5 * np.sqrt(3), 5 * np.sqrt(3)
    O44 = 12 * (X_O44 + X_O44.T)
    O60 = 1260 * np.diag([1, -5, 9, -5, -5, 9, -5, 1])
    X_O64 = np.zeros((8, 8)); X_O64[3, 7], X_O64[4, 0] = 3 * np.sqrt(35), 3 * np.sqrt(35); X_O64[2, 6], X_O64[5, 1] = -7 * np.sqrt(3), -7 * np.sqrt(3)
    O64 = 60 * (X_O64 + X_O64.T)
    H_cf = (B4 * kB) * (O40 + 5 * O44) + (B6 * kB) * (O60 - 21 * O64)
    H_zee = g_factor * muB * B_ext_z * Sz
    return H_cf + H_zee

def calculate_susceptibility(omega_array: np.ndarray, H: np.ndarray, T: float, gamma_array: np.ndarray) -> np.ndarray:
    """ハミルトニアンから磁化率χを計算する。"""
    eigenvalues, _ = np.linalg.eigh(H)
    eigenvalues -= np.min(eigenvalues)
    Z = np.sum(np.exp(-eigenvalues / (kB * T)))
    populations = np.exp(-eigenvalues / (kB * T)) / Z
    delta_E = eigenvalues[1:] - eigenvalues[:-1]
    delta_pop = populations[1:] - populations[:-1]
    omega_0 = delta_E / hbar
    m_vals = np.arange(s, -s, -1)
    transition_strength = (s + m_vals) * (s - m_vals + 1)
    # gamma_arrayがdelta_Eと同じ次元を持つように調整
    if len(gamma_array) != len(delta_E):
        if len(gamma_array) > len(delta_E):
            gamma_array = gamma_array[:len(delta_E)]
        else:
            gamma_array = np.pad(gamma_array, (0, len(delta_E) - len(gamma_array)), 'edge')

    numerator = delta_pop * transition_strength
    denominator = (omega_0[:, np.newaxis] - omega_array) - (1j * gamma_array[:, np.newaxis])
    
    # ゼロ除算チェックを追加
    safe_denominator = np.where(np.abs(denominator) < 1e-15, 1e-15, denominator)
    chi_array = np.sum(numerator[:, np.newaxis] / safe_denominator, axis=0)
    return -chi_array

def calculate_normalized_transmission(omega_array: np.ndarray, mu_r_array: np.ndarray) -> np.ndarray:
    """正規化された透過強度を計算する。"""
    n_complex = np.sqrt(eps_bg * mu_r_array + 0j)
    impe = np.sqrt(mu_r_array / eps_bg + 0j)
    lambda_0 = np.full_like(omega_array, np.inf, dtype=float)
    nonzero_mask = omega_array > 1e-12
    lambda_0[nonzero_mask] = (2 * np.pi * c) / omega_array[nonzero_mask]
    delta = 2 * np.pi * n_complex * d / lambda_0
    numerator = 4 * impe
    denominator = (1 + impe)**2 * np.exp(-1j * delta) - (1 - impe)**2 * np.exp(1j * delta)
    safe_mask = np.abs(denominator) > 1e-12
    t = np.zeros_like(denominator, dtype=complex)
    t[safe_mask] = numerator[safe_mask] / denominator[safe_mask]
    transmission = np.abs(t)**2
    min_trans, max_trans = np.min(transmission), np.max(transmission)
    return (transmission - min_trans) / (max_trans - min_trans) if max_trans > min_trans else np.full_like(transmission, 0.5)

# --- 3. PyMCカスタムOpクラス (背景透磁率モデル対応) ---

class BackgroundMuPhysicsModelOp(Op):
    """
    背景透磁率モデルを組み込んだ物理シミュレーションを実行するOp。
    """
    def __init__(self, datasets: List[Dict[str, Any]], model_type: str):
        self.datasets = datasets
        self.model_type = model_type
        # 入力: a_scale, gamma, g_factor, B4, B6, mu0_a, mu0_b, mu0_c
        self.itypes = [pt.dscalar, pt.dvector, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar]
        self.otypes = [pt.dvector]

    def perform(self, node, inputs, output_storage):
        a_scale, gamma, g_factor, B4, B6, mu0_a, mu0_b, mu0_c = inputs
        
        full_predicted_y = []
        
        for data in self.datasets:
            T = data['temperature']
            omega_array = data['omega']
            b_val = data['b_field']
            
            # 1. 背景透磁率 mu_r_0 を計算
            mu_r_0 = mu0_a + mu0_b * (b_val - 9.0) / 9.0 + mu0_c * (T - 1.5) / 1.5
            
            # 2. 磁気共鳴項 mu_r_B を計算
            G0 = a_scale * mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
            H = get_hamiltonian(b_val, g_factor, B4, B6)
            chi_raw = calculate_susceptibility(omega_array, H, T, gamma)
            chi = G0 * chi_raw
            
            # モデルタイプに応じてmu_rを計算
            if self.model_type == 'H_form':
                # H = μ₀(H + M) の関係から μᵣ = 1 + χ
                mu_r_B = 1 + chi
            elif self.model_type == 'B_form':
                # B = μ₀μᵣH の関係から μᵣ = 1/(1-χ)
                epsilon = 1e-12  # 数値安定性のため
                denominator = 1 - chi
                safe_mask = np.abs(denominator) > epsilon
                mu_r_B = np.ones_like(chi, dtype=complex)
                mu_r_B[safe_mask] = 1.0 / denominator[safe_mask]
                mu_r_B[~safe_mask] = 1e6  # 発散を避ける
            else:
                raise ValueError("Unknown model_type")            
            # 3. 全体の透磁率を計算
            mu_r_total = mu_r_0 * mu_r_B
            
            # 4. 透過スペクトルを計算
            predicted_y = calculate_normalized_transmission(omega_array, mu_r_total)
            full_predicted_y.append(predicted_y)
            
        output_storage[0][0] = np.concatenate(full_predicted_y)

# --- 4. データ読み込み関数 (全領域) ---

def load_data_full_range(file_path: str, sheet_name: str) -> List[Dict[str, Any]]:
    """Excelから全領域のデータを読み込む。"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    except Exception as e:
        raise FileNotFoundError(f"Excelファイル '{file_path}' が読み込めません: {e}")

    freq_col = 'Frequency (THz)'
    trans_cols = [col for col in df.columns if 'Transmittance' in col and 'T)' in col]
    
    datasets = []
    for col in trans_cols:
        b_match = re.search(r'\((\d+\.?\d*)T\)', col)
        if not b_match: continue
        b_value = float(b_match.group(1))
        
        df_clean = df[[freq_col, col]].dropna()
        freq = df_clean[freq_col].to_numpy()
        trans = df_clean[col].to_numpy()
        min_exp, max_exp = trans.min(), trans.max()
        trans_normalized = (trans - min_exp) / (max_exp - min_exp) if max_exp > min_exp else np.full_like(trans, 0.5)

        datasets.append({
            'temperature': TEMPERATURE,
            'b_field': b_value,
            'frequency': freq,
            'transmittance': trans_normalized,
            'omega': freq * 1e12 * 2 * np.pi
        })
        print(f"磁場 {b_value} T (温度 {TEMPERATURE} K): {len(freq)}点のデータを処理。")
        
    return datasets

# --- 5. 結果可視化関数 ---

def plot_inference_results(datasets: List[Dict[str, Any]], trace: az.InferenceData, model_type: str, n_samples: int = 100):
    """ベイズ推定の結果（平均予測、95%信用区間）をプロットする。"""
    colors = {'H_form': 'blue', 'B_form': 'red'}
    num_conditions = len(datasets)
    fig, axes = plt.subplots(1, num_conditions, figsize=(7 * num_conditions, 6), sharey=True)
    if num_conditions == 1: axes = [axes]
    
    for i, data in enumerate(datasets):
        ax = axes[i]
        T = data['temperature']
        b_val = data['b_field']
        
        # 事後分布からランダムサンプリング
        posterior_group = trace['posterior']
        n_chains = posterior_group.sizes["chain"]
        n_draws = posterior_group.sizes["draw"]
        total_samples = n_chains * n_draws
        indices = np.random.choice(total_samples, min(n_samples, total_samples), replace=False)
        
        freq_plot = np.linspace(data['frequency'].min(), data['frequency'].max(), 300)
        omega_plot = freq_plot * 1e12 * 2 * np.pi
        
        predictions = []
        for idx in indices:
            chain = idx // n_draws
            draw = idx % n_draws
            params = {var: posterior_group[var].values[chain, draw] for var in posterior_group.data_vars}
            
            mu_r_0 = params['mu0_a'].item() + params['mu0_b'].item() * (b_val - 9.0) / 9.0 + params['mu0_c'].item() * (T - 1.5) / 1.5
            G0 = params['a_scale'].item() * mu0 * N_spin * (params['g_factor'].item() * muB)**2 / (2 * hbar)
            H = get_hamiltonian(b_val, params['g_factor'].item(), params['B4'].item(), params['B6'].item())
            chi_raw = calculate_susceptibility(omega_plot, H, T, params['gamma'])
            chi = G0 * chi_raw
            mu_r_B = 1 + chi if model_type == 'H_form' else 1 / (1 - chi)
            mu_r_total = mu_r_0 * mu_r_B
            pred_y = calculate_normalized_transmission(omega_plot, mu_r_total)
            predictions.append(pred_y)
            
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        ci_lower = np.percentile(predictions, 2.5, axis=0)
        ci_upper = np.percentile(predictions, 97.5, axis=0)

        ax.plot(freq_plot, mean_pred, color=colors[model_type], lw=2.5, label='平均予測')
        ax.fill_between(freq_plot, ci_lower, ci_upper, color=colors[model_type], alpha=0.3, label='95%信用区間')
        ax.scatter(data['frequency'], data['transmittance'], color='black', s=25, alpha=0.6, label='実験データ', zorder=5)
        ax.set_title(f'磁場 {b_val} T', fontsize=14)
        ax.set_xlabel('周波数 (THz)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    axes[0].set_ylabel('正規化透過率', fontsize=12)
    fig.suptitle(f'モデル "{model_type}" ベイズ推定結果 (背景透磁率モデル)', fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(IMAGE_DIR / f'fit_result_{model_type}.png')
    plt.show()

# --- 6. メイン実行ブロック ---

if __name__ == '__main__':
    print("\n--- メイン処理を開始します ---")
    # 1. データ読み込み
    file_path = "C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\Circular_Polarization_B_Field.xlsx"
    datasets = load_data_full_range(file_path=file_path, sheet_name='Sheet2')
    trans_observed = np.concatenate([d['transmittance'] for d in datasets])
    
    # 2. モデル比較とサンプリング実行
    model_types = ['H_form', 'B_form']
    traces = {}

    for mt in model_types:
        print(f"\n--- モデル '{mt}' のサンプリングを実行中... ---")
        physics_op = BackgroundMuPhysicsModelOp(datasets, model_type=mt)
        
        with pm.Model() as model:
            # --- 物理パラメータの事前分布 ---
            a_scale = pm.TruncatedNormal('a_scale_raw', mu=a_scale_init, sigma=0.3, lower=0.8, upper=3.0)
            g_factor = pm.TruncatedNormal('g_factor_raw', mu=g_factor_init, sigma=0.05, lower=1.9, upper=2.1)
            B4 = pm.Normal('B4_raw', mu=B4_init, sigma=abs(B4_init)*0.5)
            B6 = pm.Normal('B6_raw', mu=B6_init, sigma=abs(B6_init)*0.5)
            # gammaを対数空間でサンプリング
            log_gamma_mu = pm.Normal('log_gamma_mu', mu=np.log(gamma_init), sigma=2.0)
            log_gamma_sigma = pm.HalfNormal('log_gamma_sigma', sigma=1.0)
            log_gamma_offset = pm.Normal('log_gamma_offset', mu=0, sigma=1, shape=7)
            gamma = pm.Deterministic('gamma_raw', pt.exp(log_gamma_mu + log_gamma_offset * log_gamma_sigma))
            
            # --- 背景透磁率モデルのパラメータ事前分布 ---
            mu0_a = pm.Normal('mu0_a_raw', mu=mu0_a_init, sigma=0.2)
            mu0_b = pm.Normal('mu0_b_raw', mu=mu0_b_init, sigma=0.1)
            mu0_c = pm.Normal('mu0_c_raw', mu=mu0_c_init, sigma=0.1)

            # --- 物理モデルと尤度 ---
            mu = physics_op(a_scale, gamma, g_factor, B4, B6, mu0_a, mu0_b, mu0_c)
            
            sigma = pm.HalfCauchy('sigma_raw', beta=0.05)
            nu = pm.Gamma('nu_raw', alpha=2, beta=0.1)
            Y_obs = pm.StudentT('Y_obs', mu=mu, sigma=sigma, nu=nu, observed=trans_observed)
            
            # --- フォレストプロット用のエイリアス ---
            pm.Deterministic('a_scale', a_scale)
            pm.Deterministic('g_factor', g_factor)
            pm.Deterministic('B4', B4)
            pm.Deterministic('B6', B6)
            pm.Deterministic('gamma', gamma)
            pm.Deterministic('mu0_a', mu0_a)
            pm.Deterministic('mu0_b', mu0_b)
            pm.Deterministic('mu0_c', mu0_c)

            # --- サンプリング実行 ---
            traces[mt] = pm.sample(2000, tune=2000, chains=4, cores=os.cpu_count(),
                                   target_accept=0.9, idata_kwargs={"log_likelihood": True})
        
        print(f"--- モデル '{mt}' のサンプリング完了 ---")
        summary_vars = ['a_scale', 'g_factor', 'B4', 'B6', 'gamma', 'mu0_a', 'mu0_b', 'mu0_c']
        print(az.summary(traces[mt], var_names=summary_vars))

    # 3. モデル比較とフォレストプロット
    if len(traces) > 1:
        print("\n--- 3. ベイズ的モデル比較 (LOO-CV) ---")
        compare_df = az.compare(traces)
        print(compare_df)
        az.plot_compare(compare_df, figsize=(10, 5))
        plt.savefig(IMAGE_DIR / 'model_comparison.png')
        plt.show()

        print("\n--- 4. パラメータ事後分布の比較 (フォレストプロット) ---")
        forest_vars = ['a_scale', 'g_factor', 'B4', 'B6', 'mu0_a', 'mu0_b', 'mu0_c']
        az.plot_forest(list(traces.values()), model_names=list(traces.keys()),
                       var_names=forest_vars, combined=True, figsize=(12, 8))
        plt.suptitle('パラメータ事後分布の比較 (フォレストプロット)', fontsize=16)
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig(IMAGE_DIR / 'forest_plot.png')
        plt.show()


    # 5. 結果の可視化
    print("\n--- 5. 結果の可視化 ---")
    for mt, trace in traces.items():
        print(f"\n--- モデル '{mt}' の結果を処理中 ---")
        plot_inference_results(datasets, trace, mt)

    print("\nすべての処理が完了しました。")
