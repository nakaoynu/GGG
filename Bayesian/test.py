# bayesian_full_range.py

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

# --- 0. 環境設定 ---
print("--- 環境設定を開始します ---")
# PyTensorのフラグ設定（コンパイルを高速化）
os.environ['PYTENSOR_FLAGS'] = 'optimizer=fast_compile,floatX=float64'

# 日本語フォントの設定
try:
    import japanize_matplotlib
except ImportError:
    print("警告: japanize_matplotlib が見つかりません。グラフの日本語が文字化けする可能性があります。")
    print("インストールコマンド: pip install japanize-matplotlib")

# プロットと画像保存の設定
plt.rcParams['figure.dpi'] = 120
IMAGE_DIR = pathlib.Path(__file__).parent / "pymc_full_range_results"
IMAGE_DIR.mkdir(exist_ok=True)
print(f"画像は '{IMAGE_DIR.resolve()}' に保存されます。")

# --- 1. 物理定数とシミュレーション条件 ---
print("--- 物理定数と初期値を設定します ---")
# 物理定数
kB = 1.380649e-23  # ボルツマン定数 (J/K)
muB = 9.274010e-24 # ボーア磁子 (J/T)
hbar = 1.054571e-34 # ディラック定数 (J*s)
c = 299792458      # 光速 (m/s)
mu0 = 4.0 * np.pi * 1e-7 # 真空の透磁率 (H/m)
s = 3.5            # スピン量子数
N_spin = 24 / 1.238 * 1e27 # GGGのスピン数密度 (m^-3)

# 実験条件 (将来的に変更可能なパラメータ)
TEMPERATURE = 35.0 # 温度 (K)

# パラメータの初期値（事前分布の中心として使用）
d_init = 157.8e-6
eps_bg_init = 13.14
g_factor_init = 2.02
B4_init = 0.002
B6_init = -0.00003
gamma_init = 0.11e12
a_init = 1.5

# --- 2. 物理モデル関数 ---

def get_hamiltonian(B_ext_z, g_factor, B4, B6):
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

def calculate_susceptibility(omega_array, H, T, gamma_array):
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
    
    gamma_array_adj = np.pad(gamma_array, (0, len(delta_E) - len(gamma_array)), 'edge') if len(gamma_array) < len(delta_E) else gamma_array[:len(delta_E)]

    numerator = delta_pop * transition_strength
    denominator = (omega_0**2)[:, np.newaxis] - omega_array**2 - (1j * gamma_array_adj[:, np.newaxis] * omega_array)
    
    safe_denominator = np.where(np.abs(denominator) < 1e-20, 1e-20, denominator)
    chi_array = np.sum(numerator[:, np.newaxis] / safe_denominator, axis=0)
    return -chi_array

def calculate_transmission_intensity(omega_array, mu_r_array, d_val, eps_bg_val):
    """複素透磁率μrから透過強度|t|^2を計算する。"""
    n_complex = np.sqrt(eps_bg_val * mu_r_array + 0j)
    impe = np.sqrt(mu_r_array / eps_bg_val + 0j)
    
    lambda_0 = np.full_like(omega_array, np.inf, dtype=float)
    nonzero_mask = omega_array > 1e-12
    lambda_0[nonzero_mask] = (2 * np.pi * c) / omega_array[nonzero_mask]
    
    delta = 2 * np.pi * n_complex * d_val / lambda_0
    
    numerator = 4 * impe
    denominator = (1 + impe)**2 * np.exp(-1j * delta) - (1 - impe)**2 * np.exp(1j * delta)
    
    safe_mask = np.abs(denominator) > 1e-12
    t = np.zeros_like(denominator, dtype=complex)
    t[safe_mask] = numerator[safe_mask] / denominator[safe_mask]
    return np.abs(t)**2

def calculate_normalized_transmission(omega_array, mu_r_array, d_val, eps_bg_val):
    """透過強度を計算し、[0, 1]の範囲に正規化する。"""
    transmission = calculate_transmission_intensity(omega_array, mu_r_array, d_val, eps_bg_val)
    min_trans, max_trans = np.min(transmission), np.max(transmission)
    return (transmission - min_trans) / (max_trans - min_trans) if max_trans > min_trans else np.full_like(transmission, 0.5)

# --- 3. PyMCカスタムOpクラス ---

class MultiFieldPhysicsModelOp(Op):
    """
    複数の磁場データに対して物理モデル計算をまとめて行うPyMCのOp。
    d と eps_bg を推定変数として追加。
    """
    def __init__(self, omega_arrays, B_values, T_val, model_type):
        self.omega_arrays = omega_arrays
        self.B_values = B_values
        self.T = T_val
        self.model_type = model_type
        # 入力の型を定義: a, gamma, g_factor, B4, B6, d, eps_bg
        self.itypes = [pt.dscalar, pt.dvector, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar]
        # 出力の型を定義
        self.otypes = [pt.dvector]

    def perform(self, node, inputs, output_storage):
        a, gamma, g_factor, B4, B6, d_val, eps_bg_val = inputs
        
        G0 = a * mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
        
        full_predicted_y = []
        for omega_array, b_val in zip(self.omega_arrays, self.B_values):
            H = get_hamiltonian(b_val, g_factor, B4, B6)
            chi_raw = calculate_susceptibility(omega_array, H, self.T, gamma)
            chi = G0 * chi_raw
            
            mu_r = 1 + chi if self.model_type == 'H_form' else 1 / (1 - chi)
            
            predicted_y = calculate_normalized_transmission(omega_array, mu_r, d_val, eps_bg_val)
            full_predicted_y.append(predicted_y)
            
        output_storage[0][0] = np.concatenate(full_predicted_y)

# --- 4. データ読み込み関数 (全領域対応) ---

def load_multi_field_data_full_range(file_path=None, sheet_name='Sheet2'):
    """Excelまたは手動データから全領域の実験データを読み込む。"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
        print(f"Excelファイル '{file_path}' の読み込みに成功。")
    except Exception as e:
        print(f"Excelファイルの読み込みに失敗: {e}。手動データを使用します。")
        # 手動データは省略。必要であればここに追加
        raise FileNotFoundError("手動データが定義されていません。Excelファイルパスを確認してください。")

    freq_col = 'Frequency (THz)'
    trans_cols = [col for col in df.columns if 'Transmittance' in col and 'T)' in col]
    
    multi_field_data = {}
    for col in trans_cols:
        b_match = re.search(r'\((\d+\.?\d*)T\)', col)
        if not b_match: continue
        b_value = float(b_match.group(1))
        
        # NaNを含む行を除去
        df_clean = df[[freq_col, col]].dropna()
        freq = df_clean[freq_col].values.astype(np.float64)
        
        # 全領域データで正規化
        min_exp, max_exp = df_clean[col].min(), df_clean[col].max()
        trans = df_clean[col].values
        trans_normalized = (trans - min_exp) / (max_exp - min_exp) if max_exp > min_exp else np.full_like(trans, 0.5)

        multi_field_data[b_value] = {
            'frequency': freq,
            'transmittance': trans_normalized,
            'omega': freq * 1e12 * 2 * np.pi
        }
        print(f"磁場 {b_value} T: {len(freq)}点のデータを処理しました。周波数範囲: {freq.min():.3f} - {freq.max():.3f} THz")
        
    return multi_field_data

# --- 5. 結果可視化関数 ---

def plot_inference_results(multi_field_data, trace, model_type, n_samples=200):
    """ベイズ推定の結果（平均予測、95%信用区間）をプロットする。"""
    fig, axes = plt.subplots(1, len(multi_field_data), figsize=(7 * len(multi_field_data), 6), sharey=True)
    if len(multi_field_data) == 1: axes = [axes]
    
    sorted_b_values = sorted(multi_field_data.keys())
    
    total_samples = trace.posterior.chain.size * trace.posterior.draw.size
    indices = np.random.choice(total_samples, min(n_samples, total_samples), replace=False)
    
    for i, b_val in enumerate(sorted_b_values):
        ax = axes[i]
        data = multi_field_data[b_val]
        
        freq_plot = np.linspace(data['frequency'].min(), data['frequency'].max(), 300)
        omega_plot = freq_plot * 1e12 * 2 * np.pi
        
        predictions = []
        for idx in indices:
            chain = idx // trace.posterior.draw.size
            draw = idx % trace.posterior.draw.size
            
            # 1サンプル分の全パラメータを取得
            params = {var: trace.posterior[var].values[chain, draw] for var in trace.posterior.data_vars}
            
            G0_s = params['a'] * mu0 * N_spin * (params['g_factor'] * muB)**2 / (2 * hbar)
            H = get_hamiltonian(b_val, params['g_factor'], params['B4'], params['B6'])
            chi_raw = calculate_susceptibility(omega_plot, H, TEMPERATURE, params['gamma'])
            chi = G0_s * chi_raw
            mu_r = 1 + chi if model_type == 'H_form' else 1 / (1 - chi)
            pred_y = calculate_normalized_transmission(omega_plot, mu_r, params['d'], params['eps_bg'])
            predictions.append(pred_y)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        ci_lower = np.percentile(predictions, 2.5, axis=0)
        ci_upper = np.percentile(predictions, 97.5, axis=0)
        
        ax.plot(freq_plot, mean_pred, color='red', lw=2.5, label='平均予測')
        ax.fill_between(freq_plot, ci_lower, ci_upper, color='red', alpha=0.3, label='95%信用区間')
        ax.scatter(data['frequency'], data['transmittance'], color='black', s=25, alpha=0.6, label='実験データ', zorder=5)
        
        ax.set_title(f'磁場 {b_val} T', fontsize=14)
        ax.set_xlabel('周波数 (THz)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    axes[0].set_ylabel('正規化透過率', fontsize=12)
    fig.suptitle(f'モデル "{model_type}" ベイズ推定結果 (全領域)', fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(IMAGE_DIR / f'result_full_range_{model_type}.png')
    plt.show()

# --- 6. メイン実行ブロック ---

if __name__ == '__main__':
    # 1. データ読み込み
    print("\n--- 1. 全領域の実験データを読み込みます ---")
    file_path = "C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\Circular_Polarization_B_Field.xlsx"
    multi_field_data = load_multi_field_data_full_range(file_path=file_path, sheet_name='Sheet2')
    
    # 2. PyMCモデルのためのデータ準備
    sorted_b_values = sorted(multi_field_data.keys())
    omega_arrays = [multi_field_data[b]['omega'] for b in sorted_b_values]
    trans_observed = np.concatenate([multi_field_data[b]['transmittance'] for b in sorted_b_values])
    
    # 3. モデル比較とサンプリング実行
    print("\n--- 2. ベイズ推定とモデル比較を開始します ---")
    model_types = ['H_form', 'B_form']
    traces = {}

    for mt in model_types:
        print(f"\n--- モデル '{mt}' のサンプリングを実行中... ---")
        physics_op = MultiFieldPhysicsModelOp(omega_arrays, sorted_b_values, T_val=TEMPERATURE, model_type=mt)
        
        with pm.Model() as model:
            # --- 事前分布の設定 (発散抑制のため、現実的な範囲に設定) ---
            d = pm.TruncatedNormal('d', mu=d_init, sigma=d_init*0.02, lower=d_init*0.9, upper=d_init*1.1)
            eps_bg = pm.TruncatedNormal('eps_bg', mu=eps_bg_init, sigma=0.5, lower=11.0, upper=15.0)
            a = pm.TruncatedNormal('a', mu=a_init, sigma=0.5, lower=0.1, upper=4.0)
            g_factor = pm.TruncatedNormal('g_factor', mu=g_factor_init, sigma=0.05, lower=1.9, upper=2.1)
            B4 = pm.Normal('B4', mu=B4_init, sigma=abs(B4_init)*0.5)
            B6 = pm.Normal('B6', mu=B6_init, sigma=abs(B6_init)*0.5)
            
            # 緩和係数gamma (階層モデル + 非中心化)
            gamma_log_mu = pm.Normal('gamma_log_mu', mu=np.log(gamma_init), sigma=0.5)
            gamma_log_sigma = pm.HalfNormal('gamma_log_sigma', sigma=0.5)
            gamma_log_offset = pm.Normal('gamma_log_offset', mu=0, sigma=1, shape=7)
            gamma = pm.Deterministic('gamma', pt.exp(gamma_log_mu + gamma_log_offset * gamma_log_sigma))

            # --- 物理モデルと尤度 ---
            mu = physics_op(a, gamma, g_factor, B4, B6, d, eps_bg)
            
            sigma = pm.HalfCauchy('sigma', beta=0.05)
            nu = pm.Gamma('nu', alpha=2, beta=0.1)
            Y_obs = pm.StudentT('Y_obs', mu=mu, sigma=sigma, nu=nu, observed=trans_observed)
            
            # --- サンプリング実行 ---
            traces[mt] = pm.sample(2000, tune=2000, chains=4, cores=os.cpu_count(),
                                   target_accept=0.95, idata_kwargs={"log_likelihood": True})
        
        print(f"--- モデル '{mt}' のサンプリング完了 ---")
        summary_vars = ['d', 'eps_bg', 'a', 'g_factor', 'B4', 'B6', 'gamma', 'sigma', 'nu']
        print(az.summary(traces[mt], var_names=summary_vars))

    # 4. モデル比較
    if len(traces) > 1:
        print("\n--- 3. ベイズ的モデル比較 (LOO-CV) ---")
        compare_df = az.compare(traces)
        print(compare_df)
        az.plot_compare(compare_df, figsize=(10, 5))
        plt.savefig(IMAGE_DIR / 'model_comparison.png')
        plt.show()

    # 5. 結果の可視化と分析
    print("\n--- 4. 結果の可視化と分析 ---")
    if len(traces) > 1:
        best_model_name = compare_df.index[0]
        print(f"最良モデルは '{best_model_name}' です。")
    elif len(traces) == 1:
        best_model_name = list(traces.keys())[0]
        print(f"単一モデル '{best_model_name}' の結果を表示します。")
    else:
        print("警告: 有効なモデル結果がありません。可視化をスキップします。")
        best_model_name = None

    for mt, trace in traces.items():
        print(f"\n--- モデル '{mt}' の結果を処理中 ---")
        plot_inference_results(multi_field_data, trace, mt)
        
        az.plot_trace(trace, var_names=['d', 'eps_bg', 'a', 'g_factor', 'B4', 'B6'])
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / f'plot_trace_{mt}.png')
        plt.show()

    print("\nすべての処理が完了しました。")