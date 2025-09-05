# two_step_fitting_analysis_v2.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.graph.op import Op
import os
import pathlib
import re
from typing import List, Dict, Any
from scipy.signal import find_peaks
from scipy.spatial import KDTree
from scipy.optimize import curve_fit

# --- 0. 環境設定 ---
print("--- 0. 環境設定を開始します ---")
os.environ['PYTENSOR_FLAGS'] = 'optimizer=fast_compile,floatX=float64'
try:
    import japanize_matplotlib
except ImportError:
    print("警告: japanize_matplotlib が見つかりません。")
plt.rcParams['figure.dpi'] = 120
IMAGE_DIR = pathlib.Path(__file__).parent / "two_step_analysis_v2_results"
IMAGE_DIR.mkdir(exist_ok=True)
print(f"画像は '{IMAGE_DIR.resolve()}' に保存されます。")

# --- 1. 物理定数とシミュレーション条件 ---
print("--- 1. 物理定数と初期値を設定します ---")
kB = 1.380649e-23; muB = 9.274010e-24; hbar = 1.054571e-34
c = 299792458; mu0 = 4.0 * np.pi * 1e-7
s = 3.5; N_spin = 24 / 1.238 * 1e27
TEMPERATURE = 1.5 # 温度 (K)

# パラメータの初期値
d_init = 157.8e-6; eps_bg_init = 13.14
B4_init = 0.002; B6_init = -0.00003
gamma_init = 0.11e12; a_scale_init = 1.5; g_factor_init = 2.02

# --- 2. 物理モデル関数 ---
def get_hamiltonian(B_ext_z: float, g_factor: float, B4: float, B6: float) -> np.ndarray:
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
    denominator = omega_0[:, np.newaxis] - omega_array - 1j * gamma_array[:, np.newaxis] 
    # ゼロやNaNになるのを防ぐためのより堅牢なチェック
    denominator[np.abs(denominator) < 1e-20] = 1e-20
    denominator[np.isnan(denominator)] = 1e-20
    chi_array = np.sum(numerator[:, np.newaxis] / denominator, axis=0)
    return -chi_array

def calculate_normalized_transmission(omega_array: np.ndarray, mu_r_array: np.ndarray, d: float, eps_bg: float) -> np.ndarray:
    n_complex = np.sqrt(eps_bg * mu_r_array + 0j)
    impe = np.sqrt(mu_r_array / eps_bg + 0j)
    lambda_0 = np.full_like(omega_array, np.inf, dtype=float)
    nonzero_mask = omega_array > 1e-12
    lambda_0[nonzero_mask] = (2 * np.pi * c) / omega_array[nonzero_mask]
    delta = 2 * np.pi * n_complex * d / lambda_0
    
    # オーバーフロー防止のためdeltaの大きさを制限
    delta = np.clip(delta, -700, 700)
    
    numerator = 4 * impe
    exp_pos = np.exp(-1j * delta)
    exp_neg = np.exp(1j * delta)
    denominator = (1 + impe)**2 * exp_pos - (1 - impe)**2 * exp_neg
    safe_mask = np.abs(denominator) > 1e-12
    t = np.zeros_like(denominator, dtype=complex)
    t[safe_mask] = numerator[safe_mask] / denominator[safe_mask]
    transmission = np.abs(t)**2
    
    # 数値安定性のため、異常値を除去
    transmission = np.clip(transmission, 0, 1)
    min_trans, max_trans = np.min(transmission), np.max(transmission)
    return (transmission - min_trans) / (max_trans - min_trans) if max_trans > min_trans else np.full_like(transmission, 0.5)

# --- 3. データ処理と解析ステップ ---
def load_and_split_data(file_path: str, sheet_name: str, cutoff_freq: float) -> Dict[str, List[Dict[str, Any]]]:
    """データを読み込み、高周波と低周波領域に分割する。"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    except Exception as e:
        raise FileNotFoundError(f"Excelファイル '{file_path}' が読み込めません: {e}")
    
    freq_col = 'Frequency (THz)'
    df[freq_col] = pd.to_numeric(df[freq_col], errors='coerce')
    trans_cols = [col for col in df.columns if 'Transmittance' in col and 'T)' in col]
    low_freq_datasets, high_freq_datasets = [], []
    for col in trans_cols:
        b_match = re.search(r'\((\d+\.?\d*)T\)', col)
        if not b_match: continue
        b_value = float(b_match.group(1))
        df_clean = df[[freq_col, col]].dropna()
        freq, trans = df_clean[freq_col].values.astype(np.float64), df_clean[col].values.astype(np.float64)
        low_mask, high_mask = freq < cutoff_freq, freq >= cutoff_freq
        min_low, max_low = trans[low_mask].min(), trans[low_mask].max()
        trans_norm_low = (trans[low_mask] - min_low) / (max_low - min_low)
        min_high, max_high = trans[high_mask].min(), trans[high_mask].max()
        trans_norm_high = (trans[high_mask] - min_high) / (max_high - min_high)
        base_data = {'temperature': TEMPERATURE, 'b_field': b_value}
        low_freq_datasets.append({**base_data, 'frequency': freq[low_mask], 'transmittance': trans_norm_low, 'omega': freq[low_mask] * 1e12 * 2 * np.pi})
        high_freq_datasets.append({**base_data, 'frequency': freq[high_mask], 'transmittance': trans_norm_high, 'omega': freq[high_mask] * 1e12 * 2 * np.pi})
    return {'low_freq': low_freq_datasets, 'high_freq': high_freq_datasets}

def fit_cavity_modes(datasets: List[Dict[str, Any]]) -> Dict[str, float]:
    """Step 1: 高周波データから光学的パラメータを決定する。"""
    print("\n--- Step 1: 高周波領域の共振器モードをフィッティング ---")
    def cavity_model(freq_thz, d_fit, eps_bg_fit):
        omega = freq_thz * 1e12 * 2 * np.pi
        return calculate_normalized_transmission(omega, np.ones_like(omega), d_fit, eps_bg_fit)

    fit_params = {'d': [], 'eps_bg': [], 'b_field': []}
    for data in datasets:
        try:
            popt, _ = curve_fit(cavity_model, data['frequency'], data['transmittance'], p0=[d_init, eps_bg_init])
            fit_params['d'].append(popt[0]); fit_params['eps_bg'].append(popt[1]); fit_params['b_field'].append(data['b_field'])
            print(f"  磁場 {data['b_field']} T: d = {popt[0]*1e6:.2f} um, eps_bg = {popt[1]:.3f}")
        except RuntimeError:
            print(f"  磁場 {data['b_field']} T: フィッティングに失敗しました。")
    if not fit_params['d']: return {}
    final_params = {'d': float(np.mean(fit_params['d']))}
    if len(fit_params['b_field']) > 1:
        b_fields, eps_bgs = np.array(fit_params['b_field']), np.array(fit_params['eps_bg'])
        def eps_bg_model(B, a, b): return a + b * (B - 9.0) / 9.0
        popt_eps, _ = curve_fit(eps_bg_model, b_fields, eps_bgs)
        final_params['eps_bg_a'], final_params['eps_bg_b'] = popt_eps[0], popt_eps[1]
        print("----------------------------------------------------")
        print(f"▶ Step 1 結果 (d 平均値): d = {final_params['d']*1e6:.2f} um")
        print(f"▶ Step 1 結果 (eps_bg フィット): a = {final_params['eps_bg_a']:.3f}, b = {final_params['eps_bg_b']:.3f}")
    else:
        final_params['eps_bg_a'], final_params['eps_bg_b'] = float(np.mean(fit_params['eps_bg'])), 0.0
        print(f"▶ Step 1 結果 (d, eps_bg): d = {final_params['d']*1e6:.2f} um, eps_bg = {final_params['eps_bg_a']:.3f}")
    print("----------------------------------------------------")
    return final_params

class MagneticModelOp(Op):
    """Step 2: 低周波領域の磁気パラメータを推定するためのPyMC Op。"""
    def __init__(self, datasets: List[Dict[str, Any]], d_fixed: float, eps_bg_a: float, eps_bg_b: float, model_type: str):
        self.datasets, self.d, self.eps_bg_a, self.eps_bg_b = datasets, d_fixed, eps_bg_a, eps_bg_b
        self.model_type = model_type
        self.itypes = [pt.dscalar, pt.dvector, pt.dscalar, pt.dscalar, pt.dscalar]; self.otypes = [pt.dvector]
    def perform(self, node, inputs, output_storage):
        a_scale, gamma, g_factor, B4, B6 = inputs; full_predicted_y = []
        for data in self.datasets:
            eps_bg = self.eps_bg_a + self.eps_bg_b * (data['b_field'] - 9.0) / 9.0
            H = get_hamiltonian(data['b_field'], g_factor, B4, B6)
            chi_raw = calculate_susceptibility(data['omega'], H, data['temperature'], gamma)
            G0 = a_scale * mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
            chi = G0 * chi_raw
            
            # モデルタイプに応じて透磁率を計算
            if self.model_type == 'H_form':
                mu_r = 1 + chi
            else:  # B_form
                mu_r = 1 / (1 - chi)
            
            predicted_y = calculate_normalized_transmission(data['omega'], mu_r, self.d, eps_bg)
            full_predicted_y.append(predicted_y)
        output_storage[0][0] = np.concatenate(full_predicted_y)

def run_bayesian_magnetic_fit(datasets: List[Dict[str, Any]], optical_params: Dict[str, float], model_type: str = 'H_form') -> az.InferenceData:
    """Step 2: 低周波データを用いて磁気パラメータのベイズ推定を実行する。"""
    print(f"\n--- Step 2: 低周波領域の磁気パラメータをベイズ推定 (モデル: {model_type}) ---")
    trans_observed = np.concatenate([d['transmittance'] for d in datasets])
    with pm.Model() as model:
        a_scale = pm.TruncatedNormal('a_scale', mu=a_scale_init, sigma=0.5, lower=0.1)
        g_factor = pm.TruncatedNormal('g_factor', mu=g_factor_init, sigma=0.05, lower=1.9, upper=2.1)
        B4 = pm.Normal('B4', mu=B4_init, sigma=abs(B4_init)*0.5)
        B6 = pm.Normal('B6', mu=B6_init, sigma=abs(B6_init)*0.5)
        log_gamma_mu = pm.Normal('log_gamma_mu', mu=np.log(gamma_init), sigma=2.0)
        log_gamma_sigma = pm.HalfNormal('log_gamma_sigma', sigma=1.0)
        log_gamma_offset = pm.Normal('log_gamma_offset', mu=0, sigma=1, shape=7)
        gamma = pm.Deterministic('gamma', pt.exp(log_gamma_mu + log_gamma_offset * log_gamma_sigma))
        op = MagneticModelOp(datasets, d_fixed=optical_params['d'], eps_bg_a=optical_params['eps_bg_a'], eps_bg_b=optical_params['eps_bg_b'], model_type=model_type)
        mu = op(a_scale, gamma, g_factor, B4, B6)
        sigma = pm.HalfCauchy('sigma', beta=0.05)
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=trans_observed)
        
        # より安定したサンプリング設定
        try:
            trace = pm.sample(2500, tune=2500, chains=4, cores=os.cpu_count(), target_accept=0.95, random_seed=42)
        except Exception as e:
            print(f"高精度サンプリングに失敗: {e}")
            print("基本設定でリトライします...")
            trace = pm.sample(2000, tune=2000, chains=4, cores=os.cpu_count(), target_accept=0.9, random_seed=42)
    
    print("----------------------------------------------------")
    print("▶ Step 2 結果 (サマリー):")
    summary = az.summary(trace, var_names=['a_scale', 'g_factor', 'B4', 'B6', 'gamma'])
    print(summary)
    print("----------------------------------------------------")
    return trace

def validate_and_plot_full_spectrum(all_datasets: List[Dict[str, Any]], optical_params: Dict[str, float], magnetic_trace: az.InferenceData, n_samples: int = 100, model_type: str = 'H_form'):
    """Step 3: 統合パラメータで全領域スペクトルを検証し、信用区間と共にプロット。ピークのずれはコンソールに出力。"""
    print(f"\n--- Step 3: 統合パラメータによる全領域スペクトルの検証 ({model_type}) ---")
    posterior = magnetic_trace["posterior"]
    num_conditions = len(all_datasets)
    fig, axes = plt.subplots(1, num_conditions, figsize=(7 * num_conditions, 6), sharey=True)
    if num_conditions == 1: axes = [axes]

    for i, data in enumerate(all_datasets):
        ax = axes[i]
        freq_plot = np.linspace(data['frequency'].min(), data['frequency'].max(), 500)
        omega_plot = freq_plot * 1e12 * 2 * np.pi
        current_eps_bg = optical_params['eps_bg_a'] + optical_params['eps_bg_b'] * (data['b_field'] - 9.0) / 9.0
        
        # --- 信用区間のための計算 ---
        total_samples = posterior.chain.size * posterior.draw.size
        indices = np.random.choice(total_samples, min(n_samples, total_samples), replace=False)
        predictions = []
        for idx in indices:
            chain, draw = idx // posterior.draw.size, idx % posterior.draw.size
            params = {var: posterior[var].values[chain, draw] for var in ['a_scale', 'g_factor', 'B4', 'B6', 'gamma']}
            H = get_hamiltonian(data['b_field'], float(params['g_factor']), float(params['B4']), float(params['B6']))
            chi_raw = calculate_susceptibility(omega_plot, H, data['temperature'], params['gamma'])
            G0 = params['a_scale'] * mu0 * N_spin * (float(params['g_factor']) * muB)**2 / (2 * hbar)
            chi = G0 * chi_raw
            
            # H_form vs B_formでのmu_r計算
            if model_type == 'B_form':
                mu_r = 1 / (1 - chi)
            else:  # H_form
                mu_r = 1 + chi
                
            pred_y = calculate_normalized_transmission(omega_plot, mu_r, optical_params['d'], current_eps_bg)
            predictions.append(pred_y)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        ci_lower, ci_upper = np.percentile(predictions, [2.5, 97.5], axis=0)
        
        # --- 全領域データでの正規化とプロット ---
        min_exp, max_exp = data['transmittance_full'].min(), data['transmittance_full'].max()
        trans_norm_full = (data['transmittance_full'] - min_exp) / (max_exp - min_exp)
        
        ax.plot(freq_plot, mean_pred, color='red', lw=2.5, label=f'平均予測 ({model_type})')
        ax.fill_between(freq_plot, ci_lower, ci_upper, color='red', alpha=0.3, label='95%信用区間')
        ax.scatter(data['frequency'], trans_norm_full, color='black', s=25, alpha=0.6, label='実験データ (全領域)')
        ax.set_title(f"磁場 {data['b_field']} T ({model_type})", fontsize=14); ax.set_xlabel('周波数 (THz)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6); ax.legend()
        
        # --- ピーク位置のずれをコンソールに出力 ---
        exp_peaks, _ = find_peaks(-trans_norm_full, height=-0.9, distance=8)
        pred_peaks, _ = find_peaks(-mean_pred, height=-0.9, distance=8)
        exp_peak_freqs, pred_peak_freqs = data['frequency'][exp_peaks], freq_plot[pred_peaks]
        if len(exp_peak_freqs) > 0 and len(pred_peak_freqs) > 0:
            tree = KDTree(pred_peak_freqs.reshape(-1, 1))
            _, closest_indices = tree.query(exp_peak_freqs.reshape(-1, 1))
            closest_indices = np.atleast_1d(closest_indices)
            print(f"\n--- 磁場 {data['b_field']} T のピーク位置のずれ (GHz) ---")
            for j, exp_freq in enumerate(exp_peak_freqs):
                pred_freq = pred_peak_freqs[closest_indices[j]]
                diff_ghz = (pred_freq - exp_freq) * 1000
                print(f"  Exp: {exp_freq:.4f} THz, Pred: {pred_freq:.4f} THz, Diff: {diff_ghz: >+9.2f} GHz")

    axes[0].set_ylabel('正規化透過率', fontsize=12)
    fig.suptitle(f'Step 3: 統合モデルによる全領域スペクトル検証 ({model_type})', fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95)); plt.savefig(IMAGE_DIR / f'step3_validation_full_spectrum_{model_type}.png'); plt.show()

def load_data_full_range(file_path: str, sheet_name: str) -> List[Dict[str, Any]]:
    """全周波数範囲のデータを読み込む。"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    except Exception as e:
        raise FileNotFoundError(f"Excelファイル '{file_path}' が読み込めません: {e}")
    freq_col = 'Frequency (THz)'
    df[freq_col] = pd.to_numeric(df[freq_col], errors='coerce')
    trans_cols = [col for col in df.columns if 'Transmittance' in col and 'T)' in col]
    all_datasets = []
    for col in trans_cols:
        b_match = re.search(r'\((\d+\.?\d*)T\)', col)
        if not b_match: continue
        b_value = float(b_match.group(1))
        df_clean = df[[freq_col, col]].dropna()
        freq, trans = df_clean[freq_col].values.astype(np.float64), df_clean[col].values.astype(np.float64)
        base_data = {'temperature': TEMPERATURE, 'b_field': b_value}
        all_datasets.append({**base_data, 'frequency': freq, 'transmittance_full': trans, 'omega': freq * 1e12 * 2 * np.pi})
    return all_datasets

if __name__ == '__main__':
    print("\n--- 2段階フィッティング解析を開始します ---")
    file_path = "C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\Circular_Polarization_B_Field.xlsx"
    all_data_raw = load_data_full_range(file_path, 'Sheet2')
    split_data = load_and_split_data(file_path, 'Sheet2', cutoff_freq=0.8)
    optical_params = fit_cavity_modes(split_data['high_freq'])
    if not optical_params:
        print("Step 1で光学的パラメータを決定できませんでした。プログラムを終了します。")
    else:
        # H_formとB_formの両方をテスト
        traces = {}
        for model_type in ['H_form', 'B_form']:
            print(f"\n=== {model_type} モデルでの解析 ===")
            magnetic_trace = run_bayesian_magnetic_fit(split_data['low_freq'], optical_params, model_type)
            validate_and_plot_full_spectrum(all_data_raw, optical_params, magnetic_trace, model_type=model_type)
            traces[model_type] = magnetic_trace
        
        # モデル比較
        print("\n=== モデル比較結果 ===")
        for model_type, trace in traces.items():
            waic = az.waic(trace)
            loo = az.loo(trace)
            print(f"{model_type}: WAIC = {waic.waic:.2f} ± {waic.waic_se:.2f}, LOO = {loo.loo:.2f} ± {loo.loo_se:.2f}")
            
        # パラメータ比較
        print("\n=== パラメータ推定値比較 ===")
        for param in ['a_scale', 'g_factor', 'B4', 'B6']:
            h_summary = az.summary(traces['H_form'], var_names=[param])
            b_summary = az.summary(traces['B_form'], var_names=[param])
            print(f"{param}:")
            print(f"  H_form: {h_summary['mean'].iloc[0]:.4f} ± {h_summary['sd'].iloc[0]:.4f}")
            print(f"  B_form: {b_summary['mean'].iloc[0]:.4f} ± {b_summary['sd'].iloc[0]:.4f}")
    print("\nすべての解析が完了しました。")
