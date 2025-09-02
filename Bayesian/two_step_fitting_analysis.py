# two_step_fitting_analysis.py
""""
To Do list
1. 計算方法の見直し
 ・ε_BG = a + b * ( B - 9.0 ) / 9.0 + c * ( T - 1.5 ) / 1.5に変更
 ・各磁場事にモデルの予測値を計算→プロット
2. ピーク位置のずれはプロットする必要なし（表形式でまとめる）
    この解析でうまくいかない場合は、バックグラウンドの処理を行いピーク位置の解析にのみ注力する

3. H_form, B_formの切り替わり
4. モデル評価できるようにする
"""

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
IMAGE_DIR = pathlib.Path(__file__).parent / "two_step_analysis_results"
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
# (get_hamiltonian, calculate_susceptibility, calculate_normalized_transmissionは以前のコードと同じ)
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
    gamma_array_adj = np.pad(gamma_array, (0, len(delta_E) - len(gamma_array)), 'edge') if len(gamma_array) < len(delta_E) else gamma_array[:len(delta_E)]
    numerator = delta_pop * transition_strength
    denominator = (omega_0**2)[:, np.newaxis] - omega_array**2 - (1j * gamma_array_adj[:, np.newaxis] * omega_array)
    safe_denominator = np.where(np.abs(denominator) < 1e-20, 1e-20, denominator)
    chi_array = np.sum(numerator[:, np.newaxis] / safe_denominator, axis=0)
    return -chi_array

def calculate_normalized_transmission(omega_array: np.ndarray, mu_r_array: np.ndarray, d: float, eps_bg: float) -> np.ndarray:
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

# --- 3. データ処理と解析ステップ ---

def load_and_split_data(file_path: str, sheet_name: str, cutoff_freq: float) -> Dict[str, List[Dict[str, Any]]]:
    """データを読み込み、高周波と低周波領域に分割する。"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    except Exception as e:
        raise FileNotFoundError(f"Excelファイル '{file_path}' が読み込めません: {e}")
    
    freq_col = 'Frequency (THz)'
    trans_cols = [col for col in df.columns if 'Transmittance' in col and 'T)' in col]
    
    low_freq_datasets, high_freq_datasets = [], []
    for col in trans_cols:
        b_match = re.search(r'\((\d+\.?\d*)T\)', col)
        if not b_match: continue
        b_value = float(b_match.group(1))
         
        df_temp = df[[freq_col, col]].copy()
        df_temp[freq_col] = pd.to_numeric(df_temp[freq_col], errors='coerce')
        df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
        df_clean = df_temp.dropna()
        
        freq, trans = df_clean[freq_col].values.astype(float), df_clean[col].values.astype(float)
        
        low_mask = freq < cutoff_freq
        high_mask = freq >= cutoff_freq
        
        # 領域ごとに正規化
        min_low, max_low = trans[low_mask].min(), trans[low_mask].max()
        trans_norm_low = (trans[low_mask] - min_low) / (max_low - min_low)
        
        min_high, max_high = trans[high_mask].min(), trans[high_mask].max()
        trans_norm_high = (trans[high_mask] - min_high) / (max_high - min_high)

        base_data = {'temperature': TEMPERATURE, 'b_field': b_value}
        low_freq_datasets.append({**base_data, 'frequency': freq[low_mask], 'transmittance': trans_norm_low, 'omega': freq[low_mask] * 1e12 * 2 * np.pi})
        high_freq_datasets.append({**base_data, 'frequency': freq[high_mask], 'transmittance': trans_norm_high, 'omega': freq[high_mask] * 1e12 * 2 * np.pi})
        
    return {'low_freq': low_freq_datasets, 'high_freq': high_freq_datasets}

def fit_cavity_modes(datasets: List[Dict[str, Any]]) -> Dict[str, float]:
    """Step 1: 高周波データにファブリ・ペローモデルをフィットし、光学的パラメータを決定する。"""
    print("\n--- Step 1: 高周波領域の共振器モードをフィッティング ---")
    
    def cavity_model(freq_thz, d_fit, eps_bg_fit):
        omega = freq_thz * 1e12 * 2 * np.pi
        # 磁気的効果がないため mu_r = 1
        return calculate_normalized_transmission(omega, np.ones_like(omega), d_fit, eps_bg_fit)

    fit_params = {'d': [], 'eps_bg': []}
    for data in datasets:
        try:
            popt, _ = curve_fit(cavity_model, data['frequency'], data['transmittance'], p0=[d_init, eps_bg_init])
            fit_params['d'].append(popt[0])
            fit_params['eps_bg'].append(popt[1])
            print(f"  磁場 {data['b_field']} T: d = {popt[0]*1e6:.2f} um, eps_bg = {popt[1]:.3f}")
        except RuntimeError:
            print(f"  磁場 {data['b_field']} T: フィッティングに失敗しました。")
    
    final_params = {key: float(np.mean(val)) for key, val in fit_params.items() if val}
    print("----------------------------------------------------")
    print(f"▶ Step 1 結果 (平均値): d = {final_params.get('d', 0)*1e6:.2f} um, eps_bg = {final_params.get('eps_bg', 0):.3f}")
    print("----------------------------------------------------")
    return final_params

class MagneticModelOp(Op):
    """Step 2: 低周波領域の磁気パラメータを推定するためのPyMC Op。"""
    def __init__(self, datasets: List[Dict[str, Any]], d_fixed: float, eps_bg_fixed: float):
        self.datasets = datasets
        self.d = d_fixed
        self.eps_bg = eps_bg_fixed
        self.itypes = [pt.dscalar, pt.dvector, pt.dscalar, pt.dscalar] # a_scale, gamma, B4, B6
        self.otypes = [pt.dvector]

    def perform(self, node, inputs, output_storage):
        a_scale, gamma, B4, B6 = inputs
        full_predicted_y = []
        for data in self.datasets:
            H = get_hamiltonian(data['b_field'], g_factor_init, B4, B6)
            chi_raw = calculate_susceptibility(data['omega'], H, data['temperature'], gamma)
            G0 = a_scale * mu0 * N_spin * (g_factor_init * muB)**2 / (2 * hbar)
            chi = G0 * chi_raw
            mu_r = 1 + chi # H_formを仮定
            predicted_y = calculate_normalized_transmission(data['omega'], mu_r, self.d, self.eps_bg)
            full_predicted_y.append(predicted_y)
        output_storage[0][0] = np.concatenate(full_predicted_y)

def run_bayesian_magnetic_fit(datasets: List[Dict[str, Any]], optical_params: Dict[str, float]) -> az.InferenceData:
    """Step 2: 低周波データを用いて磁気パラメータのベイズ推定を実行する。"""
    print("\n--- Step 2: 低周波領域の磁気パラメータをベイズ推定 ---")
    trans_observed = np.concatenate([d['transmittance'] for d in datasets])
    
    with pm.Model() as model:
        a_scale = pm.TruncatedNormal('a_scale', mu=a_scale_init, sigma=0.5, lower=0.1)
        B4 = pm.Normal('B4', mu=B4_init, sigma=abs(B4_init)*0.5)
        B6 = pm.Normal('B6', mu=B6_init, sigma=abs(B6_init)*0.5)
        
        log_gamma_mu = pm.Normal('log_gamma_mu', mu=np.log(gamma_init), sigma=2.0)
        log_gamma_sigma = pm.HalfNormal('log_gamma_sigma', sigma=1.0)
        log_gamma_offset = pm.Normal('log_gamma_offset', mu=0, sigma=1, shape=7)
        gamma = pm.Deterministic('gamma', pt.exp(log_gamma_mu + log_gamma_offset * log_gamma_sigma))
        
        op = MagneticModelOp(datasets, d_fixed=optical_params['d'], eps_bg_fixed=optical_params['eps_bg'])
        mu = op(a_scale, gamma, B4, B6)
        
        sigma = pm.HalfCauchy('sigma', beta=0.05)
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=trans_observed)
        
        trace = pm.sample(2000, tune=2000, chains=4, cores=os.cpu_count(), target_accept=0.9)
    
    print("----------------------------------------------------")
    print("▶ Step 2 結果 (パラメータ平均値):")
    summary = az.summary(trace, var_names=['a_scale', 'B4', 'B6', 'gamma'])
    print(summary)
    print("----------------------------------------------------")
    return trace

def validate_and_plot_full_spectrum(all_datasets: List[Dict[str, Any]], optical_params: Dict[str, float], magnetic_trace: az.InferenceData):
    """Step 3: 統合パラメータで全領域スペクトルを検証し、ピークのずれを評価する。"""
    print("\n--- Step 3: 統合パラメータによる全領域スペクトルの検証 ---")
    
    mag_params_mean = {var: magnetic_trace['posterior'][var].mean().item() for var in ['a_scale', 'B4', 'B6']}
    mag_params_mean['gamma'] = magnetic_trace['posterior']['gamma'].mean(dim=('chain', 'draw')).values
    
    num_conditions = len(all_datasets)
    fig, axes = plt.subplots(1, num_conditions, figsize=(7 * num_conditions, 6), sharey=True)
    if num_conditions == 1: axes = [axes]

    for i, data in enumerate(all_datasets):
        ax = axes[i]
        freq_plot = np.linspace(data['frequency'].min(), data['frequency'].max(), 500)
        omega_plot = freq_plot * 1e12 * 2 * np.pi
        
        # 統合パラメータで理論スペクトルを計算
        H = get_hamiltonian(data['b_field'], g_factor_init, mag_params_mean['B4'], mag_params_mean['B6'])
        chi_raw = calculate_susceptibility(omega_plot, H, data['temperature'], mag_params_mean['gamma'])
        G0 = mag_params_mean['a_scale'] * mu0 * N_spin * (g_factor_init * muB)**2 / (2 * hbar)
        chi = G0 * chi_raw
        mu_r = 1 + chi
        
        # 全領域データで正規化し直す
        full_trans_theory = calculate_normalized_transmission(omega_plot, mu_r, optical_params['d'], optical_params['eps_bg'])
        
        # 実験データも全領域で正規化
        min_exp, max_exp = data['transmittance_full'].min(), data['transmittance_full'].max()
        trans_norm_full = (data['transmittance_full'] - min_exp) / (max_exp - min_exp)
        
        ax.plot(freq_plot, full_trans_theory, color='red', lw=2.5, label='統合モデル予測')
        ax.scatter(data['frequency'], trans_norm_full, color='black', s=25, alpha=0.6, label='実験データ (全領域)')
        ax.set_title(f"磁場 {data['b_field']} T", fontsize=14)
        ax.set_xlabel('周波数 (THz)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # ピーク位置のずれを分析・プロット
        exp_peaks, _ = find_peaks(-trans_norm_full, height=-0.9, distance=8)
        pred_peaks, _ = find_peaks(-full_trans_theory, height=-0.9, distance=8)
        exp_peak_freqs = data['frequency'][exp_peaks]
        pred_peak_freqs = freq_plot[pred_peaks]
        
        if len(exp_peak_freqs) > 0 and len(pred_peak_freqs) > 0:
            tree = KDTree(pred_peak_freqs.reshape(-1, 1))
            _, closest_indices = tree.query(exp_peak_freqs.reshape(-1, 1))
            
            # Ensure closest_indices is always an array
            closest_indices = np.atleast_1d(closest_indices)

            print(f"\n--- 磁場 {data['b_field']} T のピーク位置のずれ (GHz) ---")
            peak_diffs_ghz = []
            for j, exp_freq in enumerate(exp_peak_freqs):
                pred_freq = pred_peak_freqs[closest_indices[j]]
                diff_ghz = (pred_freq - exp_freq) * 1000
                peak_diffs_ghz.append(diff_ghz)
                print(f"  Exp: {exp_freq:.4f} THz, Pred: {pred_freq:.4f} THz, Diff: {diff_ghz: >+9.2f} GHz")
            
            ax2 = ax.twinx()
            ax2.bar(exp_peak_freqs, peak_diffs_ghz, width=0.015, color='purple', alpha=0.6, label='ピーク位置のずれ (GHz)')
            ax2.set_ylabel('ピーク位置のずれ (GHz)', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')
            ax2.axhline(0, color='purple', linestyle='--', lw=1)
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend() # 凡例を一度作成
            ax2.legend(lines + lines2, labels + labels2, loc='upper right')
            ax.get_legend().remove()

    axes[0].set_ylabel('正規化透過率', fontsize=12)
    fig.suptitle('Step 3: 統合モデルによる全領域スペクトル検証', fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(IMAGE_DIR / 'step3_validation_full_spectrum.png')
    plt.show()

# --- 7. メイン実行ブロック ---

def load_data_full_range(file_path: str, sheet_name: str) -> List[Dict[str, Any]]:
    """全周波数範囲のデータを読み込む。"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    except Exception as e:
        raise FileNotFoundError(f"Excelファイル '{file_path}' が読み込めません: {e}")

    freq_col = 'Frequency (THz)'
    trans_cols = [col for col in df.columns if 'Transmittance' in col and 'T)' in col]
    
    all_datasets = []
    for col in trans_cols:
        b_match = re.search(r'\((\d+\.?\d*)T\)', col)
        if not b_match: continue
        b_value = float(b_match.group(1))
        
        df_temp = df[[freq_col, col]].copy()
        df_temp[freq_col] = pd.to_numeric(df_temp[freq_col], errors='coerce')
        df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
        df_clean = df_temp.dropna()
        
        freq, trans = df_clean[freq_col].values.astype(float), df_clean[col].values.astype(float)

        base_data = {'temperature': TEMPERATURE, 'b_field': b_value}
        all_datasets.append({
            **base_data,
            'frequency': freq,
            'transmittance_full': trans,
            'omega': freq * 1e12 * 2 * np.pi
        })
        
    return all_datasets

if __name__ == '__main__':
    print("\n--- 2段階フィッティング解析を開始します ---")
    
    # データの読み込みと分割
    file_path = "C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\Circular_Polarization_B_Field.xlsx"
    all_data_raw = load_data_full_range(file_path, 'Sheet2') # Step3の検証用に全データも読み込む
    split_data = load_and_split_data(file_path, 'Sheet2', cutoff_freq=0.8)
    
    # Step 1: 光学的パラメータの決定
    optical_params = fit_cavity_modes(split_data['high_freq'])
    
    if not optical_params:
        print("Step 1で光学的パラメータを決定できませんでした。プログラムを終了します。")
    else:
        # Step 2: 磁気パラメータのベイズ推定
        magnetic_trace = run_bayesian_magnetic_fit(split_data['low_freq'], optical_params)
        
        # Step 3: 全領域での検証
        validate_and_plot_full_spectrum(all_data_raw, optical_params, magnetic_trace)

    print("\nすべての解析が完了しました。")
