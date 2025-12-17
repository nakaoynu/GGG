# unified_weighted_bayesian_fitting.py - 磁場・温度一括ベイズ推定プログラム
#
# 【概要】
# このプログラムは、以下の2つのデータパターンを一括処理してベイズ推定を実行します:
#   ① 温度変数、磁場固定: [B, T] = [9.0, 4], [9.0, 10], [9.0, 20], ...
#   ② 温度固定、磁場変数: [B, T] = [1.0, 4], [2.0, 4], [3.0, 4], ...
#
# 【特徴】
# - weighted_bayesian_fitting_completed.pyと同様の物理モデル・手法を使用
# - γは線形の温度依存性を示す設定
# - ε_bgはそれぞれの磁場・温度で独立に非線形最小二乗法で決定
# - その他の磁気パラメータは重み付きベイズ推定で一括決定
# - H_formとB_formの両モデルに対応
#
# 【データ要件】
# Excelファイルに以下のデータが必要:
# - 温度変化データ: 各温度列 (例: '4K', '10K', '20K', ...)
# - 磁場変化データ: 各磁場列 (例: '1T', '2T', '3T', ...)

# ============================================================================
# 【重要】CPU並列化の環境変数設定
# NumPy/SciPyのインポート前に設定する必要がある
# ============================================================================
import os
import pathlib

def _setup_cpu_threads():
    """NumPyインポート前にCPU並列スレッド数を設定"""
    try:
        # YAMLだけ先にインポート（軽量）
        import yaml
        config_path = pathlib.Path(__file__).parent / "config_selected_datasets.yml"
        with open(config_path, 'r', encoding='utf-8') as f:
            temp_config = yaml.safe_load(f)
        
        exec_config = temp_config.get('execution', {})
        threads_per_chain = exec_config.get('threads_per_chain', None)
        
        if threads_per_chain is not None:
            threads_str = str(threads_per_chain)
            # 全ての並列ライブラリに設定
            os.environ['OMP_NUM_THREADS'] = threads_str
            os.environ['MKL_NUM_THREADS'] = threads_str
            os.environ['OPENBLAS_NUM_THREADS'] = threads_str
            os.environ['NUMEXPR_NUM_THREADS'] = threads_str
            os.environ['VECLIB_MAXIMUM_THREADS'] = threads_str
            
            mcmc_config = temp_config.get('mcmc', {})
            chains = mcmc_config.get('chains', 4)
            total_threads = chains * threads_per_chain
            print(f"⚡ CPU並列設定: {threads_str} threads/chain × {chains} chains = {total_threads} vCPUs")
        else:
            print("ℹ️  threads_per_chain未設定: システムデフォルトを使用")
            
    except Exception as e:
        print(f"⚠️ CPU並列設定失敗: {e}")

# NumPyより先に環境変数を設定
_setup_cpu_threads()

# PyTensor設定（CPU専用）
os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float64'

# ============================================================================

print("="*70)
print("磁場・温度一括ベイズ推定プログラム")
print("="*70)
print("\n⏳ ライブラリを読み込み中... (初回は2-5分程度かかる場合があります)")
print("   Ctrl+Cで中断しないでください\n")

import time
_import_start = time.time()

import datetime
import warnings
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph.op import Op
from typing import List, Dict, Any, Tuple, Optional, Union
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
_import_time = time.time() - _import_start
print(f"\n✅ 全ライブラリの読み込み完了! (所要時間: {_import_time:.1f}秒)\n")

# 数値計算の警告を抑制
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(all='ignore')

try:
    import japanize_matplotlib
except ImportError:
    print("警告: japanize_matplotlib が見つかりません。")

plt.rcParams['figure.dpi'] = 120

# --- 物理定数 ---
kB = 1.380649e-23
muB = 9.274010e-24
hbar = 1.054571e-34
c = 299792458
mu0 = 4.0 * np.pi * 1e-7
s = 3.5  # スピン量子数 

# --- THz単位系変換定数 ---
# 数値計算の安定性向上のため、周波数・緩和率をTHz単位で扱う
# 1 THz = 10^12 Hz = 2π × 10^12 rad/s
THZ_TO_RAD_S = 2.0 * np.pi * 1e12  # THz → rad/s 変換係数
RAD_S_TO_THZ = 1.0 / THZ_TO_RAD_S  # rad/s → THz 変換係数

# --- 設定ファイル読み込み ---
def load_config(config_path: Optional[Union[str, pathlib.Path]] = None) -> Dict[str, Any]:
    """設定ファイル(YAML)を読み込み、デフォルト値とマージする"""
    if config_path is None:
        config_path = pathlib.Path(__file__).parent / "config_selected_datasets.yml"
    
    # デフォルト設定値（複数ファイル対応）
    default_config = {
        'file_paths': {
            'data_files': [
                {
                    'file': "C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\corrected_exp_datasets\\Corrected_Transmittance_Temperature.xlsx",
                    'sheet': "Corrected Data",
                    'type': "auto",
                    'description': "デフォルトデータファイル"
                }
            ],
            'results_parent_dir': "analysis_results_unified"
        },
        'execution': {},
        'physical_parameters': {
            'B_fixed': 9.0,
            'd_fixed': 157.8e-6,
            's': 3.5,
            'N_spin': 1.9386e+28,
            'initial_values': {
                'eps_bg': 14.20,
                'g_factor': 2.003147,
                'B4': 0.000576,
                'B6': 0.000050,
                'gamma': 0.11e12,
                'a_scale': 0.604971
            }
        },
        'analysis_settings': {
            'low_freq_cutoff': 0.361505,
            'high_freq_cutoff': 0.45,
            'weight_settings': {
                'peak_height_threshold': 0.05,
                'peak_prominence_threshold': 0.05,
                'peak_distance': 10,
                'lp_up_peak_weight': 1.0,
                'between_peaks_weight': 0.1,
                'high_freq_peak_weight': 1.0,
                'background_weight': 0.0
            }
        },
        'mcmc': {
            'draws': 4000,
            'tune': 2000,
            'chains': 4,
            'target_accept': 0.90,
            'init': "adapt_diag",
            'max_iterations': 2
            
        },
        'bayesian_priors': {
            'magnetic_parameters': {
                'a_scale': {'distribution': 'HalfNormal', 'sigma': 1.0},
                'g_factor': {'distribution': 'Normal', 'sigma': 0.1},
                'B4': {'distribution': 'Normal', 'sigma': 0.001},
                'B6': {'distribution': 'Normal', 'sigma': 0.0001}
            },
            'with_prior_info': {
                'a_scale': {'distribution': 'Normal', 'sigma': 0.2},
                'g_factor': {'distribution': 'Normal', 'sigma': 0.05},
                'B4': {'distribution': 'Normal', 'sigma': 0.0005},
                'B6': {'distribution': 'Normal', 'sigma': 0.0001}
            },
            'gamma_parameters': {
                'log_gamma_mu_base': {'distribution': 'Normal', 'sigma': 1.0},
                'log_gamma_sigma_base': {'distribution': 'HalfNormal', 'sigma': 0.8},
                'log_gamma_offset_base': {'distribution': 'Normal', 'mu': 0.0, 'sigma': 0.8, 'shape': 7},
                'temp_gamma_slope': {'distribution': 'Normal', 'mu': 0.0, 'sigma': 0.01}
            },
            'noise_parameters': {
                'sigma': {'distribution': 'HalfNormal', 'sigma': 0.05}
            }
        }
    }
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
        
        def merge_dict(default, user):
            for key, value in user.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    merge_dict(default[key], value)
                else:
                    default[key] = value
        
        merge_dict(default_config, user_config)
        
        # 数値パラメータの型変換を確実に実行
        try:
            initial_vals = default_config['physical_parameters']['initial_values']
            for key in ['eps_bg', 'g_factor', 'B4', 'B6', 'gamma', 'a_scale']:
                if key in initial_vals:
                    initial_vals[key] = float(initial_vals[key])
            
            phys_params = default_config['physical_parameters']
            for key in ['B_fixed', 'T_fixed', 'd_fixed', 's', 'N_spin']:
                if key in phys_params:
                    phys_params[key] = float(phys_params[key])
        except (KeyError, ValueError, TypeError) as e:
            print(f"⚠️ 警告: パラメータの型変換中にエラー: {e}")
        
        print(f"✅ 設定ファイル '{config_path}' を読み込みました。")
        
    except FileNotFoundError:
        print(f"⚠️ 設定ファイル '{config_path}' が見つかりません。デフォルト設定を使用します。")
    except Exception as e:
        print(f"⚠️ 設定ファイルの読み込みに失敗しました: {e}。デフォルト設定を使用します。")
    
    return default_config

# --- 結果保存ディレクトリの作成 ---
def create_results_directory(config: Dict[str, Any]) -> pathlib.Path:
    """実行日時を含む一意の結果保存ディレクトリを作成"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_parent = config['file_paths']['results_parent_dir']
    results_dir = pathlib.Path(__file__).parent / results_parent / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    config_backup_path = results_dir / "config_used.yml"
    with open(config_backup_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"📁 結果保存ディレクトリ: {results_dir.resolve()}")
    return results_dir

# --- 物理モデル関数群 (weighted_bayesian_fitting_completed.pyから移植) ---

def get_hamiltonian(B_ext_z: float, g_factor: float, B4: float, B6: float) -> np.ndarray:
    """ハミルトニアンを計算する"""
    n_states = int(2 * s + 1)
    m_values = np.arange(s, -s - 1, -1)
    Sz = np.diag(m_values)
    
    if n_states == 8:
        O40 = 60 * np.diag([7, -13, -3, 9, 9, -3, -13, 7])
        X_O44 = np.zeros((8, 8))
        X_O44[3, 7], X_O44[4, 0] = np.sqrt(35), np.sqrt(35)
        X_O44[2, 6], X_O44[5, 1] = 5 * np.sqrt(3), 5 * np.sqrt(3)
        O44 = 12 * (X_O44 + X_O44.T)
        O60 = 1260 * np.diag([1, -5, 9, -5, -5, 9, -5, 1])
        X_O64 = np.zeros((8, 8))
        X_O64[3, 7], X_O64[4, 0] = 3 * np.sqrt(35), 3 * np.sqrt(35)
        X_O64[2, 6], X_O64[5, 1] = -7 * np.sqrt(3), -7 * np.sqrt(3)
        O64 = 60 * (X_O64 + X_O64.T)
    else:
        raise ValueError(f"s={s}の結晶場演算子は未実装です")
    
    H_cf = (B4 * kB) * (O40 + 5 * O44) + (B6 * kB) * (O60 - 21 * O64)
    H_zee = g_factor * muB * B_ext_z * Sz
    return H_cf + H_zee

def normalize_gamma_array(gamma_input, target_length: int = 7) -> np.ndarray:
    """ガンマ配列の正規化と型安全性を確保"""
    if np.isscalar(gamma_input):
        return np.full(target_length, gamma_input)
    elif hasattr(gamma_input, 'ndim') and gamma_input.ndim == 0:
        return np.full(target_length, float(gamma_input))
    elif hasattr(gamma_input, '__len__'):
        gamma_array = np.array(gamma_input)
        if len(gamma_array) == target_length:
            return gamma_array
        elif len(gamma_array) > target_length:
            return gamma_array[:target_length]
        else:
            return np.pad(gamma_array, (0, target_length - len(gamma_array)), 'edge')
    else:
        return np.full(target_length, gamma_input.item())

def calculate_susceptibility(freq_thz_array: np.ndarray, H: np.ndarray, T: float, 
                             gamma_thz_array: np.ndarray) -> np.ndarray:
    """
    磁気感受率を計算する
    
    Parameters
    ----------
    freq_thz_array : np.ndarray
        周波数配列 [THz]
    H : np.ndarray
        ハミルトニアン行列
    T : float
        温度 [K]
    gamma_thz_array : np.ndarray
        緩和率配列 [THz]
    
    Returns
    -------
    np.ndarray
        磁気感受率（複素数）
    """
    gamma_thz_array = normalize_gamma_array(gamma_thz_array)
    
    eigenvalues, _ = np.linalg.eigh(H)
    eigenvalues -= np.min(eigenvalues)
    eigenvalues = np.clip(eigenvalues / (kB * T), -700, 700)
    
    Z = np.sum(np.exp(-eigenvalues))
    populations = np.exp(-eigenvalues) / Z
    delta_E = (eigenvalues[1:] - eigenvalues[:-1]) * kB * T
    delta_pop = populations[1:] - populations[:-1]
    
    valid_mask = np.isfinite(delta_E) & (np.abs(delta_E) > 1e-30)
    if not np.any(valid_mask):
        return np.zeros_like(freq_thz_array, dtype=complex)
    
    # 遷移周波数をTHz単位で計算
    omega_0_rad = delta_E / hbar  # rad/s
    freq_0_thz = omega_0_rad * RAD_S_TO_THZ  # THzに変換
    
    s_val = 3.5
    m_vals = np.arange(s_val, -s_val, -1)
    transition_strength = (s_val + m_vals) * (s_val - m_vals + 1)
    
    if len(gamma_thz_array) != len(delta_E):
        if len(gamma_thz_array) > len(delta_E):
            gamma_thz_array = gamma_thz_array[:len(delta_E)]
        else:
            gamma_thz_array = np.pad(gamma_thz_array, (0, len(delta_E) - len(gamma_thz_array)), 'edge')
    
    numerator = delta_pop * transition_strength
    finite_mask = np.isfinite(numerator) & np.isfinite(freq_0_thz) & np.isfinite(gamma_thz_array)
    numerator = numerator[finite_mask]
    freq_0_filtered = freq_0_thz[finite_mask]  # THz
    gamma_filtered = gamma_thz_array[finite_mask]  # THz
    
    if len(numerator) == 0:
        return np.zeros_like(freq_thz_array, dtype=complex)
    
    # THz単位で計算（数値的に安定）
    chi_array = np.zeros_like(freq_thz_array, dtype=complex)
    for i, freq_thz in enumerate(freq_thz_array):
        if not np.isfinite(freq_thz):
            continue
        # 分母: (f0 - f) - i*γ すべてTHz単位
        denominator = freq_0_filtered - freq_thz - 1j * gamma_filtered
        denominator[np.abs(denominator) < 1e-10] = 1e-10 + 1j * 1e-10
        chi_array[i] = np.sum(numerator / denominator)
    
    return -chi_array

def calculate_normalized_transmission(freq_thz_array: np.ndarray, mu_r_array: np.ndarray, 
                                     d: float, eps_bg: float) -> np.ndarray:
    """
    正規化透過率を計算する
    
    Parameters
    ----------
    freq_thz_array : np.ndarray
        周波数配列 [THz]
    mu_r_array : np.ndarray
        比透磁率配列
    d : float
        試料厚さ [m]
    eps_bg : float
        背景誘電率
    
    Returns
    -------
    np.ndarray
        正規化透過率
    """
    eps_bg = max(eps_bg, 0.1)
    d = max(d, 1e-6)
    
    # THz → rad/s に変換して波長を計算
    omega_array = freq_thz_array * THZ_TO_RAD_S
    
    mu_r_safe = np.where(np.isfinite(mu_r_array), mu_r_array, 1.0)
    eps_mu_product = eps_bg * mu_r_safe
    eps_mu_product = np.where(eps_mu_product.real > 0, eps_mu_product, 0.1 + 1j * eps_mu_product.imag)
    
    n_complex = np.sqrt(eps_mu_product + 0j)
    impe = np.sqrt(mu_r_safe / eps_bg + 0j)
    
    lambda_0 = np.full_like(omega_array, np.inf, dtype=float)
    nonzero_mask = omega_array > 1e-12
    lambda_0[nonzero_mask] = (2 * np.pi * c) / omega_array[nonzero_mask]
    
    delta = 2 * np.pi * n_complex * d / lambda_0
    delta = np.clip(delta.real, -700, 700) + 1j * np.clip(delta.imag, -700, 700)
    
    numerator = 4 * impe
    exp_pos = np.exp(-1j * delta)
    exp_neg = np.exp(1j * delta)
    
    denominator = (1 + impe)**2 * exp_pos - (1 - impe)**2 * exp_neg
    
    safe_mask = np.abs(denominator) > 1e-15
    t = np.zeros_like(denominator, dtype=complex)
    t[safe_mask] = numerator[safe_mask] / denominator[safe_mask]
    
    transmission = np.abs(t)**2
    transmission = np.where(np.isfinite(transmission), transmission, 0.0)
    transmission = np.clip(transmission, 0, 2)
    
    min_trans, max_trans = np.min(transmission), np.max(transmission)
    if max_trans > min_trans and np.isfinite(max_trans) and np.isfinite(min_trans):
        return (transmission - min_trans) / (max_trans - min_trans)
    else:
        return np.full_like(transmission, 0.5)

# --- 重み付け関数 ---
def create_frequency_weights(dataset: Dict[str, Any], analysis_settings: Dict[str, Any]) -> np.ndarray:
    """実験データのピーク特性に基づき、尤度関数のための重み配列を生成する"""
    weight_config = analysis_settings['weight_settings']
    high_freq_cutoff = analysis_settings['high_freq_cutoff']
    
    freq = dataset['frequency']
    trans = dataset['transmittance_full']
    
    peaks, properties = find_peaks(trans,
                                   height=weight_config['peak_height_threshold'],
                                   prominence=weight_config['peak_prominence_threshold'],
                                   distance=weight_config['peak_distance'])
    
    if len(peaks) < 2:
        weights = np.zeros_like(freq)
        low_freq_mask = freq < high_freq_cutoff
        weights[low_freq_mask] = 1.0
        return weights
    
    widths, _, left_ips, right_ips = peak_widths(trans, peaks, rel_height=0.5)
    left_freq = np.interp(left_ips, np.arange(len(freq)), freq)
    right_freq = np.interp(right_ips, np.arange(len(freq)), freq)
    
    weights = np.full_like(freq, weight_config['background_weight'])
    
    low_freq_peaks = peaks[freq[peaks] < high_freq_cutoff]
    if len(low_freq_peaks) >= 2:
        peak_prominences = properties['prominences'][freq[peaks] < high_freq_cutoff]
        sorted_indices = np.argsort(peak_prominences)[::-1]
        
        lp_idx_in_all_peaks = np.where(peaks == low_freq_peaks[sorted_indices[0]])[0][0]
        up_idx_in_all_peaks = np.where(peaks == low_freq_peaks[sorted_indices[1]])[0][0]
        
        lp_fwhm_right_freq = right_freq[lp_idx_in_all_peaks]
        up_fwhm_left_freq = left_freq[up_idx_in_all_peaks]
        
        lower_bound = np.minimum(lp_fwhm_right_freq, up_fwhm_left_freq)
        upper_bound = np.maximum(lp_fwhm_right_freq, up_fwhm_left_freq)
        between_mask = (freq >= lower_bound) & (freq <= upper_bound)
        weights[between_mask] = weight_config['between_peaks_weight']
        
        lp_fwhm_mask = (freq >= left_freq[lp_idx_in_all_peaks]) & (freq <= right_freq[lp_idx_in_all_peaks])
        up_fwhm_mask = (freq >= left_freq[up_idx_in_all_peaks]) & (freq <= right_freq[up_idx_in_all_peaks])
        weights[lp_fwhm_mask] = weight_config['lp_up_peak_weight']
        weights[up_fwhm_mask] = weight_config['lp_up_peak_weight']
    elif len(low_freq_peaks) == 1:
        # 【11/25追加】 ピークが1個しかない場合の処理
        target_peak = low_freq_peaks[0]
        idx_in_all_peaks = np.where(peaks == target_peak)[0][0]
        
        # その1個のピークの半値幅領域に重みを付ける
        fwhm_mask = (freq >= left_freq[idx_in_all_peaks]) & (freq <= right_freq[idx_in_all_peaks])
        weights[fwhm_mask] = weight_config['lp_up_peak_weight']
        
        print(f"  (Info) 低周波ピークが1つのみ検出されました: {freq[target_peak]:.3f} THz")
    high_freq_peak_indices = np.where(freq[peaks] >= high_freq_cutoff)[0]
    for idx_in_all_peaks in high_freq_peak_indices:
        fwhm_mask = (freq >= left_freq[idx_in_all_peaks]) & (freq <= right_freq[idx_in_all_peaks])
        weights[fwhm_mask] = weight_config['high_freq_peak_weight']
    
    print(f"  [B={dataset['b_field']:.1f}T, T={dataset['temperature']:.1f}K]: 重み配列を生成 (データ点数: {len(freq)})")
    return weights

# --- データ読み込み関数 ---
def load_unified_data(config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    複数のExcelファイルから温度変化データと磁場変化データを統一的に読み込む
    
    Parameters
    ----------
    config : Dict[str, Any]
        設定辞書（file_paths.data_filesに複数ファイル情報を含む）
    
    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        {
            'temp_variable': [データセット1, データセット2, ...],  # 温度変数、磁場固定
            'field_variable': [データセット1, データセット2, ...]  # 磁場変数、温度固定
        }
    """
    print("\n--- 統合データ読み込み (複数ファイル対応) ---")
    
    all_temp_datasets = []
    all_field_datasets = []
    
    # 旧形式(単一ファイル)との後方互換性
    if 'data_file' in config['file_paths']:
        print("⚠️ 旧形式の設定ファイルを検出しました。単一ファイルモードで動作します。")
        file_configs = [{
            'file': config['file_paths']['data_file'],
            'sheet': config['file_paths'].get('sheet_name', 'Corrected Data'),
            'description': '単一ファイル(互換モード)'
        }]
    else:
        file_configs = config['file_paths'].get('data_files', [])
    
    if not file_configs:
        raise ValueError("設定ファイルにdata_filesが定義されていません")
    
    freq_col = 'Frequency (THz)'
    B_fixed = config['physical_parameters']['B_fixed']
    T_fixed = config['physical_parameters'].get('T_fixed', 1.5)

    # 【追加】選別リストの取得
    selected_temps = config['file_paths'].get('selected_datasets', {}).get('temperatures', [])
    selected_fields = config['file_paths'].get('selected_datasets', {}).get('fields', [])
    
    # リストが空なら「全データ使用」とみなすフラグ
    use_filter_temp = len(selected_temps) > 0
    use_filter_field = len(selected_fields) > 0
    
    if use_filter_temp:
        print(f"🎯 温度データのフィルタリング有効: {selected_temps} K のみを解析します")
    if use_filter_field:
        print(f"🎯 磁場データのフィルタリング有効: {selected_fields} T のみを解析します")
    
    # 各ファイルを処理
    for file_idx, file_config in enumerate(file_configs, 1):
        file_path = file_config['file']
        sheet_name = file_config['sheet']
        description = file_config.get('description', '')
        
        print(f"\n📁 ファイル {file_idx}/{len(file_configs)}: {pathlib.Path(file_path).name}")
        if description:
            print(f"   説明: {description}")
        print(f"   シート: {sheet_name}")
        
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
        except Exception as e:
            print(f"   ❌ ファイル読み込みエラー: {e}")
            print(f"   スキップします。")
            continue
        
        df[freq_col] = pd.to_numeric(df[freq_col], errors='coerce')
        
        # 温度変化データの処理（'K'で終わる列を自動検出）
        temp_cols = [col for col in df.columns if col.endswith('K') and col != freq_col]
        
        if temp_cols:
            print(f"   📊 温度変化データ (B={B_fixed}T固定)")
            print(f"      検出された温度列: {temp_cols}")
            
            for col in temp_cols:
                try:
                    temp_value = float(col.replace('K', ''))
                    if use_filter_temp:
                        if temp_value not in selected_temps:
                            print(f"      ⏭️ T={temp_value}K は選別リストにないためスキップ")
                            continue
                    df_clean = df[[freq_col, col]].dropna()
                    freq, trans = df_clean[freq_col].values.astype(np.float64), df_clean[col].values.astype(np.float64)
                    
                    all_temp_datasets.append({
                        'temperature': temp_value,
                        'b_field': B_fixed,
                        'frequency': freq,  # THz単位
                        'transmittance_full': trans,
                        'pattern': 'temp_variable',
                        'source_file': pathlib.Path(file_path).name
                    })
                    print(f"      ✓ T={temp_value}K, B={B_fixed}T (データ点数: {len(freq)})")
                except ValueError:
                    print(f"      ⚠️ 列 '{col}' は温度データとして解釈できません。")
        
        # 磁場変化データの処理（'T'で終わる列を自動検出、温度列を除外）
        field_cols = [col for col in df.columns if col.endswith('T') and col != freq_col and col not in temp_cols]
        
        if field_cols:
            print(f"   📊 磁場変化データ (T={T_fixed}K固定)")
            print(f"      検出された磁場列: {field_cols}")
            
            for col in field_cols:
                try:
                    B_value = float(col.replace('T', ''))
                    if use_filter_field:
                        if B_value not in selected_fields:
                            print(f"      ⏭️ B={B_value}T は選別リストにないためスキップ")
                            continue
                    df_clean = df[[freq_col, col]].dropna()
                    freq, trans = df_clean[freq_col].values.astype(np.float64), df_clean[col].values.astype(np.float64)
                    
                    all_field_datasets.append({
                        'temperature': T_fixed,
                        'b_field': B_value,
                        'frequency': freq,  # THz単位
                        'transmittance_full': trans,
                        'pattern': 'field_variable',
                        'source_file': pathlib.Path(file_path).name
                    })
                    print(f"      ✓ T={T_fixed}K, B={B_value}T (データ点数: {len(freq)})")
                except ValueError:
                    print(f"      ⚠️ 列 '{col}' は磁場データとして解釈できません。")
    
    print(f"\n" + "="*70)
    print(f"✅ 全ファイルの読み込み完了:")
    print(f"  - 温度変化データ: {len(all_temp_datasets)} データセット")
    print(f"  - 磁場変化データ: {len(all_field_datasets)} データセット")
    print(f"="*70)
    
    return {
        'temp_variable': all_temp_datasets,
        'field_variable': all_field_datasets
    }

def split_data_by_frequency(datasets: List[Dict[str, Any]], 
                           low_cutoff: float, high_cutoff: float) -> Dict[str, List[Dict[str, Any]]]:
    """データを周波数帯域で分割"""
    low_freq_datasets = []
    high_freq_datasets = []
    
    for data in datasets:
        freq = data['frequency']
        trans = data['transmittance_full']
        
        # 高周波領域
        high_mask = freq >= high_cutoff
        if np.any(high_mask):
            min_high, max_high = trans[high_mask].min(), trans[high_mask].max()
            trans_norm_high = (trans[high_mask] - min_high) / (max_high - min_high) if max_high > min_high else np.full_like(trans[high_mask], 0.5)
            high_freq_datasets.append({
                'temperature': data['temperature'],
                'b_field': data['b_field'],
                'frequency': freq[high_mask],  # THz単位
                'transmittance': trans_norm_high,
                'pattern': data['pattern']
            })
    
    return {
        'high_freq': high_freq_datasets,
        'all_full': datasets
    }

# --- eps_bgフィッティング ---
def get_eps_bg_initial_values_and_bounds(temperature: float) -> Tuple[List[float], Tuple[float, float]]:
    """温度依存eps_bg初期値と境界値の取得"""
    eps_bg_init = 14.20
    if temperature <= 10:
        initial_eps_bg_values = [eps_bg_init * 0.85, eps_bg_init * 0.90, eps_bg_init * 0.95, eps_bg_init,
                                13.0, 12.5, 12.8, 13.2, 13.5, 14.0]
        bounds_eps_bg = (11.0, 16.0)
    elif temperature <= 100:
        initial_eps_bg_values = [eps_bg_init * 0.98, eps_bg_init, eps_bg_init * 1.02, eps_bg_init * 1.05,
                                13.8, 14.0, 14.2, 13.5, 14.5, 13.2]
        bounds_eps_bg = (11.5, 16.5)
    else:
        initial_eps_bg_values = [eps_bg_init * 1.05, eps_bg_init * 1.10, eps_bg_init * 1.15, eps_bg_init,
                                14.5, 15.0, 15.5, 14.0, 16.0, 13.8]
        bounds_eps_bg = (12.0, 17.0)
    return initial_eps_bg_values, bounds_eps_bg

def fit_eps_bg_unified(dataset: Dict[str, Any], 
                      bayesian_params: Optional[Dict[str, float]] = None,
                      config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """統合eps_bgフィッティング（磁場・温度両対応）"""
    if config is None:
        raise ValueError("config パラメータは必須です")
    
    T = dataset['temperature']
    B = dataset['b_field']
    d_fixed = config['physical_parameters']['d_fixed']
    N_spin = config['physical_parameters']['N_spin']
    
    print(f"\n--- eps_bgフィッティング [B={B:.1f}T, T={T:.1f}K] ---")
    
    # パラメータの設定
    g_factor: float
    B4: float
    B6: float
    a_scale: float
    
    if bayesian_params is not None:
        g_factor = bayesian_params.get('g_factor') or config['physical_parameters']['initial_values']['g_factor']
        B4 = bayesian_params.get('B4') or config['physical_parameters']['initial_values']['B4']
        B6 = bayesian_params.get('B6') or config['physical_parameters']['initial_values']['B6']
        a_scale = bayesian_params.get('a_scale') or config['physical_parameters']['initial_values']['a_scale']
        print(f"  🔄 ベイズ推定結果を使用")
    else:
        g_factor = config['physical_parameters']['initial_values']['g_factor']
        B4 = config['physical_parameters']['initial_values']['B4']
        B6 = config['physical_parameters']['initial_values']['B6']
        a_scale = config['physical_parameters']['initial_values']['a_scale']
        print(f"  🔰 初期値を使用")
    
    def model_func(freq_thz, eps_bg_fit):
        """eps_bgのみを変数とするモデル（THz単位系）"""
        try:
            H = get_hamiltonian(B, g_factor, B4, B6)
            # gammaもTHz単位で指定（0.11 THz ≈ 0.11e12 rad/s ÷ 2π×10^12）
            gamma_thz_array = np.full(7, 0.018)  # 約0.018 THz = 0.11e12 rad/s
            chi_raw = calculate_susceptibility(freq_thz, H, T, gamma_thz_array)
            
            G0 = a_scale * mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
            chi = G0 * chi_raw
            mu_r = 1 + chi  # H_form（必要に応じて変更）
            
            return calculate_normalized_transmission(freq_thz, mu_r, d_fixed, eps_bg_fit)
        except:
            return np.ones_like(freq_thz) * 0.5
    
    # 初期値と境界
    initial_eps_bg_values, bounds_eps_bg = get_eps_bg_initial_values_and_bounds(T)
    
    for attempt, initial_eps_bg in enumerate(initial_eps_bg_values):
        try:
            popt, _ = curve_fit(model_func, dataset['frequency'], dataset['transmittance'],
                               p0=[initial_eps_bg], bounds=([bounds_eps_bg[0]], [bounds_eps_bg[1]]),
                               maxfev=3000, method='trf')
            eps_bg_fit = popt[0]
            
            if bounds_eps_bg[0] <= eps_bg_fit <= bounds_eps_bg[1]:
                print(f"  ✅ 成功: eps_bg = {eps_bg_fit:.3f}")
                return {
                    'eps_bg': eps_bg_fit,
                    'd': d_fixed,
                    'temperature': T,
                    'b_field': B
                }
        except:
            continue
    
    print(f"  ❌ 全試行失敗、デフォルト値使用")
    return {'eps_bg': 14.20, 'd': d_fixed, 'temperature': T, 'b_field': B}

# --- PyMC Op クラス（統合版・THz単位系） ---
class UnifiedMagneticModelOp(Op):
    """
    磁場・温度両対応の統合PyMC Opクラス
    
    入力パラメータの単位:
    - a_scale: 無次元
    - gamma_concat: THz単位
    - g_factor: 無次元
    - B4, B6: K単位
    """
    def __init__(self, datasets: List[Dict[str, Any]], 
                 bt_specific_params: Dict[Tuple[float, float], Dict[str, float]], 
                 model_type: str):
        super().__init__()
        self.datasets = datasets
        self.bt_specific_params = bt_specific_params  # (B, T) -> {'eps_bg', 'd'}
        self.model_type = model_type
        
        # 全ての(B, T)ペアを取得してソート
        self.bt_pairs = sorted(list(set([(d['b_field'], d['temperature']) for d in datasets])))
        
        self.itypes = [pt.dscalar, pt.dvector, pt.dscalar, pt.dscalar, pt.dscalar]
        self.otypes = [pt.dvector]
    
    def perform(self, node, inputs, output_storage):
        a_scale, gamma_thz_concat, g_factor, B4, B6 = inputs
        full_predicted_y = []
        gamma_start_idx = 0
        
        for data in self.datasets:
            B = data['b_field']
            T = data['temperature']
            freq_thz = data['frequency']  # THz単位
            
            # (B, T)に対応するeps_bgとdを取得
            bt_key = (B, T)
            if bt_key in self.bt_specific_params:
                d_fixed = self.bt_specific_params[bt_key]['d']
                eps_bg_fixed = self.bt_specific_params[bt_key]['eps_bg']
            else:
                d_fixed = 157.8e-6
                eps_bg_fixed = 14.20
            
            # 温度依存gammaの取得（THz単位）
            gamma_end_idx = gamma_start_idx + 7
            gamma_thz_for_bt = gamma_thz_concat[gamma_start_idx:gamma_end_idx]
            gamma_start_idx = gamma_end_idx
            
            # 物理モデル計算（全てTHz単位）
            H = get_hamiltonian(B, g_factor, B4, B6)
            chi_raw = calculate_susceptibility(freq_thz, H, T, gamma_thz_for_bt)
            
            G0 = a_scale * mu0 * 1.9386e+28 * (g_factor * muB)**2 / (2 * hbar)
            chi = G0 * chi_raw
            
            if self.model_type == 'B_form':
                mu_r = 1 / (1 - chi)
            else:  # H_form
                mu_r = 1 + chi
            
            predicted_trans = calculate_normalized_transmission(freq_thz, mu_r, d_fixed, eps_bg_fixed)
            predicted_trans = np.where(np.isfinite(predicted_trans), predicted_trans, 0.5)
            predicted_trans = np.clip(predicted_trans, 0, 1)
            
            full_predicted_y.extend(predicted_trans)
        
        output_storage[0][0] = np.array(full_predicted_y)

# --- ベイズ推定関数 ---
def create_single_prior(name: str, config: Dict[str, Any], mu: Optional[float] = None) -> Any:
    """
    設定に基づいて単一の事前分布を作成する汎用関数
    
    Parameters
    ----------
    name : str
        パラメータ名
    config : Dict[str, Any]
        分布設定（distribution, mu, sigma, lower, upper等）
    mu : Optional[float]
        中心値（configにmuがない場合に使用）
    
    Returns
    -------
    PyMC分布オブジェクト
    """
    dist_type = config.get('distribution', 'Normal')
    sigma = config['sigma']
    
    # muの決定: config > 引数 > デフォルト0.0
    mu_value = config.get('mu', mu if mu is not None else 0.0)
    
    if dist_type == 'Normal':
        return pm.Normal(name, mu=mu_value, sigma=sigma)
    
    elif dist_type == 'HalfNormal':
        return pm.HalfNormal(name, sigma=sigma)
    
    elif dist_type == 'TruncatedNormal':
        # 正値制約付き正規分布（物理パラメータに推奨）
        lower = config.get('lower', 0.0)
        upper = config.get('upper', None)
        return pm.TruncatedNormal(name, mu=mu_value, sigma=sigma, lower=lower, upper=upper)
    
    elif dist_type == 'LogNormal':
        # 対数正規分布（正値のみ、スケールパラメータに適する）
        # mu, sigmaは対数スケールでの値として解釈
        return pm.LogNormal(name, mu=np.log(mu_value) if mu_value > 0 else 0.0, sigma=sigma)
    
    else:
        raise ValueError(f"未対応の分布タイプ: {dist_type}")


def create_prior_distributions(prior_config: Dict[str, Any], 
                              prior_magnetic_params: Optional[Dict[str, float]] = None,
                              initial_values: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    事前分布を作成（設定ファイルからの分布タイプ読み取りに対応）
    
    対応分布:
    - Normal: 通常の正規分布
    - HalfNormal: 半正規分布（正値のみ、μ=0）
    - TruncatedNormal: 切断正規分布（指定範囲に制約）
    - LogNormal: 対数正規分布（正値のみ）
    """
    priors = {}
    
    if prior_magnetic_params is None:
        # 初回実行時: initial_valuesを中心値として使用
        if initial_values is None:
            raise ValueError("initial_values パラメータは prior_magnetic_params が None の場合に必須です")
        mag_config = prior_config['magnetic_parameters']
        
        # a_scale: 正値のみ（TruncatedNormalまたはHalfNormal推奨）
        priors['a_scale'] = create_single_prior('a_scale', mag_config['a_scale'], 
                                                mu=initial_values.get('a_scale', 1.0))
        
        # g_factor: 正値のみ（TruncatedNormal推奨）
        priors['g_factor'] = create_single_prior('g_factor', mag_config['g_factor'],
                                                 mu=initial_values['g_factor'])
        
        # B4, B6: 正負両方あり得る（Normal）
        priors['B4'] = create_single_prior('B4', mag_config['B4'], mu=initial_values['B4'])
        priors['B6'] = create_single_prior('B6', mag_config['B6'], mu=initial_values['B6'])
    else:
        # 2回目以降: 前回の推定結果を中心値として使用
        prior_config_info = prior_config['with_prior_info']
        
        priors['a_scale'] = create_single_prior('a_scale', prior_config_info['a_scale'],
                                                mu=prior_magnetic_params['a_scale'])
        priors['g_factor'] = create_single_prior('g_factor', prior_config_info['g_factor'],
                                                 mu=prior_magnetic_params['g_factor'])
        priors['B4'] = create_single_prior('B4', prior_config_info['B4'],
                                          mu=prior_magnetic_params['B4'])
        priors['B6'] = create_single_prior('B6', prior_config_info['B6'],
                                          mu=prior_magnetic_params['B6'])
    
    return priors

def create_gamma_priors(gamma_config: Dict[str, Any], gamma_thz_init: float) -> Dict[str, Any]:
    """
    gamma事前分布を作成（THz単位）
    
    Parameters
    ----------
    gamma_config : Dict[str, Any]
        gamma関連の事前分布設定
    gamma_thz_init : float
        gamma初期値 [THz]
    
    Returns
    -------
    Dict[str, Any]
        gamma事前分布のPyMCオブジェクト
    """
    gamma_thz_init_float = float(gamma_thz_init)
    if gamma_thz_init_float <= 0:
        raise ValueError(f"gamma_thz_init must be positive, got {gamma_thz_init_float}")
    
    gamma_priors = {}
    # log(gamma)のベース値（THz単位での対数）
    gamma_priors['log_gamma_mu_base'] = pm.Normal('log_gamma_mu_base', 
                                                  mu=np.log(gamma_thz_init_float), 
                                                  sigma=gamma_config['log_gamma_mu_base']['sigma'])
    gamma_priors['log_gamma_sigma_base'] = pm.HalfNormal('log_gamma_sigma_base', 
                                                         sigma=gamma_config['log_gamma_sigma_base']['sigma'])
    gamma_priors['log_gamma_offset_base'] = pm.Normal('log_gamma_offset_base', 
                                                      mu=gamma_config['log_gamma_offset_base']['mu'], 
                                                      sigma=gamma_config['log_gamma_offset_base']['sigma'], 
                                                      shape=gamma_config['log_gamma_offset_base']['shape'])
    gamma_priors['temp_gamma_slope'] = pm.Normal('temp_gamma_slope', 
                                                 mu=gamma_config['temp_gamma_slope']['mu'], 
                                                 sigma=gamma_config['temp_gamma_slope']['sigma'])
    return gamma_priors

def run_unified_bayesian_fit(datasets: List[Dict[str, Any]], 
                             bt_specific_params: Dict[Tuple[float, float], Dict[str, float]],
                             weights_list: List[np.ndarray], 
                             results_dir: pathlib.Path,
                             config: Dict[str, Any],
                             prior_magnetic_params: Optional[Dict[str, float]] = None, 
                             model_type: str = 'H_form') -> Optional[az.InferenceData]:
    """統合ベイズ推定（磁場・温度一括処理）"""
    print(f"\n{'='*70}")
    print(f"統合ベイズ推定実行 (モデル: {model_type})")
    print(f"{'='*70}")
    print(f"データセット数: {len(datasets)}")
    print(f"(B, T)ペア数: {len(set([(d['b_field'], d['temperature']) for d in datasets]))}")
    
    combined_weights = np.concatenate(weights_list)
    
    with pm.Model() as model:
        prior_config = config['bayesian_priors']
        initial_values = config['physical_parameters']['initial_values']
        
        # 磁気パラメータ事前分布
        magnetic_priors = create_prior_distributions(prior_config, prior_magnetic_params, initial_values)
        a_scale = magnetic_priors['a_scale']
        g_factor = magnetic_priors['g_factor']
        B4 = magnetic_priors['B4']
        B6 = magnetic_priors['B6']
        
        # gamma事前分布（THz単位）
        # 設定ファイルのgamma初期値をTHz単位に変換
        gamma_init_raw = initial_values['gamma']
        # gamma_init_rawがrad/s単位の場合（>1e9）、THz単位に変換
        if gamma_init_raw > 1e9:
            gamma_thz_init = gamma_init_raw * RAD_S_TO_THZ
            print(f"  gamma初期値: {gamma_init_raw:.2e} rad/s → {gamma_thz_init:.4f} THz に変換")
        else:
            gamma_thz_init = gamma_init_raw
            print(f"  gamma初期値: {gamma_thz_init:.4f} THz (既にTHz単位)")
        
        gamma_priors = create_gamma_priors(prior_config['gamma_parameters'], 
                                          gamma_thz_init)
        log_gamma_mu_base = gamma_priors['log_gamma_mu_base']
        log_gamma_sigma_base = gamma_priors['log_gamma_sigma_base']
        log_gamma_offset_base = gamma_priors['log_gamma_offset_base']
        temp_gamma_slope = gamma_priors['temp_gamma_slope']
        
        # 全(B, T)ペアでのgamma計算（THz単位）
        bt_pairs = sorted(list(set([(d['b_field'], d['temperature']) for d in datasets])))
        gamma_thz_all_bt = []
        base_temp = 4.0
        
        for B, T in bt_pairs:
            temp_diff = T - base_temp
            temp_correction = temp_gamma_slope * temp_diff
            log_gamma_mu_temp = log_gamma_mu_base + temp_correction
            # gamma_bt はTHz単位
            gamma_thz_bt = pt.exp(log_gamma_mu_temp + log_gamma_offset_base * log_gamma_sigma_base)
            gamma_thz_all_bt.append(gamma_thz_bt)
        
        # データセットの順序に合わせてgammaを選択
        gamma_thz_final = []
        for dataset in datasets:
            bt_key = (dataset['b_field'], dataset['temperature'])
            bt_idx = bt_pairs.index(bt_key)
            gamma_thz_final.append(gamma_thz_all_bt[bt_idx])
        
        gamma_thz_concat = pt.concatenate(gamma_thz_final, axis=0)
        
        # 重み付きデータセット作成（frequency使用、omegaは不要）
        datasets_weighted = []
        weights_start_idx = 0
        for i, data in enumerate(datasets):
            n_points = len(data['transmittance_full'])
            weights_end_idx = weights_start_idx + n_points
            
            dataset_weights = combined_weights[weights_start_idx:weights_end_idx]
            dataset_valid_indices = np.where(dataset_weights > 0)[0]
            
            if len(dataset_valid_indices) > 0:
                weighted_dataset = {
                    'temperature': data['temperature'],
                    'b_field': data['b_field'],
                    'frequency': data['frequency'][dataset_valid_indices],  # THz単位
                    'transmittance_full': data['transmittance_full'][dataset_valid_indices],
                    'weights': dataset_weights[dataset_valid_indices],
                    'pattern': data['pattern']
                }
                datasets_weighted.append(weighted_dataset)
            
            weights_start_idx = weights_end_idx
        
        if not datasets_weighted:
            print("⚠️ 有効な重み付きデータポイントがありません。")
            return None
        
        # PyMC Op（gamma_thz_concatはTHz単位）
        op_weighted = UnifiedMagneticModelOp(datasets_weighted, bt_specific_params, model_type)
        mu = op_weighted(a_scale, gamma_thz_concat, g_factor, B4, B6)
        
        weights_tensor = pt.as_tensor_variable(np.concatenate([d['weights'] for d in datasets_weighted]))
        
        noise_config = prior_config['noise_parameters']['sigma']
        sigma = pm.HalfNormal('sigma', sigma=noise_config['sigma'])
        sigma_adjusted = sigma / pt.sqrt(weights_tensor)
        
        trans_target = np.concatenate([d['transmittance_full'] for d in datasets_weighted])
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma_adjusted, observed=trans_target)
        
        # サンプリング
        mcmc_config = config['mcmc']
        try:
            # 並列コア数の決定（設定ファイル > チェーン数 > 自動検出）
            n_cores = mcmc_config.get('cores', mcmc_config['chains'])
            if n_cores == 'auto':
                import multiprocessing
                n_cores = min(multiprocessing.cpu_count(), mcmc_config['chains'])
            
            sample_kwargs = {
                'draws': mcmc_config['draws'],
                'tune': mcmc_config['tune'],
                'chains': mcmc_config['chains'],
                'cores': n_cores,  # 並列実行コア数
                'target_accept': mcmc_config['target_accept'],
                'random_seed': mcmc_config.get('random_seed', None),
                'init': mcmc_config.get('init', 'auto'),
                'return_inferencedata': True,
                'progressbar': True,
                'idata_kwargs': {'log_likelihood': True}
            }
            
            print(f"⚡ 並列サンプリング: {mcmc_config['chains']}チェーン × {n_cores}コア")

            if 'nuts_sampler' in mcmc_config:
                sample_kwargs['nuts_sampler'] = mcmc_config['nuts_sampler']
                print(f"🚀 高速サンプラーを使用: {mcmc_config['nuts_sampler']}")
            
            trace = pm.sample(**sample_kwargs)
            print("✅ ベイズサンプリングが正常に完了しました。")
            
        except Exception as e:
            print(f"❌ ベイズサンプリングに失敗しました: {e}")
            return None
    
    # 結果保存
    trace_filename = results_dir / f'trace_{model_type}.nc'
    az.to_netcdf(trace, trace_filename)
    print(f"✅ Traceオブジェクトを保存: {trace_filename}")
    
    return trace

def extract_bayesian_parameters(trace: az.InferenceData) -> Dict[str, float]:
    """ベイズ推定結果からパラメータ抽出"""
    posterior = trace["posterior"]
    a_scale_mean = posterior['a_scale'].mean().item()
    g_factor_mean = posterior['g_factor'].mean().item()
    result = {
        'a_scale': a_scale_mean,
        'g_factor': g_factor_mean,
        'B4': posterior['B4'].mean().item(),
        'B6': posterior['B6'].mean().item(),
        'G0': a_scale_mean * mu0 * 1.9386e+28 * (g_factor_mean * muB)**2 / (2 * hbar)
    }
    
    try:
        result['log_gamma_mu_base'] = posterior['log_gamma_mu_base'].mean().item()
        result['temp_gamma_slope'] = posterior['temp_gamma_slope'].mean().item()
    except KeyError:
        pass
    
    return result

# --- 結果保存関数 ---
def save_unified_results(final_traces: Dict[str, az.InferenceData], 
                        bt_params: Dict[str, Dict[Tuple[float, float], Dict[str, float]]],
                        results_dir: pathlib.Path):
    """統合結果の保存"""
    print("\n--- 結果をCSVに保存中 ---")
    
    for model_type, trace in final_traces.items():
        params = extract_bayesian_parameters(trace)
        params_df = pd.DataFrame([params])
        params_file = results_dir / f'fitting_parameters_{model_type}.csv'
        params_df.to_csv(params_file, index=False)
        print(f"✅ 磁気パラメータを保存: {params_file}")
    
    for model_type, bt_specific_params in bt_params.items():
        bt_params_list = []
        for (B, T), params in sorted(bt_specific_params.items()):
            bt_params_list.append(params)
        
        if bt_params_list:
            bt_df = pd.DataFrame(bt_params_list)
            bt_file = results_dir / f'bt_optical_parameters_{model_type}.csv'
            bt_df.to_csv(bt_file, index=False)
            print(f"✅ {model_type}の(B,T)別光学パラメータを保存: {bt_file}")

# --- メインワークフロー ---
def main():
    """メイン実行関数"""
    print("="*70)
    print("磁場・温度一括ベイズ推定プログラム")
    print("="*70)
    
    # 設定読み込み
    config = load_config()
    
    # 乱数シード設定
    if 'random_seed' in config['mcmc']:
        RANDOM_SEED = config['mcmc']['random_seed']
        np.random.seed(RANDOM_SEED)
        print(f"🎲 乱数シード設定: {RANDOM_SEED}")
    
    # 結果ディレクトリ作成
    results_dir = create_results_directory(config)
    
    # データ読み込み
    print("\n" + "="*70)
    print("ステップ1: データ読み込み")
    print("="*70)
    
    unified_data = load_unified_data(config)
    
    # 全データセットを結合
    all_datasets = unified_data['temp_variable'] + unified_data['field_variable']
    
    if not all_datasets:
        print("❌ データが読み込めませんでした。終了します。")
        return
    
    # データを周波数帯域で分割
    split_data = split_data_by_frequency(
        all_datasets,
        config['analysis_settings']['low_freq_cutoff'],
        config['analysis_settings']['high_freq_cutoff']
    )
    
    high_freq_datasets = split_data['high_freq']
    all_datasets_full = split_data['all_full']
    
    # 初回eps_bgフィッティング
    print("\n" + "="*70)
    print("ステップ2: 初回eps_bgフィッティング")
    print("="*70)
    
    bt_specific_params = {}
    for dataset in high_freq_datasets:
        result = fit_eps_bg_unified(dataset, bayesian_params=None, config=config)
        bt_key = (result['b_field'], result['temperature'])
        bt_specific_params[bt_key] = result
    
    # 重み配列生成
    print("\n" + "="*70)
    print("ステップ3: 重み配列生成")
    print("="*70)
    
    weights_list = [create_frequency_weights(d, config['analysis_settings']) for d in all_datasets_full]
    
    # パラメータ更新ループ
    final_traces = {}
    model_bt_params = {
        'H_form': bt_specific_params.copy(),
        'B_form': bt_specific_params.copy()
    }
    max_iterations = config['mcmc']['max_iterations']
    
    for iteration in range(max_iterations):
        print(f"\n{'='*70}")
        print(f"反復 {iteration + 1}/{max_iterations}")
        print(f"{'='*70}")
        
        for model_type in ['H_form', 'B_form']:
            print(f"\n{model_type}モデルの処理")
            
            model_specific_prior = None
            if iteration > 0 and model_type in final_traces:
                model_specific_prior = extract_bayesian_parameters(final_traces[model_type])
                print(f"  📌 前回の{model_type}結果を事前分布として使用")
            
            trace = run_unified_bayesian_fit(
                all_datasets_full,
                model_bt_params[model_type],
                weights_list,
                results_dir,
                config,
                prior_magnetic_params=model_specific_prior,
                model_type=model_type
            )
            
            if trace:
                final_traces[model_type] = trace
                
                # eps_bg更新（最後の反復でない場合）
                if iteration < max_iterations - 1:
                    print(f"\n{model_type}のeps_bg更新中...")
                    bayesian_params = extract_bayesian_parameters(trace)
                    updated_bt_params = {}
                    
                    for dataset in high_freq_datasets:
                        result = fit_eps_bg_unified(dataset, bayesian_params=bayesian_params, config=config)
                        bt_key = (result['b_field'], result['temperature'])
                        updated_bt_params[bt_key] = result
                    
                    model_bt_params[model_type] = updated_bt_params
    
    # 結果保存
    if final_traces:
        print(f"\n{'='*70}")
        print("最終結果の保存")
        print(f"{'='*70}")
        save_unified_results(final_traces, model_bt_params, results_dir)
        
        print("\n🎉 全ての処理が完了しました")
        print(f"📁 結果は '{results_dir}' に保存されています")
    else:
        print("❌ ベイズ推定が失敗しました")

if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()
