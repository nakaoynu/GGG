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
# - 磁場変化データ: 各磁場列 (例: '1T', '2T', '3T', ...) ※実装予定

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.graph.op import Op
import os
import pathlib
import yaml
import datetime
import warnings
from typing import List, Dict, Any, Tuple, Optional, Union
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit

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

# --- 設定ファイル読み込み ---
def load_config(config_path: Optional[Union[str, pathlib.Path]] = None) -> Dict[str, Any]:
    """設定ファイル(YAML)を読み込み、デフォルト値とマージする"""
    if config_path is None:
        config_path = pathlib.Path(__file__).parent / "config_unified.yml"
    
    # デフォルト設定値（weighted_bayesian_fitting_completed.pyと同じ）
    default_config = {
        'file_paths': {
            'data_file': "C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\corrected_exp_datasets\\Corrected_Transmittance_Temperature.xlsx",
            'sheet_name': "Corrected Data",
            'results_parent_dir': "analysis_results_unified"
        },
        'execution': {'use_gpu': False},
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

def get_hamiltonian(B_ext_z: float, g_factor: float, B4: float, B6: float, s: float = 3.5) -> np.ndarray:
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

def calculate_susceptibility(omega_array: np.ndarray, H: np.ndarray, T: float, gamma_array: np.ndarray) -> np.ndarray:
    """磁気感受率を計算する"""
    gamma_array = normalize_gamma_array(gamma_array)
    
    eigenvalues, _ = np.linalg.eigh(H)
    eigenvalues -= np.min(eigenvalues)
    eigenvalues = np.clip(eigenvalues / (kB * T), -700, 700)
    
    Z = np.sum(np.exp(-eigenvalues))
    populations = np.exp(-eigenvalues) / Z
    delta_E = (eigenvalues[1:] - eigenvalues[:-1]) * kB * T
    delta_pop = populations[1:] - populations[:-1]
    
    valid_mask = np.isfinite(delta_E) & (np.abs(delta_E) > 1e-30)
    if not np.any(valid_mask):
        return np.zeros_like(omega_array, dtype=complex)
    
    omega_0 = delta_E / hbar
    s_val = 3.5
    m_vals = np.arange(s_val, -s_val, -1)
    transition_strength = (s_val + m_vals) * (s_val - m_vals + 1)
    
    if len(gamma_array) != len(delta_E):
        if len(gamma_array) > len(delta_E):
            gamma_array = gamma_array[:len(delta_E)]
        else:
            gamma_array = np.pad(gamma_array, (0, len(delta_E) - len(gamma_array)), 'edge')
    
    numerator = delta_pop * transition_strength
    finite_mask = np.isfinite(numerator) & np.isfinite(omega_0) & np.isfinite(gamma_array)
    numerator = numerator[finite_mask]
    omega_0_filtered = omega_0[finite_mask]
    gamma_filtered = gamma_array[finite_mask]
    
    if len(numerator) == 0:
        return np.zeros_like(omega_array, dtype=complex)
    
    chi_array = np.zeros_like(omega_array, dtype=complex)
    for i, omega in enumerate(omega_array):
        if not np.isfinite(omega):
            continue
        denominator = omega_0_filtered - omega - 1j * gamma_filtered
        denominator[np.abs(denominator) < 1e-20] = 1e-20 + 1j * 1e-20
        chi_array[i] = np.sum(numerator / denominator)
    
    return -chi_array

def calculate_normalized_transmission(omega_array: np.ndarray, mu_r_array: np.ndarray, 
                                     d: float, eps_bg: float) -> np.ndarray:
    """正規化透過率を計算する"""
    eps_bg = max(eps_bg, 0.1)
    d = max(d, 1e-6)
    
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
    
    high_freq_peak_indices = np.where(freq[peaks] >= high_freq_cutoff)[0]
    for idx_in_all_peaks in high_freq_peak_indices:
        fwhm_mask = (freq >= left_freq[idx_in_all_peaks]) & (freq <= right_freq[idx_in_all_peaks])
        weights[fwhm_mask] = weight_config['high_freq_peak_weight']
    
    print(f"  [B={dataset['b_field']:.1f}T, T={dataset['temperature']:.1f}K]: 重み配列を生成 (データ点数: {len(freq)})")
    return weights

# --- データ読み込み関数 ---
def load_unified_data(file_path: str, sheet_name: str, config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    温度変化データと磁場変化データを統一的に読み込む
    
    Returns:
        {
            'temp_variable': [データセット1, データセット2, ...],  # 温度変数、磁場固定
            'field_variable': [データセット1, データセット2, ...]  # 磁場変数、温度固定
        }
    """
    print("\n--- 統合データ読み込み ---")
    
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    except Exception as e:
        raise FileNotFoundError(f"Excelファイル '{file_path}' が読み込めません: {e}")
    
    freq_col = 'Frequency (THz)'
    df[freq_col] = pd.to_numeric(df[freq_col], errors='coerce')
    
    # 温度変化データ（磁場固定）
    temp_cols = config['analysis_settings']['temperature_columns']
    B_fixed = config['physical_parameters']['B_fixed']
    temp_variable_datasets = []
    
    print(f"\n📊 温度変化データ (磁場固定: B={B_fixed}T)")
    for col in temp_cols:
        if col not in df.columns:
            print(f"⚠️ 列 '{col}' が見つかりません。スキップします。")
            continue
        
        temp_value = float(col.replace('K', ''))
        df_clean = df[[freq_col, col]].dropna()
        freq, trans = df_clean[freq_col].values.astype(np.float64), df_clean[col].values.astype(np.float64)
        
        temp_variable_datasets.append({
            'temperature': temp_value,
            'b_field': B_fixed,
            'frequency': freq,
            'transmittance_full': trans,
            'omega': freq * 1e12 * 2 * np.pi,
            'pattern': 'temp_variable'
        })
        print(f"  ✓ T={temp_value}K, B={B_fixed}T (データ点数: {len(freq)})")
    
    # 磁場変化データ(温度固定) - 列名パターン自動検出: 'T'で終わる列(温度列を除く)
    field_cols = [col for col in df.columns if col.endswith('T') and col != freq_col and col not in temp_cols]
    T_fixed = config['physical_parameters'].get('T_fixed', 4.0)  # 設定ファイルから取得
    field_variable_datasets = []
    
    if field_cols:
        print(f"\n📊 磁場変化データ (温度固定: T={T_fixed}K)")
        print(f"  検出された磁場列: {field_cols}")
        for col in field_cols:
            try:
                B_value = float(col.replace('T', ''))
                df_clean = df[[freq_col, col]].dropna()
                freq, trans = df_clean[freq_col].values.astype(np.float64), df_clean[col].values.astype(np.float64)
                
                field_variable_datasets.append({
                    'temperature': T_fixed,
                    'b_field': B_value,
                    'frequency': freq,
                    'transmittance_full': trans,
                    'omega': freq * 1e12 * 2 * np.pi,
                    'pattern': 'field_variable'
                })
                print(f"  ✓ T={T_fixed}K, B={B_value}T (データ点数: {len(freq)})")
            except ValueError:
                print(f"⚠️ 列 '{col}' は磁場データとして解釈できません。スキップします。")
    else:
        print(f"\n⚠️ 磁場変化データが見つかりませんでした。温度変化データのみ処理します。")
    
    print(f"\n✅ データ読み込み完了:")
    print(f"  - 温度変化データ: {len(temp_variable_datasets)} データセット")
    print(f"  - 磁場変化データ: {len(field_variable_datasets)} データセット")
    
    return {
        'temp_variable': temp_variable_datasets,
        'field_variable': field_variable_datasets
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
                'frequency': freq[high_mask],
                'transmittance': trans_norm_high,
                'omega': data['omega'][high_mask],
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
                      config: Dict[str, Any] = None) -> Dict[str, float]:
    """統合eps_bgフィッティング（磁場・温度両対応）"""
    T = dataset['temperature']
    B = dataset['b_field']
    d_fixed = config['physical_parameters']['d_fixed']
    N_spin = config['physical_parameters']['N_spin']
    
    print(f"\n--- eps_bgフィッティング [B={B:.1f}T, T={T:.1f}K] ---")
    
    # パラメータの設定
    if bayesian_params is not None:
        g_factor = bayesian_params.get('g_factor', config['physical_parameters']['initial_values']['g_factor'])
        B4 = bayesian_params.get('B4', config['physical_parameters']['initial_values']['B4'])
        B6 = bayesian_params.get('B6', config['physical_parameters']['initial_values']['B6'])
        a_scale = bayesian_params.get('a_scale', config['physical_parameters']['initial_values']['a_scale'])
        print(f"  🔄 ベイズ推定結果を使用")
    else:
        g_factor = config['physical_parameters']['initial_values']['g_factor']
        B4 = config['physical_parameters']['initial_values']['B4']
        B6 = config['physical_parameters']['initial_values']['B6']
        a_scale = config['physical_parameters']['initial_values']['a_scale']
        print(f"  🔰 初期値を使用")
    
    def model_func(freq_thz, eps_bg_fit):
        """eps_bgのみを変数とするモデル"""
        try:
            omega = freq_thz * 1e12 * 2 * np.pi
            H = get_hamiltonian(B, g_factor, B4, B6)
            gamma_array = np.full(7, 0.11e12)  # 高周波領域では固定gamma
            chi_raw = calculate_susceptibility(omega, H, T, gamma_array)
            
            G0 = a_scale * mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
            chi = G0 * chi_raw
            mu_r = 1 + chi  # H_form（必要に応じて変更）
            
            return calculate_normalized_transmission(omega, mu_r, d_fixed, eps_bg_fit)
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

# --- PyMC Op クラス（統合版） ---
class UnifiedMagneticModelOp(Op):
    """磁場・温度両対応の統合PyMC Opクラス"""
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
        a_scale, gamma_concat, g_factor, B4, B6 = inputs
        full_predicted_y = []
        gamma_start_idx = 0
        
        for data in self.datasets:
            B = data['b_field']
            T = data['temperature']
            
            # (B, T)に対応するeps_bgとdを取得
            bt_key = (B, T)
            if bt_key in self.bt_specific_params:
                d_fixed = self.bt_specific_params[bt_key]['d']
                eps_bg_fixed = self.bt_specific_params[bt_key]['eps_bg']
            else:
                d_fixed = 157.8e-6
                eps_bg_fixed = 14.20
            
            # 温度依存gammaの取得
            gamma_end_idx = gamma_start_idx + 7
            gamma_for_bt = gamma_concat[gamma_start_idx:gamma_end_idx]
            gamma_start_idx = gamma_end_idx
            
            # 物理モデル計算
            H = get_hamiltonian(B, g_factor, B4, B6)
            chi_raw = calculate_susceptibility(data['omega'], H, T, gamma_for_bt)
            
            G0 = a_scale * mu0 * 1.9386e+28 * (g_factor * muB)**2 / (2 * hbar)
            chi = G0 * chi_raw
            
            if self.model_type == 'B_form':
                mu_r = 1 / (1 - chi)
            else:  # H_form
                mu_r = 1 + chi
            
            predicted_trans = calculate_normalized_transmission(data['omega'], mu_r, d_fixed, eps_bg_fixed)
            predicted_trans = np.where(np.isfinite(predicted_trans), predicted_trans, 0.5)
            predicted_trans = np.clip(predicted_trans, 0, 1)
            
            full_predicted_y.extend(predicted_trans)
        
        output_storage[0][0] = np.array(full_predicted_y)

# --- ベイズ推定関数 ---
def create_prior_distributions(prior_config: Dict[str, Any], 
                              prior_magnetic_params: Optional[Dict[str, float]] = None,
                              initial_values: Dict[str, float] = None) -> Dict[str, Any]:
    """事前分布を作成"""
    priors = {}
    
    if prior_magnetic_params is None:
        mag_config = prior_config['magnetic_parameters']
        priors['a_scale'] = pm.HalfNormal('a_scale', sigma=mag_config['a_scale']['sigma'])
        priors['g_factor'] = pm.Normal('g_factor', mu=initial_values['g_factor'], 
                                      sigma=mag_config['g_factor']['sigma'])
        priors['B4'] = pm.Normal('B4', mu=initial_values['B4'], sigma=mag_config['B4']['sigma'])
        priors['B6'] = pm.Normal('B6', mu=initial_values['B6'], sigma=mag_config['B6']['sigma'])
    else:
        prior_config_info = prior_config['with_prior_info']
        priors['a_scale'] = pm.Normal('a_scale', mu=prior_magnetic_params['a_scale'], 
                                     sigma=prior_config_info['a_scale']['sigma'])
        priors['g_factor'] = pm.Normal('g_factor', mu=prior_magnetic_params['g_factor'], 
                                      sigma=prior_config_info['g_factor']['sigma'])
        priors['B4'] = pm.Normal('B4', mu=prior_magnetic_params['B4'], 
                                sigma=prior_config_info['B4']['sigma'])
        priors['B6'] = pm.Normal('B6', mu=prior_magnetic_params['B6'], 
                                sigma=prior_config_info['B6']['sigma'])
    
    return priors

def create_gamma_priors(gamma_config: Dict[str, Any], gamma_init: float) -> Dict[str, Any]:
    """gamma事前分布を作成"""
    gamma_priors = {}
    gamma_priors['log_gamma_mu_base'] = pm.Normal('log_gamma_mu_base', 
                                                  mu=np.log(gamma_init), 
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
        
        # gamma事前分布
        gamma_priors = create_gamma_priors(prior_config['gamma_parameters'], 
                                          initial_values['gamma'])
        log_gamma_mu_base = gamma_priors['log_gamma_mu_base']
        log_gamma_sigma_base = gamma_priors['log_gamma_sigma_base']
        log_gamma_offset_base = gamma_priors['log_gamma_offset_base']
        temp_gamma_slope = gamma_priors['temp_gamma_slope']
        
        # 全(B, T)ペアでのgamma計算
        bt_pairs = sorted(list(set([(d['b_field'], d['temperature']) for d in datasets])))
        gamma_all_bt = []
        base_temp = 4.0
        
        for B, T in bt_pairs:
            temp_diff = T - base_temp
            temp_correction = temp_gamma_slope * temp_diff
            log_gamma_mu_temp = log_gamma_mu_base + temp_correction
            gamma_bt = pt.exp(log_gamma_mu_temp + log_gamma_offset_base * log_gamma_sigma_base)
            gamma_all_bt.append(gamma_bt)
        
        # データセットの順序に合わせてgammaを選択
        gamma_final = []
        for dataset in datasets:
            bt_key = (dataset['b_field'], dataset['temperature'])
            bt_idx = bt_pairs.index(bt_key)
            gamma_final.append(gamma_all_bt[bt_idx])
        
        gamma_concat = pt.concatenate(gamma_final, axis=0)
        
        # 重み付きデータセット作成
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
                    'frequency': data['frequency'][dataset_valid_indices],
                    'transmittance_full': data['transmittance_full'][dataset_valid_indices],
                    'omega': data['omega'][dataset_valid_indices],
                    'weights': dataset_weights[dataset_valid_indices],
                    'pattern': data['pattern']
                }
                datasets_weighted.append(weighted_dataset)
            
            weights_start_idx = weights_end_idx
        
        if not datasets_weighted:
            print("⚠️ 有効な重み付きデータポイントがありません。")
            return None
        
        # PyMC Op
        op_weighted = UnifiedMagneticModelOp(datasets_weighted, bt_specific_params, model_type)
        mu = op_weighted(a_scale, gamma_concat, g_factor, B4, B6)
        
        weights_tensor = pt.as_tensor_variable(np.concatenate([d['weights'] for d in datasets_weighted]))
        
        noise_config = prior_config['noise_parameters']['sigma']
        sigma = pm.HalfNormal('sigma', sigma=noise_config['sigma'])
        sigma_adjusted = sigma / pt.sqrt(weights_tensor)
        
        trans_target = np.concatenate([d['transmittance_full'] for d in datasets_weighted])
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma_adjusted, observed=trans_target)
        
        # サンプリング
        mcmc_config = config['mcmc']
        try:
            sample_kwargs = {
                'draws': mcmc_config['draws'],
                'tune': mcmc_config['tune'],
                'chains': mcmc_config['chains'],
                'target_accept': mcmc_config['target_accept'],
                'random_seed': mcmc_config.get('random_seed', None),
                'init': mcmc_config.get('init', 'auto'),
                'return_inferencedata': True,
                'progressbar': True,
                'idata_kwargs': {'log_likelihood': True}
            }
            
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
    
    unified_data = load_unified_data(
        file_path=config['file_paths']['data_file'],
        sheet_name=config['file_paths']['sheet_name'],
        config=config
    )
    
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
