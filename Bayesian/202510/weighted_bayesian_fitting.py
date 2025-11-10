# weighted_bayesian_fitting.py - 重み付き尤度関数を用いた温度依存ベイズ推定

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
import warnings
import yaml
import datetime
from typing import List, Dict, Any, Tuple, Optional
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit

# 数値計算の警告を抑制
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(all='ignore')  # NumPyの警告も抑制

# --- 設定ファイル読み込み機能 ---
def load_config(config_path: str = None) -> Dict[str, Any]:
    """設定ファイル(YAML)を読み込み、デフォルト値とマージする"""
    if config_path is None:
        config_path = pathlib.Path(__file__).parent / "config.yml"
    
    # デフォルト設定値
    default_config = {
        'file_paths': {
            'data_file': "C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\corrected_exp_datasets\\Corrected_Transmittance_Temperature.xlsx",
            'sheet_name': "Corrected Data",
            'results_parent_dir': "analysis_results"
        },
        'execution': {
            'use_gpu': True
        },
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
            'temperature_columns': ['4K', '30K', '100K', '300K'],
            'low_freq_cutoff': 0.361505,
            'high_freq_cutoff': 0.45
        },
        'mcmc': {
            'draws': 3000,
            'tune': 2000,
            'chains': 4,
            'target_accept': 0.92
        }
    }
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
        
        # 再帰的にデフォルト設定を更新
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
    
    # 使用した設定ファイルもコピー
    config_backup_path = results_dir / "config_used.yml"
    with open(config_backup_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"📁 結果保存ディレクトリ: {results_dir.resolve()}")
    return results_dir

# --- 0. 環境設定 ---
# このセクションはスクリプト実行時に最初に読み込まれます
print("--- 0. 環境設定を開始します ---")

# 設定ファイルの読み込み
CONFIG = load_config()

# ▼▼▼【変更点3】GPU利用設定 ▼▼▼
# 設定ファイルからGPU使用フラグを取得
USE_GPU = CONFIG['execution']['use_gpu']
if USE_GPU:
    try:
        # cupyの存在をチェックしてGPUが利用可能か簡易的に判断
        import cupy
        print("✅ GPU (CuPy) が利用可能です。GPU設定を試みます。")
        # PyTensorのデバイス設定を 'cuda' に変更
        os.environ['PYTENSOR_FLAGS'] = 'device=cuda,floatX=float64'
    except ImportError:
        print("⚠️ 警告: CuPyが見つかりません。GPUは使用できません。CPUにフォールバックします。")
        os.environ['PYTENSOR_FLAGS'] = 'optimizer=fast_compile,floatX=float64'
else:
    print("💻 CPU設定を使用します。")
    os.environ['PYTENSOR_FLAGS'] = 'optimizer=fast_compile,floatX=float64'

try:
    import japanize_matplotlib
except ImportError:
    print("警告: japanize_matplotlib が見つかりません。")

plt.rcParams['figure.dpi'] = 120

# 結果保存ディレクトリを作成
RESULTS_DIR = create_results_directory(CONFIG)
print(f"画像は '{RESULTS_DIR.resolve()}' に保存されます。")
# ▲▲▲【変更点3】GPU利用設定 ▲▲▲


# --- 1. 物理定数とシミュレーション条件 ---
# 設定ファイルから値を読み込み
print("--- 1. 物理定数と初期値を設定します ---")
kB = 1.380649e-23; muB = 9.274010e-24; hbar = 1.054571e-34
c = 299792458; mu0 = 4.0 * np.pi * 1e-7

# 設定ファイルから物理パラメータを取得
s = CONFIG['physical_parameters']['s']
N_spin = CONFIG['physical_parameters']['N_spin']
B_FIXED = CONFIG['physical_parameters']['B_fixed']
d_fixed = CONFIG['physical_parameters']['d_fixed']

# 初期値を設定ファイルから取得
initial_values = CONFIG['physical_parameters']['initial_values']
eps_bg_init = initial_values['eps_bg']
B4_init = initial_values['B4']
B6_init = initial_values['B6']
gamma_init = initial_values['gamma']
a_scale_init = initial_values['a_scale']
g_factor_init = initial_values['g_factor']

# ファイルパス設定
DATA_FILE_PATH = CONFIG['file_paths']['data_file']
DATA_SHEET_NAME = CONFIG['file_paths']['sheet_name']

# 解析設定
TEMPERATURE_COLUMNS = CONFIG['analysis_settings']['temperature_columns']
LOW_FREQUENCY_CUTOFF = CONFIG['analysis_settings']['low_freq_cutoff']
HIGH_FREQUENCY_CUTOFF = CONFIG['analysis_settings']['high_freq_cutoff']

# MCMCサンプリング設定
MCMC_CONFIG = CONFIG['mcmc']


# --- 物理モデル・データ処理関数 ---
# 元のtemperature_dependent_bayesian_fitting.pyから必要な関数をインポート

# キャッシュとユーティリティ関数
class HamiltonianCache:
    """ハミルトニアン計算のキャッシュクラス（パフォーマンス最適化）"""
    def __init__(self):
        self._cache = {}
    
    def get_hamiltonian_cached(self, B_ext_z: float, g_factor: float, B4: float, B6: float) -> np.ndarray:
        """キャッシュされたハミルトニアンを取得（同一パラメータでの重複計算を回避）"""
        key = (round(B_ext_z, 6), round(g_factor, 6), round(B4, 8), round(B6, 8))
        if key not in self._cache:
            self._cache[key] = get_hamiltonian(B_ext_z, g_factor, B4, B6)
        return self._cache[key]
    
    def clear_cache(self):
        """キャッシュをクリア"""
        self._cache.clear()

# グローバルキャッシュインスタンス
_hamiltonian_cache = HamiltonianCache()

def get_hamiltonian_cached(B_ext_z: float, g_factor: float, B4: float, B6: float) -> np.ndarray:
    """キャッシュされたハミルトニアン計算のパブリック関数"""
    return _hamiltonian_cache.get_hamiltonian_cached(B_ext_z, g_factor, B4, B6)

# 事前計算された固定ガンマ配列（メモリ効率最適化）
_FIXED_GAMMA_ARRAY = np.full(7, 0.11e12)

def normalize_gamma_array(gamma_input) -> np.ndarray:
    """ガンマ配列の正規化と型安全性を確保（コード重複削減）"""
    if np.isscalar(gamma_input):
        return np.full(7, gamma_input)
    elif hasattr(gamma_input, 'ndim') and gamma_input.ndim == 0:
        return np.full(7, float(gamma_input))
    elif hasattr(gamma_input, '__len__'):
        if len(gamma_input) == 7:
            return np.array(gamma_input)
        elif len(gamma_input) > 7:
            return np.array(gamma_input[:7])
        else:
            return np.pad(np.array(gamma_input), (0, 7 - len(gamma_input)), 'edge')
    else:
        return np.full(7, gamma_input.item())

def get_eps_bg_initial_values_and_bounds(temperature: float) -> Tuple[List[float], Tuple[float, float]]:
    """温度依存eps_bg初期値と境界値の取得（統合ユーティリティ関数）"""
    eps_bg_init = 14.20
    if temperature <= 10:
        # 低温では低めの初期値から開始（フォノンの凍結効果）
        initial_eps_bg_values = [eps_bg_init * 0.85, eps_bg_init * 0.90, eps_bg_init * 0.95, eps_bg_init,
                                13.0, 12.5, 12.8, 13.2, 13.5, 14.0]
        bounds_eps_bg = (11.0, 16.0)
    elif temperature <= 100:
        # 中間温度では標準的な初期値
        initial_eps_bg_values = [eps_bg_init * 0.98, eps_bg_init, eps_bg_init * 1.02, eps_bg_init * 1.05,
                                13.8, 14.0, 14.2, 13.5, 14.5, 13.2]
        bounds_eps_bg = (11.5, 16.5)
    else:
        # 高温では高めの初期値から開始（フォノンの活性化）
        initial_eps_bg_values = [eps_bg_init * 1.05, eps_bg_init * 1.10, eps_bg_init * 1.15, eps_bg_init,
                                14.5, 15.0, 15.5, 14.0, 16.0, 13.8]
        bounds_eps_bg = (12.0, 17.0)
    return initial_eps_bg_values, bounds_eps_bg

def get_hamiltonian(B_ext_z: float, g_factor: float, B4: float, B6: float) -> np.ndarray:
    """ハミルトニアンを計算する"""
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
    """磁気感受率を計算する（温度依存gamma対応・型安全版）"""
    
    # 統合されたガンマ配列正規化関数を使用（パフォーマンス最適化）
    gamma_array = normalize_gamma_array(gamma_array)

    eigenvalues, _ = np.linalg.eigh(H)
    eigenvalues -= np.min(eigenvalues)
    
    # 数値的安定性のためのクリッピング
    eigenvalues = np.clip(eigenvalues / (kB * T), -700, 700)
    
    Z = np.sum(np.exp(-eigenvalues))
    populations = np.exp(-eigenvalues) / Z
    delta_E = (eigenvalues[1:] - eigenvalues[:-1]) * kB * T  # 元の単位に戻す
    delta_pop = populations[1:] - populations[:-1]
    
    # 無効な値をチェック
    valid_mask = np.isfinite(delta_E) & (np.abs(delta_E) > 1e-30)
    if not np.any(valid_mask):
        return np.zeros_like(omega_array, dtype=complex)
    
    omega_0 = delta_E / hbar
    m_vals = np.arange(s, -s, -1)
    transition_strength = (s + m_vals) * (s - m_vals + 1)
    
    # === デバッグ出力: gamma_arrayがdelta_Eと同じ次元を持つように調整 ===
    if len(gamma_array) != len(delta_E):
        if len(gamma_array) > len(delta_E):
            gamma_array = gamma_array[:len(delta_E)]
        else:
            gamma_array = np.pad(gamma_array, (0, len(delta_E) - len(gamma_array)), 'edge')
    
    # 数値的安定性の向上
    numerator = delta_pop * transition_strength
    
    # 無効な値をフィルタリング
    finite_mask = np.isfinite(numerator) & np.isfinite(omega_0) & np.isfinite(gamma_array)
    numerator = numerator[finite_mask]
    omega_0_filtered = omega_0[finite_mask]
    gamma_filtered = gamma_array[finite_mask]
    
    if len(numerator) == 0:
        return np.zeros_like(omega_array, dtype=complex)
    
    # denominatorの計算を安全に実行
    chi_array = np.zeros_like(omega_array, dtype=complex)
    for i, omega in enumerate(omega_array):
        if not np.isfinite(omega):
            continue
        denominator = omega_0_filtered - omega - 1j * gamma_filtered
        # 非常に小さい値を避ける
        denominator[np.abs(denominator) < 1e-20] = 1e-20 + 1j * 1e-20
        chi_array[i] = np.sum(numerator / denominator)
    
    return -chi_array

def calculate_normalized_transmission(omega_array: np.ndarray, mu_r_array: np.ndarray, d: float, eps_bg: float) -> np.ndarray:
    """正規化透過率を計算する（改良版：数値安定性とピーク位置精度の向上）"""
    # 入力値の検証と安全な処理
    eps_bg = max(eps_bg, 0.1)  # 最小値を設定
    d = max(d, 1e-6)  # 最小値を設定
    
    # 複素屈折率と impedance の計算
    mu_r_safe = np.where(np.isfinite(mu_r_array), mu_r_array, 1.0)
    eps_mu_product = eps_bg * mu_r_safe
    eps_mu_product = np.where(eps_mu_product.real > 0, eps_mu_product, 0.1 + 1j * eps_mu_product.imag)
    
    n_complex = np.sqrt(eps_mu_product + 0j)
    impe = np.sqrt(mu_r_safe / eps_bg + 0j)
    
    # 波長計算（ゼロ周波数を避ける）
    lambda_0 = np.full_like(omega_array, np.inf, dtype=float)
    nonzero_mask = omega_array > 1e-12
    lambda_0[nonzero_mask] = (2 * np.pi * c) / omega_array[nonzero_mask]
    
    # 位相計算（オーバーフロー防止）
    delta = 2 * np.pi * n_complex * d / lambda_0
    delta = np.clip(delta.real, -700, 700) + 1j * np.clip(delta.imag, -700, 700)
    
    # 透過率計算
    numerator = 4 * impe
    exp_pos = np.exp(-1j * delta)
    exp_neg = np.exp(1j * delta)
    
    denominator = (1 + impe)**2 * exp_pos - (1 - impe)**2 * exp_neg
    
    # 分母がゼロに近い場合の処理
    safe_mask = np.abs(denominator) > 1e-15
    t = np.zeros_like(denominator, dtype=complex)
    t[safe_mask] = numerator[safe_mask] / denominator[safe_mask]
    
    transmission = np.abs(t)**2
    
    # 数値安定性のため、異常値を除去
    transmission = np.where(np.isfinite(transmission), transmission, 0.0)
    transmission = np.clip(transmission, 0, 2)  # 物理的に意味のある範囲に制限
    
    # 正規化
    min_trans, max_trans = np.min(transmission), np.max(transmission)
    if max_trans > min_trans and np.isfinite(max_trans) and np.isfinite(min_trans):
        return (transmission - min_trans) / (max_trans - min_trans)
    else:
        return np.full_like(transmission, 0.5)

def load_data_full_range_temperature(file_path: str, sheet_name: str) -> List[Dict[str, Any]]:
    """全周波数範囲の温度依存データを読み込む"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    except Exception as e:
        raise FileNotFoundError(f"Excelファイル '{file_path}' が読み込めません: {e}")
    
    freq_col = 'Frequency (THz)'
    df[freq_col] = pd.to_numeric(df[freq_col], errors='coerce')
    temp_cols = TEMPERATURE_COLUMNS
    
    all_datasets_full = []
    
    for col in temp_cols:
        if col not in df.columns:
            print(f"警告: 列 '{col}' が見つかりません。スキップします。")
            continue
            
        temp_value = float(col.replace('K', ''))
        df_clean = df[[freq_col, col]].dropna()
        freq, trans = df_clean[freq_col].values.astype(np.float64), df_clean[col].values.astype(np.float64)
        
        # 全範囲データ
        all_datasets_full.append({
            'temperature': temp_value, 
            'b_field': B_FIXED, 
            'frequency': freq, 
            'transmittance_full': trans, 
            'omega': freq * 1e12 * 2 * np.pi
        })
    
    print(f"全範囲温度依存データ読み込み完了: {len(all_datasets_full)} データセット")
    return all_datasets_full

def load_and_split_data_three_regions_temperature(file_path: str, sheet_name: str, 
                                                 low_cutoff: float = LOW_FREQUENCY_CUTOFF, 
                                                 high_cutoff: float = HIGH_FREQUENCY_CUTOFF) -> Dict[str, Any]:
    """温度依存データを1回だけ読み込み、すべての形式で提供する統一関数"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    except Exception as e:
        raise FileNotFoundError(f"Excelファイル '{file_path}' が読み込めません: {e}")
    
    freq_col = 'Frequency (THz)'
    df[freq_col] = pd.to_numeric(df[freq_col], errors='coerce')
    temp_cols = TEMPERATURE_COLUMNS
    
    # 全てのデータ形式を一度に作成
    low_freq_datasets, mid_freq_datasets, high_freq_datasets = [], [], []
    all_datasets_full = []
    
    for col in temp_cols:
        if col not in df.columns:
            print(f"警告: 列 '{col}' が見つかりません。スキップします。")
            continue
            
        temp_value = float(col.replace('K', ''))
        df_clean = df[[freq_col, col]].dropna()
        freq, trans = df_clean[freq_col].values.astype(np.float64), df_clean[col].values.astype(np.float64)
        
        # 3つの領域にマスクを定義
        low_mask = freq <= low_cutoff
        mid_mask = (freq > low_cutoff) & (freq < high_cutoff)
        high_mask = freq >= high_cutoff
        
        base_data = {'temperature': temp_value, 'b_field': B_FIXED}
        
        # 低周波領域
        if np.any(low_mask):
            min_low, max_low = trans[low_mask].min(), trans[low_mask].max()
            trans_norm_low = (trans[low_mask] - min_low) / (max_low - min_low) if max_low > min_low else np.full_like(trans[low_mask], 0.5)
            low_freq_datasets.append({**base_data, 'frequency': freq[low_mask], 'transmittance': trans_norm_low, 'omega': freq[low_mask] * 1e12 * 2 * np.pi})
        
        # 中間領域
        if np.any(mid_mask):
            min_mid, max_mid = trans[mid_mask].min(), trans[mid_mask].max()
            trans_norm_mid = (trans[mid_mask] - min_mid) / (max_mid - min_mid) if max_mid > min_mid else np.full_like(trans[mid_mask], 0.5)
            mid_freq_datasets.append({**base_data, 'frequency': freq[mid_mask], 'transmittance': trans_norm_mid, 'omega': freq[mid_mask] * 1e12 * 2 * np.pi})
        
        # 高周波領域
        if np.any(high_mask):
            min_high, max_high = trans[high_mask].min(), trans[high_mask].max()
            trans_norm_high = (trans[high_mask] - min_high) / (max_high - min_high) if max_high > min_high else np.full_like(trans[high_mask], 0.5)
            high_freq_datasets.append({**base_data, 'frequency': freq[high_mask], 'transmittance': trans_norm_high, 'omega': freq[high_mask] * 1e12 * 2 * np.pi})
        
        # 全範囲データ
        all_datasets_full.append({**base_data, 'frequency': freq, 'transmittance_full': trans, 'omega': freq * 1e12 * 2 * np.pi})
    
    print(f"温度依存データ読み込み完了:")
    print(f"  低周波領域 [~, {low_cutoff}THz]: {len(low_freq_datasets)} データセット")
    print(f"  中間領域 [{low_cutoff}THz, {high_cutoff}THz]: {len(mid_freq_datasets)} データセット")
    print(f"  高周波領域 [{high_cutoff}THz, ~]: {len(high_freq_datasets)} データセット")
    print(f"  全範囲データ: {len(all_datasets_full)} データセット")
    
    return {
        'low_freq': low_freq_datasets, 
        'mid_freq': mid_freq_datasets,
        'high_freq': high_freq_datasets,
        'all_full': all_datasets_full
    }

def fit_eps_bg_only_temperature(dataset: Dict[str, Any], 
                               fixed_params: Optional[Dict[str, Any]] = None,
                               G0_from_bayesian: Optional[float] = None) -> Dict[str, float]:
    """各温度で高周波データからeps_bgのみをフィッティングする（他パラメータは固定）"""
    print(f"\n--- 温度 {dataset['temperature']} K の高周波eps_bgフィッティング ---")
    
    # 固定パラメータの取得
    if fixed_params is None:
        fixed_params = {
            'd': d_fixed,
            'g_factor': g_factor_init,
            'B4': B4_init,
            'B6': B6_init,
            'gamma_fixed': 0.11e12  # 高周波領域では単一の固定値を使用
        }
    
    def magnetic_cavity_model_eps_bg_only(freq_thz, eps_bg_fit):
        """eps_bgのみを変数とする高周波透過率モデル（γは物理的に妥当な単一値で固定）"""
        try:
            omega = freq_thz * 1e12 * 2 * np.pi
            
            # 固定パラメータから値を取得
            g_factor_fit = fixed_params['g_factor']
            B4_fit = fixed_params['B4']
            B6_fit = fixed_params['B6']
            gamma_fixed = fixed_params['gamma_fixed']  # 単一の固定γ値
            
            # ハミルトニアンと磁気感受率の計算（キャッシュ使用）
            H = get_hamiltonian_cached(B_FIXED, g_factor_fit, B4_fit, B6_fit)
            
            # 高周波領域では単一のγ値を7要素に複製（メモリ節約版）
            gamma_array = _FIXED_GAMMA_ARRAY  # 事前計算された配列を使用
            chi_raw = calculate_susceptibility(omega, H, dataset['temperature'], gamma_array)
            
            # 磁気感受率のスケーリング
            G0 = mu0 * N_spin * (g_factor_fit * muB)**2 / (2 * hbar)
            chi = G0 * chi_raw
            
            # H_formで透磁率を計算
            mu_r = 1 + chi
            
            # d_fixedを直接使用
            return calculate_normalized_transmission(omega, mu_r, d_fixed, eps_bg_fit)
        except Exception as e:
            print(f"    警告: モデル計算エラー {e}")
            return np.ones_like(freq_thz) * 0.5

    # 複数の初期値を試行
    success = False
    result = {}
    
    # 温度依存の初期値と境界値を取得（統合関数使用）
    initial_eps_bg_values, bounds_eps_bg = get_eps_bg_initial_values_and_bounds(dataset['temperature'])
    
    for attempt, initial_eps_bg in enumerate(initial_eps_bg_values):
        try:
            print(f"  試行 {attempt + 1}: eps_bg初期値 = {initial_eps_bg:.3f}")
            
            popt, pcov = curve_fit(
                magnetic_cavity_model_eps_bg_only,
                dataset['frequency'],
                dataset['transmittance'],
                p0=[initial_eps_bg],
                bounds=([bounds_eps_bg[0]], [bounds_eps_bg[1]]),
                maxfev=3000,
                method='trf'
            )
            
            eps_bg_fit = popt[0]
            
            # パラメータが物理的に妥当かチェック
            if bounds_eps_bg[0] <= eps_bg_fit <= bounds_eps_bg[1]:
                print(f"  成功 (試行 {attempt + 1}): eps_bg = {eps_bg_fit:.3f}")
                result = {
                    'eps_bg': eps_bg_fit,
                    'd': d_fixed,  # 固定値を直接使用
                    'temperature': dataset['temperature']
                }
                success = True
                break
            else:
                print(f"  失敗 (試行 {attempt + 1}): eps_bg = {eps_bg_fit:.3f} は範囲外")
                
        except RuntimeError as e:
            print(f"  失敗 (試行 {attempt + 1}): 最適化エラー - {e}")
        except Exception as e:
            print(f"  失敗 (試行 {attempt + 1}): その他のエラー - {e}")
    
    if not success:
        print("  ❌ 全ての試行に失敗")
        result = {}
    
    return result


# ▼▼▼【変更点1】尤度重み付け関数の追加 ▼▼▼
def create_frequency_weights(dataset: Dict[str, Any]) -> np.ndarray:
    """
    実験データのピーク特性に基づき、尤度関数のための重み配列を生成する。
    - LP, UP, 高周波モードの半値幅領域: 重み 1.0
    - LP と UP の間の領域: 重み 0.1
    - その他の領域: 重み 0.0 (フィッティングから除外)
    """
    freq = dataset['frequency']
    trans = dataset['transmittance_full']
    
    # 吸収スペクトルに変換してピークを検出しやすくする
    absorption = 1 - (trans / np.max(trans))
    
    # ピーク検出
    peaks, properties = find_peaks(absorption, height=0.05, prominence=0.05, distance=10)
    
    if len(peaks) < 2:
        # ピークが少なすぎる場合は、低周波領域全体に均一な重みを付ける
        weights = np.zeros_like(freq)
        low_freq_mask = freq < HIGH_FREQUENCY_CUTOFF
        weights[low_freq_mask] = 1.0
        return weights

    # 半値幅を計算
    widths, _, left_ips, right_ips = peak_widths(absorption, peaks, rel_height=0.5)
    
    # 周波数単位に変換
    left_freq = np.interp(left_ips, np.arange(len(freq)), freq)
    right_freq = np.interp(right_ips, np.arange(len(freq)), freq)

    # 重み配列の初期化 (デフォルトは0)
    weights = np.zeros_like(freq)

    # LPとUPを特定 (低周波側の2つの主要なピークと仮定)
    low_freq_peaks = peaks[freq[peaks] < HIGH_FREQUENCY_CUTOFF]
    if len(low_freq_peaks) >= 2:
        # プロミネンスでソートして上位2つを取得
        peak_prominences = properties['prominences'][freq[peaks] < HIGH_FREQUENCY_CUTOFF]
        sorted_indices = np.argsort(peak_prominences)[::-1]
        
        lp_idx_in_all_peaks = np.where(peaks == low_freq_peaks[sorted_indices[1]])[0][0]
        up_idx_in_all_peaks = np.where(peaks == low_freq_peaks[sorted_indices[0]])[0][0]
        
        lp_freq_peak = freq[peaks[lp_idx_in_all_peaks]]
        up_freq_peak = freq[peaks[up_idx_in_all_peaks]]
        
        # LP-UP間に重み0.1を付与
        between_mask = (freq >= lp_freq_peak) & (freq <= up_freq_peak)
        weights[between_mask] = 0.1
        
        # LPとUPの半値幅領域に重み1.0を付与
        lp_fwhm_mask = (freq >= left_freq[lp_idx_in_all_peaks]) & (freq <= right_freq[lp_idx_in_all_peaks])
        up_fwhm_mask = (freq >= left_freq[up_idx_in_all_peaks]) & (freq <= right_freq[up_idx_in_all_peaks])
        weights[lp_fwhm_mask] = 1.0
        weights[up_fwhm_mask] = 1.0
        
    # 高周波の共振器モードにも重み1.0を付与
    high_freq_peak_indices = np.where(freq[peaks] >= HIGH_FREQUENCY_CUTOFF)[0]
    for idx_in_all_peaks in high_freq_peak_indices:
        fwhm_mask = (freq >= left_freq[idx_in_all_peaks]) & (freq <= right_freq[idx_in_all_peaks])
        weights[fwhm_mask] = 1.0

    print(f"  温度 {dataset['temperature']}K: 重み配列を生成。重み>0のデータ点数: {np.sum(weights > 0)} / {len(freq)}")
    return weights
# ▲▲▲【変更点1】尤度重み付け関数の追加 ▲▲▲


class TemperatureMagneticModelOp(Op):
    """温度依存の低周波領域の磁気パラメータを推定するためのPyMC Op（温度依存gamma対応）。"""
    def __init__(self, datasets: List[Dict[str, Any]], temperature_specific_params: Dict[float, Dict[str, float]], model_type: str):
        super().__init__()
        self.datasets = datasets
        self.temperature_specific_params = temperature_specific_params
        self.model_type = model_type
        self.temp_list = sorted(list(set([d['temperature'] for d in datasets])))
        # 温度依存gammaに対応するためinputタイプを拡張
        self.itypes = [pt.dscalar, pt.dvector, pt.dscalar, pt.dscalar, pt.dscalar]  # a_scale, gamma_concat, g_factor, B4, B6
        self.otypes = [pt.dvector]
    
    def perform(self, node, inputs, output_storage):
        a_scale, gamma_concat, g_factor, B4, B6 = inputs
        full_predicted_y = []
        gamma_start_idx = 0
        
        for data in self.datasets:
            # 該当する温度の固定パラメータを取得
            temperature = data['temperature']
            if temperature in self.temperature_specific_params:
                d_fixed = self.temperature_specific_params[temperature]['d']
                eps_bg_fixed = self.temperature_specific_params[temperature]['eps_bg']
            else:
                # フォールバック
                d_fixed = globals()['d_fixed']
                eps_bg_fixed = eps_bg_init
            
            # 温度依存gammaの取得（7個ずつ分割）
            gamma_end_idx = gamma_start_idx + 7
            gamma_for_temp = gamma_concat[gamma_start_idx:gamma_end_idx]
            gamma_start_idx = gamma_end_idx
            
            # 物理モデル計算
            H = get_hamiltonian(B_FIXED, g_factor, B4, B6)
            chi_raw = calculate_susceptibility(data['omega'], H, temperature, gamma_for_temp)
            
            # スケーリング係数の計算
            G0 = a_scale * mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
            chi = G0 * chi_raw
            
            # モデル形式に応じた透磁率計算
            if self.model_type == 'B_form':
                mu_r = 1 / (1 - chi)
            else:  # H_form (デフォルト)
                mu_r = 1 + chi
            
            # 透過率計算
            predicted_trans = calculate_normalized_transmission(data['omega'], mu_r, d_fixed, eps_bg_fixed)
            
            # 数値的安定性のチェック
            predicted_trans = np.where(np.isfinite(predicted_trans), predicted_trans, 0.5)
            predicted_trans = np.clip(predicted_trans, 0, 1)
            
            full_predicted_y.extend(predicted_trans)
        
        output_storage[0][0] = np.array(full_predicted_y)

def extract_bayesian_parameters(trace: az.InferenceData) -> Dict[str, float]:
    """ベイズ推定結果から平均パラメータを抽出（温度依存gamma対応）"""
    posterior = trace["posterior"]
    a_scale_mean = posterior['a_scale'].mean().item()
    g_factor_mean = posterior['g_factor'].mean().item()
    result = {
        'a_scale': a_scale_mean,
        'g_factor': g_factor_mean,
        'B4': posterior['B4'].mean().item(),
        'B6': posterior['B6'].mean().item(),
        'G0': a_scale_mean * mu0 * N_spin * (g_factor_mean * muB)**2 / (2 * hbar)
    }
    
    # 温度依存gammaパラメータも追加
    try:
        result['log_gamma_mu_base'] = posterior['log_gamma_mu_base'].mean().item()
        result['temp_gamma_slope'] = posterior['temp_gamma_slope'].mean().item()
        result['temp_gamma_nonlinear'] = posterior['temp_gamma_nonlinear'].mean().item()
    except KeyError:
        # 温度依存gammaがない場合はスキップ
        pass
    
    return result


def run_temperature_bayesian_fit(datasets: List[Dict[str, Any]], 
                                temperature_specific_params: Dict[float, Dict[str, float]],
                                # ▼▼▼【変更点1】重み配列を引数に追加 ▼▼▼
                                weights_list: List[np.ndarray], 
                                # ▲▲▲【変更点1】重み配列を引数に追加 ▲▲▲
                                prior_magnetic_params: Optional[Dict[str, float]] = None, 
                                model_type: str = 'H_form') -> Optional[az.InferenceData]:
    print(f"\n--- 重み付きベイズ推定 (モデル: {model_type}) ---")
    
    # 観測データと重みを結合
    trans_observed = np.concatenate([d['transmittance_full'] for d in datasets])
    combined_weights = np.concatenate(weights_list)

    with pm.Model() as model:
        # 事前分布の設定（設定ファイルから初期値を取得）
        if prior_magnetic_params is None:
            # 初回実行時は設定ファイルの初期値を使用
            a_scale = pm.Normal('a_scale', mu=a_scale_init, sigma=0.3)
            g_factor = pm.Normal('g_factor', mu=g_factor_init, sigma=0.2)
            B4 = pm.Normal('B4', mu=B4_init, sigma=0.0002)
            B6 = pm.Normal('B6', mu=B6_init, sigma=0.00002)
        else:
            # 事前情報がある場合はそれを使用
            a_scale = pm.Normal('a_scale', mu=prior_magnetic_params['a_scale'], sigma=0.2)
            g_factor = pm.Normal('g_factor', mu=prior_magnetic_params['g_factor'], sigma=0.1)
            B4 = pm.Normal('B4', mu=prior_magnetic_params['B4'], sigma=0.0001)
            B6 = pm.Normal('B6', mu=prior_magnetic_params['B6'], sigma=0.00001)
        
        # 温度依存gammaパラメータ
        log_gamma_mu_base = pm.Normal('log_gamma_mu_base', mu=np.log(gamma_init), sigma=1.0)
        temp_gamma_slope = pm.Normal('temp_gamma_slope', mu=0.0, sigma=0.01)
        temp_gamma_nonlinear = pm.Normal('temp_gamma_nonlinear', mu=0.0, sigma=0.001)
        log_gamma_sigma_base = pm.HalfNormal('log_gamma_sigma_base', sigma=1.0)
        log_gamma_offset_base = pm.Normal('log_gamma_offset_base', mu=0.0, sigma=1.0)
        
        # 各温度の各遷移のgammaを生成
        gamma_list = []
        for data in datasets:
            T = data['temperature']
            log_gamma_mu_temp = log_gamma_mu_base + temp_gamma_slope * T + temp_gamma_nonlinear * T**2
            gamma_temp = pm.LogNormal(f'gamma_T{T}', mu=log_gamma_mu_temp + log_gamma_offset_base, sigma=log_gamma_sigma_base, shape=7)
            gamma_list.append(gamma_temp)
        
        gamma_concat = pt.concatenate(gamma_list)
        
        op = TemperatureMagneticModelOp(datasets, temperature_specific_params, model_type)
        mu = op(a_scale, gamma_concat, g_factor, B4, B6)
        
        # ▼▼▼【変更点1】重み付き尤度関数の実装 ▼▼▼
        # 重みが0より大きいデータ点のみをフィッティングに使用する
        valid_indices = np.where(combined_weights > 0)[0]
        
        # 重み付けされたデータのみを使用してモデルを再構築
        trans_observed_weighted = trans_observed[valid_indices]
        weights_weighted = combined_weights[valid_indices]
        
        # 重み付きデータに対応するための新しいOpを作成
        datasets_weighted = []
        weights_start_idx = 0
        for i, data in enumerate(datasets):
            n_points = len(data['transmittance_full'])
            weights_end_idx = weights_start_idx + n_points
            
            # このデータセットの重み
            dataset_weights = combined_weights[weights_start_idx:weights_end_idx]
            dataset_valid_indices = np.where(dataset_weights > 0)[0]
            
            if len(dataset_valid_indices) > 0:
                # 重み付けされたデータセットを作成
                weighted_dataset = {
                    'temperature': data['temperature'],
                    'b_field': data['b_field'],
                    'frequency': data['frequency'][dataset_valid_indices],
                    'transmittance_full': data['transmittance_full'][dataset_valid_indices],
                    'omega': data['omega'][dataset_valid_indices],
                    'weights': dataset_weights[dataset_valid_indices]
                }
                datasets_weighted.append(weighted_dataset)
            
            weights_start_idx = weights_end_idx
        
        # 重み付きデータセット用のOpを使用
        if datasets_weighted:
            op_weighted = TemperatureMagneticModelOp(datasets_weighted, temperature_specific_params, model_type)
            mu = op_weighted(a_scale, gamma_concat, g_factor, B4, B6)
            
            # 重み配列を統合
            weights_tensor = pt.as_tensor_variable(np.concatenate([d['weights'] for d in datasets_weighted]))
            
            sigma = pm.HalfNormal('sigma', sigma=0.05)
            
            # 重みに応じてsigmaを調整 (重みが大きいほどsigmaは小さくなる)
            sigma_adjusted = sigma / pt.sqrt(weights_tensor)
            
            # 重み付きデータのターゲット
            trans_target = np.concatenate([d['transmittance_full'] for d in datasets_weighted])
            
            # 重み付けされたデータで尤度を計算
            Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma_adjusted, observed=trans_target)
        else:
            print("⚠️ 警告: 有効な重み付きデータポイントがありません。")
            return None
        # ▲▲▲【変更点1】重み付き尤度関数の実装 ▲▲▲
        
        # サンプリング設定（設定ファイルから取得）
        try:
            trace = pm.sample(
                draws=MCMC_CONFIG['draws'], 
                tune=MCMC_CONFIG['tune'], 
                chains=MCMC_CONFIG['chains'],
                target_accept=MCMC_CONFIG['target_accept'],
                return_inferencedata=True
            )
            print("✅ ベイズサンプリングが正常に完了しました。")
        except Exception as e:
            print(f"❌ ベイズサンプリングに失敗しました: {e}")
            return None

    # ベイズ推定結果をファイルに保存
    trace_filename = RESULTS_DIR / f'trace_{model_type}.nc'
    az.to_netcdf(trace, trace_filename)
    print(f"✅ Traceオブジェクトを保存しました: {trace_filename}")

    print("----------------------------------------------------")
    print("▶ 温度依存ベイズ推定結果 (サマリー):")
    try:
        summary = az.summary(trace, var_names=['a_scale', 'g_factor', 'B4', 'B6'])
        print(summary)
    except KeyError as e:
        print(f"サマリー表示でエラーが発生しました: {e}")
        try:
            summary = az.summary(trace)
            print(summary)
        except:
            print("サマリーの表示に失敗しました。")
    print("----------------------------------------------------")
    return trace


# ▼▼▼【変更点2】事後分布プロット関数の追加 ▼▼▼
def plot_posterior_distributions(trace: az.InferenceData, model_type: str):
    """
    ベイズ推定の事後分布とトレースプロットを可視化し、保存する。
    これにより、サンプリングの収束やパラメータの分布形状を確認できる。
    """
    print(f"\n--- {model_type}モデルの事後分布をプロット中 ---")
    
    # プロットする主要な変数を指定
    var_names = ['a_scale', 'g_factor', 'B4', 'B6', 
                 'log_gamma_mu_base', 'temp_gamma_slope', 'temp_gamma_nonlinear']
    
    try:
        # ArviZのplot_traceを使用してプロット
        axes = az.plot_trace(trace, var_names=var_names, compact=True, kind='rank_bars')
        plt.suptitle(f'Posterior Trace Plot for {model_type} Model', fontsize=16, y=1.02)
        fig = plt.gcf() # 現在のFigureを取得
        fig.savefig(RESULTS_DIR / f'posterior_trace_{model_type}.png', bbox_inches='tight')
        plt.show()
        print(f"✅ 事後分布プロットを保存しました: posterior_trace_{model_type}.png")

    except Exception as e:
        print(f"⚠️ 事後分布のプロットに失敗しました: {e}")
# ▲▲▲【変更点2】事後分布プロット関数の追加 ▲▲▲


def run_analysis_workflow():
    """
    【変更点1】反復プロセスを廃止し、重み付けを利用した単一の解析ワークフロー。
    1. データをロードし、高周波と全周波数領域のデータセットを作成。
    2. 高周波データから各温度のeps_bgを一度だけフィッティング。
    3. 全周波数データからピーク情報を基に周波数ごとの重み配列を生成。
    4. 重み付けした尤度関数を用いてベイズ推定を実行。
    5. 結果を可視化・保存する。
    """
    print("🚀 重み付きベイズ推定ワークフローを開始します")
    
    # 1. データの読み込み
    all_datasets_full_range = load_data_full_range_temperature(DATA_FILE_PATH, DATA_SHEET_NAME)
    high_freq_datasets = load_and_split_data_three_regions_temperature(
        file_path=DATA_FILE_PATH, sheet_name=DATA_SHEET_NAME
    )['high_freq']

    if not all_datasets_full_range or not high_freq_datasets:
        print("❌ 必要なデータが読み込めませんでした。処理を終了します。")
        return
        
    # 2. 各温度のeps_bgを一度だけフィッティング
    print("\n--- ステップ1: 各温度の高周波eps_bgフィッティング ---")
    temperature_specific_params = {}
    for dataset in high_freq_datasets:
        temp = dataset['temperature']
def fit_single_temperature_cavity_modes(dataset: Dict[str, Any]) -> Dict[str, float]:
    """高周波データからeps_bgフィッティング（fit_eps_bg_only_temperatureのエイリアス）"""
    return fit_eps_bg_only_temperature(dataset)

def save_fitting_parameters_to_csv(final_traces: Dict[str, az.InferenceData], 
                                  temperature_specific_params: Dict[float, Dict[str, float]]):
    """フィッティング結果をCSVファイルに保存"""
    print("\n--- フィッティング結果をCSVに保存中 ---")
    
    # 磁気パラメータの結果を保存
    for model_type, trace in final_traces.items():
        params = extract_bayesian_parameters(trace)
        
        # パラメータ結果
        params_df = pd.DataFrame([params])
        params_file = RESULTS_DIR / f'fitting_parameters_{model_type}.csv'
        params_df.to_csv(params_file, index=False)
        print(f"✅ 磁気パラメータを保存: {params_file}")
    
    # 温度別光学パラメータの結果を保存
    temp_params_list = []
    for temp, params in sorted(temperature_specific_params.items()):
        temp_params_list.append(params)
    
    if temp_params_list:
        temp_df = pd.DataFrame(temp_params_list)
        temp_file = RESULTS_DIR / 'temperature_optical_parameters.csv'
        temp_df.to_csv(temp_file, index=False)
        print(f"✅ 温度別光学パラメータを保存: {temp_file}")

def plot_temperature_dependencies(temperature_specific_params: Dict[float, Dict[str, float]], 
                                trace: az.InferenceData):
    """温度依存性をプロットする（膜厚は固定値のため除外）"""
    print("\n--- 温度依存性の可視化 ---")
    
    temperatures = sorted(temperature_specific_params.keys())
    eps_bg_values = [temperature_specific_params[T]['eps_bg'] for T in temperatures]
    
    # 磁気パラメータを抽出
    magnetic_params = extract_bayesian_parameters(trace)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # eps_bg の温度依存性
    ax1.plot(temperatures, eps_bg_values, 'ro-', linewidth=2, markersize=8, label='背景誘電率')
    ax1.set_xlabel('温度 (K)', fontsize=12)
    ax1.set_ylabel('背景誘電率 eps_bg', fontsize=12)
    ax1.set_title('背景誘電率の温度依存性', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # 値をテキストで表示
    for T, eps in zip(temperatures, eps_bg_values):
        ax1.annotate(f'{eps:.2f}', (T, eps), textcoords="offset points", xytext=(0,10), ha='center')
    
    # 温度による効果の概要（サマリーパネル）
    ax2.text(0.1, 0.9, f'温度範囲: {min(temperatures)} - {max(temperatures)} K', fontsize=14, transform=ax2.transAxes)
    ax2.text(0.1, 0.8, f'eps_bg変化率: {(max(eps_bg_values)-min(eps_bg_values))/min(eps_bg_values)*100:.1f}%', fontsize=14, transform=ax2.transAxes)
    ax2.text(0.1, 0.7, f'固定磁場: {B_FIXED} T', fontsize=14, transform=ax2.transAxes)
    
    ax2.text(0.1, 0.5, '磁気パラメータ (温度非依存):', fontsize=14, transform=ax2.transAxes, weight='bold')
    ax2.text(0.1, 0.4, f'g因子 = {magnetic_params["g_factor"]:.4f}', fontsize=12, transform=ax2.transAxes)
    ax2.text(0.1, 0.3, f'B4 = {magnetic_params["B4"]:.6f} K', fontsize=12, transform=ax2.transAxes)
    ax2.text(0.1, 0.2, f'B6 = {magnetic_params["B6"]:.6f} K', fontsize=12, transform=ax2.transAxes)
    ax2.text(0.1, 0.1, f'G0 = {magnetic_params["G0"]:.3e}', fontsize=12, transform=ax2.transAxes)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('解析結果サマリー', fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'temperature_dependencies.png', dpi=300, bbox_inches='tight')
    plt.show()

# ダミー関数（不足している関数の最小実装）
def plot_combined_temperature_model_comparison(*args, **kwargs):
    print("⚠️ plot_combined_temperature_model_comparison は未実装です。")

def plot_model_selection_results_temperature(*args, **kwargs):
    print("⚠️ plot_model_selection_results_temperature は未実装です。")

def calculate_temperature_peak_errors(*args, **kwargs):
    print("⚠️ calculate_temperature_peak_errors は未実装です。")
    return {}

def save_peak_analysis_to_csv(*args, **kwargs):
    print("⚠️ save_peak_analysis_to_csv は未実装です。")
def fit_single_temperature_cavity_modes(dataset: Dict[str, Any]) -> Dict[str, float]:
    """高周波データからeps_bgフィッティング（fit_eps_bg_only_temperatureのエイリアス）"""
    return fit_eps_bg_only_temperature(dataset)

def save_fitting_parameters_to_csv(final_traces: Dict[str, az.InferenceData], 
                                  temperature_specific_params: Dict[float, Dict[str, float]]):
    """フィッティング結果をCSVファイルに保存"""
    print("\n--- フィッティング結果をCSVに保存中 ---")
    
    # 磁気パラメータの結果を保存
    for model_type, trace in final_traces.items():
        params = extract_bayesian_parameters(trace)
        
        # パラメータ結果
        params_df = pd.DataFrame([params])
        params_file = RESULTS_DIR / f'fitting_parameters_{model_type}.csv'
        params_df.to_csv(params_file, index=False)
        print(f"✅ 磁気パラメータを保存: {params_file}")
    
    # 温度別光学パラメータの結果を保存
    temp_params_list = []
    for temp, params in sorted(temperature_specific_params.items()):
        temp_params_list.append(params)
    
    if temp_params_list:
        temp_df = pd.DataFrame(temp_params_list)
        temp_file = RESULTS_DIR / 'temperature_optical_parameters.csv'
        temp_df.to_csv(temp_file, index=False)
        print(f"✅ 温度別光学パラメータを保存: {temp_file}")

def plot_temperature_dependencies(temperature_specific_params: Dict[float, Dict[str, float]], 
                                trace: az.InferenceData):
    """温度依存性をプロットする（膜厚は固定値のため除外）"""
    print("\n--- 温度依存性の可視化 ---")
    
    temperatures = sorted(temperature_specific_params.keys())
    eps_bg_values = [temperature_specific_params[T]['eps_bg'] for T in temperatures]
    
    # 磁気パラメータを抽出
    magnetic_params = extract_bayesian_parameters(trace)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # eps_bg の温度依存性
    ax1.plot(temperatures, eps_bg_values, 'ro-', linewidth=2, markersize=8, label='背景誘電率')
    ax1.set_xlabel('温度 (K)', fontsize=12)
    ax1.set_ylabel('背景誘電率 eps_bg', fontsize=12)
    ax1.set_title('背景誘電率の温度依存性', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # 値をテキストで表示
    for T, eps in zip(temperatures, eps_bg_values):
        ax1.annotate(f'{eps:.2f}', (T, eps), textcoords="offset points", xytext=(0,10), ha='center')
    
    # 温度による効果の概要（サマリーパネル）
    ax2.text(0.1, 0.9, f'温度範囲: {min(temperatures)} - {max(temperatures)} K', fontsize=14, transform=ax2.transAxes)
    ax2.text(0.1, 0.8, f'eps_bg変化率: {(max(eps_bg_values)-min(eps_bg_values))/min(eps_bg_values)*100:.1f}%', fontsize=14, transform=ax2.transAxes)
    ax2.text(0.1, 0.7, f'固定磁場: {B_FIXED} T', fontsize=14, transform=ax2.transAxes)
    
    ax2.text(0.1, 0.5, '磁気パラメータ (温度非依存):', fontsize=14, transform=ax2.transAxes, weight='bold')
    ax2.text(0.1, 0.4, f'g因子 = {magnetic_params["g_factor"]:.4f}', fontsize=12, transform=ax2.transAxes)
    ax2.text(0.1, 0.3, f'B4 = {magnetic_params["B4"]:.6f} K', fontsize=12, transform=ax2.transAxes)
    ax2.text(0.1, 0.2, f'B6 = {magnetic_params["B6"]:.6f} K', fontsize=12, transform=ax2.transAxes)
    ax2.text(0.1, 0.1, f'G0 = {magnetic_params["G0"]:.3e}', fontsize=12, transform=ax2.transAxes)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('解析結果サマリー', fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'temperature_dependencies.png', dpi=300, bbox_inches='tight')
    plt.show()

# ダミー関数（不足している関数の最小実装）
def plot_combined_temperature_model_comparison(*args, **kwargs):
    print("⚠️ plot_combined_temperature_model_comparison は未実装です。")

def plot_model_selection_results_temperature(*args, **kwargs):
    print("⚠️ plot_model_selection_results_temperature は未実装です。")

def calculate_temperature_peak_errors(*args, **kwargs):
    print("⚠️ calculate_temperature_peak_errors は未実装です。")
    return {}

def save_peak_analysis_to_csv(*args, **kwargs):
    print("⚠️ save_peak_analysis_to_csv は未実装です。")

def run_analysis_workflow():
    """
    【変更点1】反復プロセスを廃止し、重み付けを利用した単一の解析ワークフロー。
    1. データをロードし、高周波と全周波数領域のデータセットを作成。
    2. 高周波データから各温度のeps_bgを一度だけフィッティング。
    3. 全周波数データからピーク情報を基に周波数ごとの重み配列を生成。
    4. 重み付けした尤度関数を用いてベイズ推定を実行。
    5. 結果を可視化・保存する。
    """
    print("🚀 重み付きベイズ推定ワークフローを開始します")
    
    # 1. データの読み込み
    all_datasets_full_range = load_data_full_range_temperature(DATA_FILE_PATH, DATA_SHEET_NAME)
    high_freq_datasets = load_and_split_data_three_regions_temperature(
        file_path=DATA_FILE_PATH, sheet_name=DATA_SHEET_NAME
    )['high_freq']

    if not all_datasets_full_range or not high_freq_datasets:
        print("❌ 必要なデータが読み込めませんでした。処理を終了します。")
        return
        
    # 2. 各温度のeps_bgを一度だけフィッティング
    print("\n--- ステップ1: 各温度の高周波eps_bgフィッティング ---")
    temperature_specific_params = {}
    for dataset in high_freq_datasets:
        temp = dataset['temperature']
        result = fit_single_temperature_cavity_modes(dataset)
        if result:
            temperature_specific_params[temp] = result
        else:
            # 失敗した場合は初期値を使用
            temperature_specific_params[temp] = {'eps_bg': eps_bg_init, 'd': d_fixed, 'temperature': temp}

    # 3. 周波数ごとの重み配列を生成
    print("\n--- ステップ2: 尤度関数のための重み配列を生成 ---")
    weights_list = [create_frequency_weights(d) for d in all_datasets_full_range]

    # 4. 重み付きベイズ推定の実行 (H-form と B-form)
    final_traces = {}
    prior_params = None
    
    for model_type in ['H_form', 'B_form']:
        print(f"\n--- ステップ3: {model_type}モデルのベイズ推定を実行 ---")
        trace = run_temperature_bayesian_fit(
            all_datasets_full_range,
            temperature_specific_params,
            weights_list,
            prior_magnetic_params=prior_params,
            model_type=model_type
        )
        if trace:
            final_traces[model_type] = trace
            # 最初のモデル(H_form)の結果を次のモデル(B_form)の事前分布に利用
            if prior_params is None:
                prior_params = extract_bayesian_parameters(trace)
            
            # ▼▼▼【変更点2】事後分布をプロット ▼▼▼
            plot_posterior_distributions(trace, model_type)
            # ▲▲▲【変更点2】事後分布をプロット ▲▲▲
            
    # 5. 結果の評価と可視化
    if not final_traces:
        print("❌ ベイズ推定が両モデルで失敗しました。処理を終了します。")
        return
        
    print("\n--- ステップ4: 最終結果の評価と可視化 ---")
    if len(final_traces) >= 2:
        plot_combined_temperature_model_comparison(all_datasets_full_range, temperature_specific_params, final_traces)
        plot_model_selection_results_temperature(final_traces)
        peak_analysis_results = calculate_temperature_peak_errors(all_datasets_full_range, temperature_specific_params, final_traces)
        save_peak_analysis_to_csv(peak_analysis_results)
    
    save_fitting_parameters_to_csv(final_traces, temperature_specific_params)
    plot_temperature_dependencies(temperature_specific_params, list(final_traces.values())[0])

    print("\n🎉 全ての解析ワークフローが完了しました。")


if __name__ == "__main__":
    # 解析ワークフローを実行
    run_analysis_workflow()