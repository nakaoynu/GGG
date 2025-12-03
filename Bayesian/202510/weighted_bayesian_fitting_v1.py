# weighted_bayesian_fitting_2param_gamma.py - 2パラメータgammaを用いた重み付きベイズ推定
#
# 【重要】NUTSサンプラーバックエンドについて:
#   - config.ymlで nuts_sampler: "numpyro" を指定した場合、
#     numpyroとjaxのインストールが必要です:
#       pip install numpyro jax jaxlib
#   - numpyroは高速ですが、インストールできない場合は
#     config.ymlから nuts_sampler 行を削除すれば PyMC標準のNUTSが使用されます
#
# 【再現性について】(2025-11-06追加):
#   - config.ymlの mcmc.random_seed でシード値を設定すると結果が再現可能になります
#   - random_seed: 42 などの整数を指定してください
#   - マルチチェーンサンプリングでも、各チェーンに異なるシードが自動割り当てされ、
#     同じrandom_seedからの実行は常に同じ結果を生成します
#
# 【gammaモデルの変更点】 (2025-11-14):
#   - 温度依存性を削除
#   - 7要素のgammaを2つのパラメータ（log_gamma_min, log_gamma_other）で記述
#   - gamma配列は [gamma_min, gamma_other, gamma_other, ...] となる
#   - 目的: パラメータ数を削減し、収束性を向上させる

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
from typing import List, Dict, Any, Tuple, Optional, Union
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit

# 数値計算の警告を抑制
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(all='ignore')  # NumPyの警告も抑制

# --- 設定ファイル読み込み機能 ---
def load_config(config_path: Optional[Union[str, pathlib.Path]] = None) -> Dict[str, Any]:
    """設定ファイル(YAML)を読み込み、デフォルト値とマージする"""
    if config_path is None:
        config_path = pathlib.Path(__file__).parent / "config.yml" # デフォルトの設定ファイルパス
    
    # デフォルト設定値
    default_config = {
        'file_paths': {
            'data_file': "C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\corrected_exp_datasets\\Corrected_Transmittance_Temperature.xlsx",
            'sheet_name': "Corrected Data",
            'results_parent_dir': "analysis_results_2param_gamma" # 結果ディレクトリ名を変更
        },
        'execution': {
            'use_gpu': False
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
            'max_iterations': 2 # (例：反復回数を設定)
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
            # ▼▼▼【変更】gammaの事前分布を2パラメータに変更 ▼▼▼
            'gamma_parameters': {
                'log_gamma_min': {'distribution': 'Normal', 'sigma': 1.0},
                'log_gamma_other': {'distribution': 'Normal', 'sigma': 1.0}
            },
            'noise_parameters': {
                'sigma': {'distribution': 'HalfNormal', 'sigma': 0.05}
            }
        }
    }
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
        
        # 再帰的にデフォルト設定を更新
        def merge_dict(default, user):
            for key, value in user.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    merge_dict(default[key], value) # デフォルトの辞書を再帰的に.yamlファイルの内容で更新
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
print("--- 0. 環境設定を開始します ---")

# 設定ファイルの読み込み（config.ymlから変更したい場合はここを編集）
CONFIG = load_config(config_path=pathlib.Path(__file__).parent / "config_v1.yml")

# 再現性確保のための乱数シード設定 (2025-11-06追加)
# NumPy乱数生成器を固定（データ前処理や最適化で使用される可能性がある）
if 'random_seed' in CONFIG['mcmc']:
    RANDOM_SEED = CONFIG['mcmc']['random_seed']
    np.random.seed(RANDOM_SEED)
    print(f"🎲 乱数シード設定: {RANDOM_SEED} (NumPy & PyMC)")
else:
    RANDOM_SEED = None
    print("ℹ️ 乱数シード未設定（結果は実行ごとに変わります）")

# GPU利用設定（PyTensor新バージョン対応・安全版）
USE_GPU = CONFIG['execution']['use_gpu']
print("🔧 PyTensor設定を初期化中...")

# PyTensorの環境変数は必ずimport前に設定する必要がある
if USE_GPU:
    try:
        import cupy
        print("✅ CuPy が利用可能です。")
        # 新しいPyTensorでのGPU設定の試行
        try:
            os.environ['PYTENSOR_FLAGS'] = 'device=cuda,floatX=float64'
            print("🚀 GPU (CUDA) 設定を適用しました。")
        except:
            print("⚠️ CUDA設定に失敗。CPUにフォールバックします。")
            os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float64'
    except ImportError:
        print("⚠️ CuPy が見つかりません。CPUを使用します。")
        os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float64'
else:
    print("💻 CPU設定を使用します。")
    os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float64'

try:
    import japanize_matplotlib
except ImportError:
    print("警告: japanize_matplotlib が見つかりません。")

plt.rcParams['figure.dpi'] = 120

# 結果保存ディレクトリ（メインプロセスでのみ作成）
RESULTS_DIR = None

# --- 1. 物理定数とシミュレーション条件 ---
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
eps_bg_init = float(initial_values['eps_bg'])
B4_init = float(initial_values['B4'])
B6_init = float(initial_values['B6'])
gamma_init = float(initial_values['gamma']) # 2パラメータgammaの事前分布の平均値として使用
a_scale_init = float(initial_values['a_scale'])
g_factor_init = float(initial_values['g_factor'])

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
# ▼▼▼【変更】デフォルトのgamma_init値で初期化 ▼▼▼
_FIXED_GAMMA_ARRAY = np.full(7, gamma_init)

def normalize_gamma_array(gamma_input, target_length: int = 7) -> np.ndarray:
    """
    ガンマ配列の正規化と型安全性を確保（可変長対応版）
    
    Args:
        gamma_input: スカラー、配列、またはPyTensorテンソル
        target_length: 目標配列長（デフォルト7）
    """
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
            # 短い場合は最後の値で埋める（edge mode）
            return np.pad(gamma_array, (0, target_length - len(gamma_array)), 'edge')
    else:
        # PyTensorテンソルなどの .item() 呼び出し
        try:
            return np.full(target_length, gamma_input.item())
        except Exception:
            # フォールバック
            return np.full(target_length, gamma_init)


def get_eps_bg_initial_values_and_bounds(temperature: float) -> Tuple[List[float], Tuple[float, float]]:
    """温度依存eps_bg初期値と境界値の取得（統合ユーティリティ関数）"""
    eps_bg_init_val = eps_bg_init # グローバル設定から読み込み
    if temperature <= 10:
        # 低温では低めの初期値から開始（フォノンの凍結効果）
        initial_eps_bg_values = [eps_bg_init_val * 0.85, eps_bg_init_val * 0.90, eps_bg_init_val * 0.95, eps_bg_init_val,
                                13.0, 12.5, 12.8, 13.2, 13.5, 14.0]
        bounds_eps_bg = (11.0, 16.0)
    elif temperature <= 100:
        # 中間温度では標準的な初期値
        initial_eps_bg_values = [eps_bg_init_val * 0.98, eps_bg_init_val, eps_bg_init_val * 1.02, eps_bg_init_val * 1.05,
                                13.8, 14.0, 14.2, 13.5, 14.5, 13.2]
        bounds_eps_bg = (11.5, 16.5)
    else:
        # 高温では高めの初期値から開始（フォノンの活性化）
        initial_eps_bg_values = [eps_bg_init_val * 1.05, eps_bg_init_val * 1.10, eps_bg_init_val * 1.15, eps_bg_init_val,
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
    # (入力gamma_arrayはPyMCモデル側で7要素に整形されて渡される)
    gamma_array = normalize_gamma_array(gamma_array)

    # eighは固有値を昇順で返す (E0, E1, ..., E7)
    eigenvalues, _ = np.linalg.eigh(H)
    eigenvalues -= np.min(eigenvalues)
    
    # 数値的安定性のためのクリッピング
    eigenvalues = np.clip(eigenvalues / (kB * T), -700, 700)
    
    Z = np.sum(np.exp(-eigenvalues))
    populations = np.exp(-eigenvalues) / Z
    
    # delta_E[0] = E1-E0 (最低次遷移), ..., delta_E[6] = E7-E6
    delta_E = (eigenvalues[1:] - eigenvalues[:-1]) * kB * T  # 元の単位に戻す
    delta_pop = populations[1:] - populations[:-1]
    
    # 無効な値をチェック
    valid_mask = np.isfinite(delta_E) & (np.abs(delta_E) > 1e-30)
    if not np.any(valid_mask):
        return np.zeros_like(omega_array, dtype=complex)
    
    omega_0 = delta_E / hbar
    m_vals = np.arange(s, -s, -1) # 7要素
    transition_strength = (s + m_vals) * (s - m_vals + 1) # 7要素
    
    # === デバッグ出力: gamma_arrayがdelta_Eと同じ次元を持つように調整 ===
    if len(gamma_array) != len(delta_E):
        # print(f"  Gamma/Delta_E 長さ不一致: gamma={len(gamma_array)}, delta_E={len(delta_E)}")
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

# ▼▼▼ 設定ファイル対応の重み付け関数 ▼▼▼
def create_frequency_weights(dataset: Dict[str, Any], analysis_settings: Dict[str, Any]) -> np.ndarray:
    """
    実験データのピーク特性に基づき、尤度関数のための重み配列を生成する。
    設定ファイルの値を使用してピーク検出と重み付けを行う。
    
    Args:
        dataset: 周波数と透過率データを含む辞書
        analysis_settings: 解析設定（weight_settings と high_freq_cutoff を含む）
    """
    # 設定値を取得
    weight_config = analysis_settings['weight_settings']
    high_freq_cutoff = analysis_settings['high_freq_cutoff']
    
    freq = dataset['frequency']
    trans = dataset['transmittance_full']
    
    # ピーク検出 (設定ファイルの値を使用)
    peaks, properties = find_peaks(trans,
                                   height=weight_config['peak_height_threshold'],
                                   prominence=weight_config['peak_prominence_threshold'],
                                   distance=weight_config['peak_distance'])
    
    if len(peaks) < 2:
        # ピークが少なすぎる場合は、低周波領域全体に均一な重みを付ける
        weights = np.zeros_like(freq)
        low_freq_mask = freq < high_freq_cutoff
        weights[low_freq_mask] = 1.0
        return weights

    # 各ピークの半値幅を計算
    widths, _, left_ips, right_ips = peak_widths(trans, peaks, rel_height=0.5)
    
    # 周波数単位に変換
    left_freq = np.interp(left_ips, np.arange(len(freq)), freq)
    right_freq = np.interp(right_ips, np.arange(len(freq)), freq)

    # 重み配列の初期化
    weights = np.full_like(freq, weight_config['background_weight'])

    # LPとUPを特定（プロミネンスでソート）
    low_freq_peaks = peaks[freq[peaks] < high_freq_cutoff]
    if len(low_freq_peaks) >= 2:
        # プロミネンスでソートして上位2つを取得
        peak_prominences = properties['prominences'][freq[peaks] < high_freq_cutoff]
        sorted_indices = np.argsort(peak_prominences)[::-1]  # 降順ソート

        lp_idx_in_all_peaks = np.where(peaks == low_freq_peaks[sorted_indices[0]])[0][0] #実験データに基づいてsorted_indicesを決める→LPとUPを決定
        up_idx_in_all_peaks = np.where(peaks == low_freq_peaks[sorted_indices[1]])[0][0]

        # ピーク間領域の重み付け（半値幅の外側の間）
        # 堅牢性のため、周波数の大小関係を自動判定
        lp_fwhm_right_freq = right_freq[lp_idx_in_all_peaks]
        up_fwhm_left_freq = left_freq[up_idx_in_all_peaks]
        
        lower_bound = np.minimum(lp_fwhm_right_freq, up_fwhm_left_freq)
        upper_bound = np.maximum(lp_fwhm_right_freq, up_fwhm_left_freq)
        between_mask = (freq >= lower_bound) & (freq <= upper_bound)
        weights[between_mask] = weight_config['between_peaks_weight']
        
        # LPとUPの半値幅領域に重みを付与
        lp_fwhm_mask = (freq >= left_freq[lp_idx_in_all_peaks]) & (freq <= right_freq[lp_idx_in_all_peaks])
        up_fwhm_mask = (freq >= left_freq[up_idx_in_all_peaks]) & (freq <= right_freq[up_idx_in_all_peaks])
        weights[lp_fwhm_mask] = weight_config['lp_up_peak_weight']
        weights[up_fwhm_mask] = weight_config['lp_up_peak_weight']
        
    # 高周波の共振器モードにも重みを付与
    high_freq_peak_indices = np.where(freq[peaks] >= high_freq_cutoff)[0]
    for idx_in_all_peaks in high_freq_peak_indices:
        fwhm_mask = (freq >= left_freq[idx_in_all_peaks]) & (freq <= right_freq[idx_in_all_peaks])
        weights[fwhm_mask] = weight_config['high_freq_peak_weight']

    print(f"  温度 {dataset['temperature']}K: 重み配列を生成。(全データ数：{len(freq)})\n"
          f"    - LP/UPピーク (重み={weight_config['lp_up_peak_weight']}): {np.sum(weights == weight_config['lp_up_peak_weight'])} 点\n"
          f"    - LP-UP間 (重み={weight_config['between_peaks_weight']}): {np.sum(weights == weight_config['between_peaks_weight'])} 点\n"
          f"    - 高周波ピーク (重み={weight_config['high_freq_peak_weight']}): {np.sum(weights == weight_config['high_freq_peak_weight'])} 点")
    return weights

# --- データ読み込み関数群 ---
def load_all_full_range_data(file_path: str, sheet_name: str) -> List[Dict[str, Any]]:
    """全範囲データのみを読み込む（eps_bgフィッティング用ではない）"""
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
            'transmittance_full': trans, # 正規化されていない全データ
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
                               bayesian_params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """各温度で高周波データからeps_bgのみをフィッティングする（他パラメータは固定またはベイズ推定結果を使用）"""
    print(f"\n--- 温度 {dataset['temperature']} K の高周波eps_bgフィッティング ---")
    
    # パラメータの優先順位：ベイズ推定結果 > 固定パラメータ > 初期値
    if bayesian_params is not None:
        # ベイズ推定結果を使用（更新されたパラメータ）
        effective_params = {
            'd': d_fixed,
            'g_factor': bayesian_params.get('g_factor', g_factor_init),
            'B4': bayesian_params.get('B4', B4_init),
            'B6': bayesian_params.get('B6', B6_init),
            'a_scale': bayesian_params.get('a_scale', a_scale_init)  # スケーリング係数も更新
        }
        print(f"  🔄 ベイズ推定結果を使用:")
        print(f"     g_factor = {effective_params['g_factor']:.6f}")
        print(f"     B4 = {effective_params['B4']:.6f}")
        print(f"     B6 = {effective_params['B6']:.6f}")
        print(f"     a_scale = {effective_params['a_scale']:.6f}")
        
        # ▼▼▼【変更】 2パラメータgammaもベイズ推定結果から使用 ▼▼▼
        if 'gamma_min' in bayesian_params and 'gamma_other' in bayesian_params:
            gamma_min_fit = bayesian_params['gamma_min']
            gamma_other_fit = bayesian_params['gamma_other']
            # [min, other, other, ...] の順序
            gamma_array_fit = np.array([gamma_min_fit] + [gamma_other_fit] * 6)
            print(f"     gamma_min = {gamma_min_fit:.3e}")
            print(f"     gamma_other = {gamma_other_fit:.3e}")
        else:
            # フォールバック
            gamma_array_fit = _FIXED_GAMMA_ARRAY
            
    elif fixed_params is not None:
        effective_params = fixed_params
        gamma_array_fit = _FIXED_GAMMA_ARRAY
        print(f"  📌 固定パラメータを使用")
    else:
        # デフォルト初期値を使用
        effective_params = {
            'd': d_fixed,
            'g_factor': g_factor_init,
            'B4': B4_init,
            'B6': B6_init,
        }
        gamma_array_fit = _FIXED_GAMMA_ARRAY
        print(f"  🔰 初期値を使用")
    
    def magnetic_cavity_model_eps_bg_only(freq_thz, eps_bg_fit):
        """eps_bgのみを変数とする高周波透過率モデル（他パラメータは更新された値で固定）"""
        try:
            omega = freq_thz * 1e12 * 2 * np.pi
            
            # 更新されたパラメータから値を取得
            g_factor_fit = effective_params['g_factor']
            B4_fit = effective_params['B4']
            B6_fit = effective_params['B6']
            
            # ハミルトニアンと磁気感受率の計算（キャッシュ使用）
            H = get_hamiltonian_cached(B_FIXED, g_factor_fit, B4_fit, B6_fit)
            
            # ▼▼▼【変更】 ベイズ推定結果のgamma配列を使用 ▼▼▼
            chi_raw = calculate_susceptibility(omega, H, dataset['temperature'], gamma_array_fit)
            
            # 磁気感受率のスケーリング（ベイズ推定結果のa_scaleを使用）
            if 'a_scale' in effective_params:
                # ベイズ推定結果のa_scaleを使用
                G0 = effective_params['a_scale'] * mu0 * N_spin * (g_factor_fit * muB)**2 / (2 * hbar)
            else:
                # 従来の方法
                G0 = mu0 * N_spin * (g_factor_fit * muB)**2 / (2 * hbar)
            
            chi = G0 * chi_raw
            
            # H_formで透磁率を計算
            mu_r = 1 + chi
            
            return calculate_normalized_transmission(omega, mu_r, d_fixed, eps_bg_fit)
        except Exception as e:
            print(f"    警告: モデル計算エラー {e}")
            return np.ones_like(freq_thz) * 0.5

    # 複数の初期値を試行
    success = False
    result = {}
    
    # 温度依存の初期値と境界値を取得
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
                print(f"  ✅ 成功 (試行 {attempt + 1}): eps_bg = {eps_bg_fit:.3f}")
                result = {
                    'eps_bg': eps_bg_fit,
                    'd': d_fixed,
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

# --- PyMC Op クラス ---
class TemperatureMagneticModelOp(Op):
    """
    ▼▼▼【変更】 2パラメータgamma対応 ▼▼▼
    温度"非"依存の2パラメータgammaを扱うためのPyMC Op。
    入力のgamma_concatは (全温度データセット数 * 7) の長さを持つが、
    中身は [gamma_min, gamma_other, ...] の繰り返しとなっている。
    """
    def __init__(self, datasets: List[Dict[str, Any]], temperature_specific_params: Dict[float, Dict[str, float]], model_type: str):
        super().__init__()
        self.datasets = datasets
        self.temperature_specific_params = temperature_specific_params
        self.model_type = model_type
        # self.temp_list = sorted(list(set([d['temperature'] for d in datasets]))) # 温度依存性がないため不要
        
        # 入力タイプは変更なし (a_scale, gamma_concat, g_factor, B4, B6)
        self.itypes = [pt.dscalar, pt.dvector, pt.dscalar, pt.dscalar, pt.dscalar]  
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
            
            # ▼▼▼【変更】 7個ずつ切り出すロジックは同じだが、中身は温度によらず一定 ▼▼▼
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

# ▼▼▼【変更】 2パラメータgamma対応 ▼▼▼
def extract_bayesian_parameters(trace: az.InferenceData) -> Dict[str, float]:
    """ベイズ推定結果から平均パラメータを抽出（2パラメータgamma対応）"""
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
    
    # 2パラメータgammaの値を追加
    try:
        result['log_gamma_min'] = posterior['log_gamma_min'].mean().item()
        result['log_gamma_other'] = posterior['log_gamma_other'].mean().item()
        # 物理的なgamma値も計算して追加
        result['gamma_min'] = np.exp(result['log_gamma_min'])
        result['gamma_other'] = np.exp(result['log_gamma_other'])
    except KeyError:
        # gammaパラメータがない場合はスキップ
        print("警告: gamma_min / gamma_other がトレースに見つかりません。")
        pass
    
    return result

# --- 事前分布設定関数 ---
def create_prior_distributions(prior_config: Dict[str, Any], 
                              prior_magnetic_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """設定ファイルから磁気パラメータの事前分布を作成する"""
    priors = {}
    
    if prior_magnetic_params is None:
        # 初回実行時：設定ファイルの magnetic_parameters を使用
        mag_config = prior_config['magnetic_parameters']
        
        # a_scale
        if mag_config['a_scale']['distribution'] == 'HalfNormal':
            priors['a_scale'] = pm.HalfNormal('a_scale', sigma=mag_config['a_scale']['sigma'])
        
        # g_factor
        if mag_config['g_factor']['distribution'] == 'Normal':
            priors['g_factor'] = pm.Normal('g_factor', mu=g_factor_init, sigma=mag_config['g_factor']['sigma'])
        
        # B4
        if mag_config['B4']['distribution'] == 'Normal':
            priors['B4'] = pm.Normal('B4', mu=B4_init, sigma=mag_config['B4']['sigma'])
        
        # B6
        if mag_config['B6']['distribution'] == 'Normal':
            priors['B6'] = pm.Normal('B6', mu=B6_init, sigma=mag_config['B6']['sigma'])
            
    else:
        # 事前情報がある場合：with_prior_info を使用
        prior_config_info = prior_config['with_prior_info']
        
        # a_scale
        if prior_config_info['a_scale']['distribution'] == 'Normal':
            priors['a_scale'] = pm.Normal('a_scale', mu=prior_magnetic_params['a_scale'], 
                                        sigma=prior_config_info['a_scale']['sigma'])
        
        # g_factor
        if prior_config_info['g_factor']['distribution'] == 'Normal':
            priors['g_factor'] = pm.Normal('g_factor', mu=prior_magnetic_params['g_factor'], 
                                         sigma=prior_config_info['g_factor']['sigma'])
        
        # B4
        if prior_config_info['B4']['distribution'] == 'Normal':
            priors['B4'] = pm.Normal('B4', mu=prior_magnetic_params['B4'], 
                                   sigma=prior_config_info['B4']['sigma'])
        
        # B6
        if prior_config_info['B6']['distribution'] == 'Normal':
            priors['B6'] = pm.Normal('B6', mu=prior_magnetic_params['B6'], 
                                   sigma=prior_config_info['B6']['sigma'])
    
    return priors

# ▼▼▼【変更】 2パラメータgammaの事前分布作成関数 ▼▼▼
def create_gamma_priors(gamma_config: Dict[str, Any], gamma_init_val: float) -> Dict[str, Any]:
    """
    gamma関連の事前分布を作成する (2パラメータ・温度非依存版)
    
    Args:
        gamma_config: configの 'gamma_parameters' セクション
        gamma_init_val: 初期値 (事前分布の平均として使用)
    """
    gamma_priors = {}
    log_gamma_init = np.log(gamma_init_val)
    
    # log_gamma_min (最低次遷移)
    cfg_min = gamma_config.get('log_gamma_min', {'distribution': 'Normal', 'sigma': 1.0})
    if cfg_min['distribution'] == 'Normal':
        gamma_priors['log_gamma_min'] = pm.Normal('log_gamma_min', 
                                                mu=log_gamma_init, 
                                                sigma=cfg_min['sigma'])
    
    # log_gamma_other (その他高次遷移)
    cfg_other = gamma_config.get('log_gamma_other', {'distribution': 'Normal', 'sigma': 1.0})
    if cfg_other['distribution'] == 'Normal':
        gamma_priors['log_gamma_other'] = pm.Normal('log_gamma_other', 
                                                  mu=log_gamma_init, 
                                                  sigma=cfg_other['sigma'])
    
    return gamma_priors

# --- 重み付きベイズ推定メイン関数 ---
def run_temperature_bayesian_fit(datasets: List[Dict[str, Any]], 
                                temperature_specific_params: Dict[float, Dict[str, float]],
                                weights_list: List[np.ndarray], 
                                results_dir: pathlib.Path,
                                prior_magnetic_params: Optional[Dict[str, float]] = None, 
                                model_type: str = 'H_form') -> Optional[az.InferenceData]:
    print(f"\n--- 重み付きベイズ推定 (モデル: {model_type}, Gamma: 2パラメータ) ---")
    
    # 観測データと重みを結合
    combined_weights = np.concatenate(weights_list)

    with pm.Model() as model:
        # ▼▼▼【変更】設定ファイルベースの事前分布設定 ▼▼▼
        
        # 設定ファイルから事前分布設定を取得
        prior_config = CONFIG['bayesian_priors']
        
        # 磁気パラメータの事前分布を作成
        magnetic_priors = create_prior_distributions(prior_config, prior_magnetic_params)
        a_scale = magnetic_priors['a_scale']
        g_factor = magnetic_priors['g_factor'] 
        B4 = magnetic_priors['B4']
        B6 = magnetic_priors['B6']
        
        # ▼▼▼【変更】 2パラメータgammaの事前分布を作成 ▼▼▼
        gamma_priors = create_gamma_priors(prior_config['gamma_parameters'], gamma_init)
        log_gamma_min = gamma_priors['log_gamma_min']
        log_gamma_other = gamma_priors['log_gamma_other']
        
        # ▼▼▼【変更】 温度非依存のgamma配列を構築 ▼▼▼
        gamma_min = pt.exp(log_gamma_min)
        gamma_other = pt.exp(log_gamma_other)
        
        # 7要素のベース配列を作成 [min, other, other, other, other, other, other]
        # (calculate_susceptibilityの実装上、最初の要素が E1-E0 に対応するため)
        gamma_base_vec = pt.concatenate([
            pt.stack([gamma_min]),         # 1要素
            pt.repeat(gamma_other, 6)      # 6要素
        ])
        
        # 各温度データセットでこの 'gamma_base_vec' を使用する
        gamma_final = []
        for _ in datasets: # データセットの数だけ複製
            gamma_final.append(gamma_base_vec)
        
        # Opに渡すための長い配列 (T_count * 7 要素)
        gamma_concat = pt.concatenate(gamma_final, axis=0)
        
        # 重み付きデータセット用のOpを使用
        # 重みが0より大きいデータ点のみをフィッティングに使用する
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
        
        if datasets_weighted:
            op_weighted = TemperatureMagneticModelOp(datasets_weighted, temperature_specific_params, model_type)
            mu = op_weighted(a_scale, gamma_concat, g_factor, B4, B6)
            
            # 重み配列を統合
            weights_tensor = pt.as_tensor_variable(np.concatenate([d['weights'] for d in datasets_weighted]))
            
            # ノイズパラメータも設定ファイルから取得
            noise_config = prior_config['noise_parameters']['sigma']
            if noise_config['distribution'] == 'HalfNormal':
                sigma = pm.HalfNormal('sigma', sigma=noise_config['sigma'])
            else:
                sigma = pm.HalfNormal('sigma', sigma=0.05)  # フォールバック
            
            # 重みに応じてsigmaを調整 (重みが大きいほどsigmaは小さくなる)
            sigma_adjusted = sigma / pt.sqrt(weights_tensor)
            
            # 重み付きデータのターゲット
            trans_target = np.concatenate([d['transmittance_full'] for d in datasets_weighted])
            
            # 重み付けされたデータで尤度を計算
            Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma_adjusted, observed=trans_target)
        else:
            print("⚠️ 警告: 有効な重み付きデータポイントがありません。")
            return None
        
        # サンプリング設定（設定ファイルから取得）
        try:
            # サンプリングパラメータの準備
            sample_kwargs = {
                'draws': MCMC_CONFIG['draws'], 
                'tune': MCMC_CONFIG['tune'], 
                'chains': MCMC_CONFIG['chains'],
                'target_accept': MCMC_CONFIG['target_accept'],
                'random_seed': MCMC_CONFIG.get('random_seed', None),  # 再現性確保
                'init': MCMC_CONFIG.get('init', 'auto'),
                'return_inferencedata': True,
                'progressbar': True,
                'idata_kwargs': {'log_likelihood': True}  # LOO-CV用にlog_likelihoodを保存
            }
            
            # ▼▼▼ NUTSサンプラーの指定（config.ymlに基づく） ▼▼▼
            if 'nuts_sampler' in MCMC_CONFIG:
                sample_kwargs['nuts_sampler'] = MCMC_CONFIG['nuts_sampler']
                print(f"🚀 NUTS Sampler: {sample_kwargs['nuts_sampler']} を使用します。")
            else:
                print("🚀 NUTS Sampler: PyMC default を使用します。")

            trace = pm.sample(**sample_kwargs)
            print("✅ ベイズサンプリングが正常に完了しました。")
            
            # 収束診断の自動チェック
            print("\n--- 収束診断 ---")
            # ▼▼▼【変更】 診断対象にgammaパラメータ追加 ▼▼▼
            diagnostic_vars = ['a_scale', 'g_factor', 'B4', 'B6', 'log_gamma_min', 'log_gamma_other']
            convergence_issues = []
            
            for var in diagnostic_vars:
                try:
                    var_summary = az.summary(trace, var_names=[var])
                    r_hat = var_summary['r_hat'].values[0]
                    ess_bulk = var_summary['ess_bulk'].values[0]
                    
                    # r_hat診断
                    if r_hat > 1.05:
                        convergence_issues.append(f"⚠️ {var}: r_hat={r_hat:.3f} (目標<1.05)")
                    elif r_hat > 1.01:
                        print(f"⚡ {var}: r_hat={r_hat:.3f} (許容範囲内だが注意)")
                    else:
                        print(f"✅ {var}: r_hat={r_hat:.3f} (良好)")
                    
                    # ESS診断
                    if ess_bulk < 400:
                        convergence_issues.append(f"⚠️ {var}: ess_bulk={ess_bulk:.0f} (目標>400)")
                    else:
                        print(f"✅ {var}: ess_bulk={ess_bulk:.0f} (十分)")
                        
                except Exception as e:
                    print(f"  {var}の診断に失敗: {e}")
            
            if convergence_issues:
                print("\n🔴 収束に問題があります:")
                for issue in convergence_issues:
                    print(f"  {issue}")
                print("  推奨: draws/tuneを増やすか、モデルを簡素化してください。")
            else:
                print("\n✅ 全パラメータが収束基準を満たしています。")
                
        except Exception as e:
            print(f"❌ ベイズサンプリングに失敗しました: {e}")
            return None

    # ベイズ推定結果をファイルに保存
    trace_filename = results_dir / f'trace_{model_type}.nc'
    az.to_netcdf(trace, trace_filename)
    print(f"✅ Traceオブジェクトを保存しました: {trace_filename}")

    print("----------------------------------------------------")
    print("▶ 温度依存ベイズ推定結果 (サマリー):")
    try:
        # ▼▼▼【変更】 gammaパラメータをサマリーに追加 ▼▼▼
        summary_vars = ['a_scale', 'g_factor', 'B4', 'B6', 'log_gamma_min', 'log_gamma_other', 'sigma']
        summary = az.summary(trace, var_names=summary_vars)
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

# ▼▼▼ 事後分布プロット関数の追加 ▼▼▼
# ▼▼▼ 2パラメータgamma対応 ▼▼▼
def plot_posterior_distributions(trace: az.InferenceData, model_type: str, results_dir: pathlib.Path):
    """
    ベイズ推定の事後分布とトレースプロットを可視化し、保存する。
    これにより、サンプリングの収束やパラメータの分布形状を確認できる。
    """
    print(f"\n--- {model_type}モデルの事後分布をプロット中 ---")
    
    # プロットする主要な変数を指定 (2パラメータgamma対応)
    var_names = ['a_scale', 'g_factor', 'B4', 'B6', 
                 'log_gamma_min', 'log_gamma_other'] 
    
    try:
        # ArviZのplot_traceを使用してプロット
        axes = az.plot_trace(trace, var_names=var_names, compact=True, kind='rank_bars')
        plt.suptitle(f'Posterior Trace Plot for {model_type} Model', fontsize=16, y=1.02)
        fig = plt.gcf() # 現在のFigureを取得
        fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=3.0, w_pad=2.0)
        fig.savefig(results_dir / f'posterior_trace_{model_type}.png', bbox_inches='tight')
        print(f"✅ 事後分布プロットを保存しました: posterior_trace_{model_type}.png")

    except Exception as e:
        print(f"⚠️ 事後分布のプロットに失敗しました: {e}")

# --- 結果保存・可視化関数群 ---
def fit_single_temperature_cavity_modes(dataset: Dict[str, Any], 
                                      bayesian_params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """高周波データからeps_bgフィッティング（ベイズ推定結果対応版）"""
    return fit_eps_bg_only_temperature(dataset, bayesian_params=bayesian_params)

def save_fitting_parameters_to_csv(final_traces: Dict[str, az.InferenceData], 
                                  model_temperature_params: Dict[str, Dict[float, Dict[str, float]]],
                                  results_dir: pathlib.Path):
    """フィッティング結果をCSVファイルに保存（両モデル対応版）"""
    print("\n--- フィッティング結果をCSVに保存中 ---")
    
    # 磁気パラメータの結果を保存
    for model_type, trace in final_traces.items():
        # ▼▼▼【変更】 2パラメータgamma対応の関数を呼ぶ ▼▼▼
        params = extract_bayesian_parameters(trace)
        
        # パラメータ結果
        params_df = pd.DataFrame([params])
        params_file = results_dir / f'fitting_parameters_{model_type}.csv'
        params_df.to_csv(params_file, index=False)
        print(f"✅ 磁気パラメータを保存: {params_file}")
    
    # 各モデルの温度別光学パラメータを保存
    for model_type, temperature_specific_params in model_temperature_params.items():
        temp_params_list = []
        for temp, params in sorted(temperature_specific_params.items()):
            temp_params_list.append(params)
        
        if temp_params_list:
            temp_df = pd.DataFrame(temp_params_list)
            temp_file = results_dir / f'temperature_optical_parameters_{model_type}.csv'
            temp_df.to_csv(temp_file, index=False)
            print(f"✅ {model_type}の温度別光学パラメータを保存: {temp_file}")

def plot_temperature_dependencies(model_temperature_params: Dict[str, Dict[float, Dict[str, float]]], 
                                final_traces: Dict[str, az.InferenceData],
                                results_dir: pathlib.Path):
    """温度依存性をプロットする（両モデル対応版）"""
    print("\n--- 温度依存性の可視化（両モデル） ---")
    
    # 全モデルで共通の温度リストを取得
    all_temps = set()
    for model_params in model_temperature_params.values():
        all_temps.update(model_params.keys())
    temperatures = sorted(all_temps)
    
    # 各モデルのeps_bg値を収集
    model_eps_bg = {}
    for model_type, temp_params in model_temperature_params.items():
        model_eps_bg[model_type] = [temp_params[T]['eps_bg'] for T in temperatures if T in temp_params]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # サブプロット1: 両モデルのeps_bg比較
    ax1 = axes[0, 0]
    colors = {'H_form': 'red', 'B_form': 'blue'}
    markers = {'H_form': 'o', 'B_form': 's'}
    
    for model_type, eps_values in model_eps_bg.items():
        temps_for_model = [T for T in temperatures if T in model_temperature_params[model_type]]
        ax1.plot(temps_for_model, eps_values, 
                color=colors.get(model_type, 'gray'), 
                marker=markers.get(model_type, 'o'),
                linewidth=2, markersize=8, 
                label=f'{model_type}')
    
    ax1.set_xlabel('温度 (K)', fontsize=12)
    ax1.set_ylabel('背景誘電率 eps_bg', fontsize=12)
    ax1.set_title('背景誘電率の温度依存性（モデル比較）', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11)
    
    # サブプロット2: eps_bgの差分
    ax2 = axes[0, 1]
    if 'H_form' in model_eps_bg and 'B_form' in model_eps_bg:
        common_temps = [T for T in temperatures 
                       if T in model_temperature_params['H_form'] 
                       and T in model_temperature_params['B_form']]
        h_form_eps = [model_temperature_params['H_form'][T]['eps_bg'] for T in common_temps]
        b_form_eps = [model_temperature_params['B_form'][T]['eps_bg'] for T in common_temps]
        eps_diff = [h - b for h, b in zip(h_form_eps, b_form_eps)]
        
        ax2.plot(common_temps, eps_diff, 'go-', linewidth=2, markersize=8)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('温度 (K)', fontsize=12)
        ax2.set_ylabel('Δeps_bg (H_form - B_form)', fontsize=12)
        ax2.set_title('両モデル間のeps_bg差分', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.6)
    else:
        ax2.text(0.5, 0.5, 'モデル比較データなし', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.axis('off')
    
    # ▼▼▼【変更】 2パラメータgammaの結果を表示 ▼▼▼
    
    # サブプロット3: H_formの磁気パラメータサマリー
    ax3 = axes[1, 0]
    if 'H_form' in final_traces:
        h_params = extract_bayesian_parameters(final_traces['H_form'])
        ax3.text(0.1, 0.95, 'H_formモデル 磁気パラメータ:', fontsize=14, transform=ax3.transAxes, weight='bold')
        ax3.text(0.1, 0.80, f'g因子 = {h_params["g_factor"]:.6f}', fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.70, f'B4 = {h_params["B4"]:.6f} K', fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.60, f'B6 = {h_params["B6"]:.6f} K', fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.50, f'a_scale = {h_params["a_scale"]:.6f}', fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.40, f'G0 = {h_params["G0"]:.3e}', fontsize=12, transform=ax3.transAxes)
        if 'gamma_min' in h_params:
            ax3.text(0.1, 0.25, f'gamma_min = {h_params["gamma_min"]:.3e}', fontsize=12, transform=ax3.transAxes)
            ax3.text(0.1, 0.15, f'gamma_other = {h_params["gamma_other"]:.3e}', fontsize=12, transform=ax3.transAxes)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('H_form解析結果', fontsize=14)
    ax3.axis('off')
    
    # サブプロット4: B_formの磁気パラメータサマリー
    ax4 = axes[1, 1]
    if 'B_form' in final_traces:
        b_params = extract_bayesian_parameters(final_traces['B_form'])
        ax4.text(0.1, 0.95, 'B_formモデル 磁気パラメータ:', fontsize=14, transform=ax4.transAxes, weight='bold')
        ax4.text(0.1, 0.80, f'g因子 = {b_params["g_factor"]:.6f}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.70, f'B4 = {b_params["B4"]:.6f} K', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.60, f'B6 = {b_params["B6"]:.6f} K', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.50, f'a_scale = {b_params["a_scale"]:.6f}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.40, f'G0 = {b_params["G0"]:.3e}', fontsize=12, transform=ax4.transAxes)
        if 'gamma_min' in b_params:
            ax4.text(0.1, 0.25, f'gamma_min = {b_params["gamma_min"]:.3e}', fontsize=12, transform=ax4.transAxes)
            ax4.text(0.1, 0.15, f'gamma_other = {b_params["gamma_other"]:.3e}', fontsize=12, transform=ax4.transAxes)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('B_form解析結果', fontsize=14)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'temperature_dependencies_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ 温度依存性比較プロットを保存: temperature_dependencies_comparison.png")

# --- ダミー関数（将来的な機能拡張用）---
def plot_combined_temperature_model_comparison(*args, **kwargs):
    print("⚠️ plot_combined_temperature_model_comparison は未実装です。")

def plot_model_selection_results_temperature(*args, **kwargs):
    print("⚠️ plot_model_selection_results_temperature は未実装です。")

def calculate_temperature_peak_errors(*args, **kwargs):
    print("⚠️ calculate_temperature_peak_errors は未実装です。")
    return {}

def save_peak_analysis_to_csv(*args, **kwargs):
    print("⚠️ save_peak_analysis_to_csv は未実装です。")

# --- メインワークフロー関数 ---
def run_analysis_workflow():
    """
    重み付けを利用した温度依存ベイズ推定ワークフロー。
    1. データをロードし、高周波と全周波数領域のデータセットを作成。
    2. 高周波データから各温度のeps_bgを一度だけフィッティング。
    3. 全周波数データからピーク情報を基に周波数ごとの重み配列を生成。
    4. 重み付けした尤度関数を用いてベイズ推定を実行。
    5. 結果を可視化・保存する。
    """
    # 結果保存ディレクトリを作成（メインプロセスでのみ実行）
    results_dir = create_results_directory(CONFIG)
    print(f"画像は '{results_dir.resolve()}' に保存されます。")
    
    # グローバル変数も更新（他の部分で使用されている場合のため）
    global RESULTS_DIR
    RESULTS_DIR = results_dir
    
    print("🚀 重み付きベイズ推定ワークフローを開始します (Gamma: 2パラメータ版)")
    
    # 1. データの読み込み
    print("\n--- ステップ1: データの読み込みと分割 ---")
    all_data = load_and_split_data_three_regions_temperature(
        file_path=DATA_FILE_PATH,
        sheet_name=DATA_SHEET_NAME
    )
    all_datasets_full_range = all_data['all_full']
    high_freq_datasets = all_data['high_freq']

    if not all_datasets_full_range or not high_freq_datasets:
        print("❌ 必要なデータが読み込めませんでした。処理を終了します。")
        return
        
    # 2. 初回のeps_bgフィッティング（初期値使用）
    print("\n--- ステップ2: 初回eps_bgフィッティング（初期値使用） ---")
    temperature_specific_params = {}
    for dataset in high_freq_datasets:
        temp = dataset['temperature']
        result = fit_eps_bg_only_temperature(dataset, bayesian_params=None)  # 初回は初期値
        if result:
            temperature_specific_params[temp] = result
        else:
            temperature_specific_params[temp] = {'eps_bg': eps_bg_init, 'd': d_fixed, 'temperature': temp}

    # 3. 周波数ごとの重み配列を生成
    print("\n--- ステップ3: 尤度関数のための重み配列を生成 ---")
    analysis_settings = CONFIG['analysis_settings']
    weights_list = [create_frequency_weights(d, analysis_settings) for d in all_datasets_full_range]
    
    # 4. パラメータ更新ループ（ベイズ推定 → eps_bg更新）
    # 各モデルが独立したeps_bgパラメータを持つ
    final_traces = {}
    model_temperature_params = {
        'H_form': temperature_specific_params.copy(),  # H_form用のeps_bg
        'B_form': temperature_specific_params.copy()   # B_form用のeps_bg
    }
    max_iterations = CONFIG['mcmc'].get('max_iterations', 3)  # 更新回数の制限 (デフォルト3回)
    
    for iteration in range(max_iterations):
        print(f"\n{'='*60}")
        print(f"🔄 反復 {iteration + 1}/{max_iterations}: パラメータ更新ループ")
        print(f"{'='*60}")
        
        # 4-1. 各モデルで独立にベイズ推定 → eps_bg更新を実行
        for model_type in ['H_form', 'B_form']:
            print(f"\n{'='*50}")
            print(f"🔬 {model_type}モデルの独立処理")
            print(f"{'='*50}")
            
            # 4-1-1. ベイズ推定
            print(f"\n--- ステップ4-{iteration+1}-1: {model_type}モデルのベイズ推定 ---")
            
            # 各モデルは自分自身の前回の結果のみを事前分布として使用（公平な比較のため）
            model_specific_prior = None
            if iteration > 0 and model_type in final_traces:
                # 2回目以降の反復では、前回の自分自身の結果を使用
                model_specific_prior = extract_bayesian_parameters(final_traces[model_type])
                print(f"  📌 前回の{model_type}の結果を事前分布として使用:")
                print(f"     a_scale = {model_specific_prior['a_scale']:.6f}")
                print(f"     g_factor = {model_specific_prior['g_factor']:.6f}")
                print(f"     B4 = {model_specific_prior['B4']:.6f}")
                print(f"     B6 = {model_specific_prior['B6']:.6f}")
            else:
                print(f"  🔰 初回実行のため、設定ファイルの初期値を事前分布として使用")
            
            # このモデル専用のeps_bgパラメータを使用
            trace = run_temperature_bayesian_fit(
                all_datasets_full_range,
                model_temperature_params[model_type],  # モデル独立のeps_bg
                weights_list,
                results_dir,  # 引数として渡す
                prior_magnetic_params=model_specific_prior,  # モデルごとに独立した事前分布
                model_type=model_type
            )
            
            if trace:
                final_traces[model_type] = trace  # 最新結果を保存
                
                # 事後分布をプロット
                plot_posterior_distributions(trace, f"{model_type}_iter{iteration+1}", results_dir)
                
                # 4-1-2. このモデルの結果でeps_bgを更新（最後の反復でない場合）
                if iteration < max_iterations - 1:
                    print(f"\n--- ステップ4-{iteration+1}-2: {model_type}のeps_bg更新 ---")
                    bayesian_params = extract_bayesian_parameters(trace)
                    print(f"  🔄 {model_type}モデルの結果でeps_bgパラメータを更新:")
                    print(f"     a_scale: {bayesian_params['a_scale']:.6f}")
                    print(f"     g_factor: {bayesian_params['g_factor']:.6f}")
                    print(f"     B4: {bayesian_params['B4']:.6f}")
                    print(f"     B6: {bayesian_params['B6']:.6f}")
                    if 'gamma_min' in bayesian_params:
                         print(f"     gamma_min: {bayesian_params['gamma_min']:.3e}")
                         print(f"     gamma_other: {bayesian_params['gamma_other']:.3e}")
                    
                    # 各温度でeps_bgを再フィッティング（更新されたパラメータ使用）
                    updated_temperature_params = {}
                    for dataset in high_freq_datasets:
                        temp = dataset['temperature']
                        result = fit_eps_bg_only_temperature(
                            dataset, 
                            bayesian_params=bayesian_params  # 更新されたパラメータを使用
                        )
                        if result:
                            updated_temperature_params[temp] = result
                        else:
                            # フォールバック：前回の結果を使用
                            updated_temperature_params[temp] = model_temperature_params[model_type].get(
                                temp, {'eps_bg': eps_bg_init, 'd': d_fixed, 'temperature': temp}
                            )
                    
                    # パラメータの変化をレポート
                    print(f"\n  📊 {model_type}のeps_bgパラメータの変化（反復 {iteration + 1} → {iteration + 2}）:")
                    for temp in sorted(model_temperature_params[model_type].keys()):
                        old_eps = model_temperature_params[model_type][temp]['eps_bg']
                        new_eps = updated_temperature_params[temp]['eps_bg']
                        change = new_eps - old_eps
                        print(f"     温度 {temp}K: {old_eps:.3f} → {new_eps:.3f} (変化: {change:+.3f})")
                    
                    # このモデル専用のeps_bgを更新
                    model_temperature_params[model_type] = updated_temperature_params
            else:
                print(f"  ❌ {model_type}モデルのベイズ推定に失敗しました。")

    # 5. 最終結果の評価と可視化
    if not final_traces:
        print("❌ ベイズ推定が失敗しました。処理を終了します。")
        return
        
    print(f"\n{'='*60}")
    print("🎯 最終結果の評価と可視化")
    print(f"{'='*60}")
    
    # 各モデルのeps_bgパラメータをサマリー表示
    print("\n📊 各モデルの最終eps_bgパラメータ:")
    for model_type in ['H_form', 'B_form']:
        if model_type in model_temperature_params:
            print(f"\n  {model_type}:")
            for temp in sorted(model_temperature_params[model_type].keys()):
                eps_bg = model_temperature_params[model_type][temp]['eps_bg']
                print(f"    温度 {temp}K: eps_bg = {eps_bg:.3f}")
    
    # モデル比較のために、H_formのeps_bgを使用（ダミー関数用）
    # Note: 各モデルは独立したeps_bgを持つ
    temperature_specific_params = model_temperature_params.get('H_form', {})
    
    if len(final_traces) >= 2:
        plot_combined_temperature_model_comparison(all_datasets_full_range, temperature_specific_params, final_traces)
        plot_model_selection_results_temperature(final_traces)
        peak_analysis_results = calculate_temperature_peak_errors(all_datasets_full_range, temperature_specific_params, final_traces)
        save_peak_analysis_to_csv(peak_analysis_results)
    
    # 両モデルのパラメータを保存・可視化
    save_fitting_parameters_to_csv(final_traces, model_temperature_params, results_dir)
    plot_temperature_dependencies(model_temperature_params, final_traces, results_dir)

    print("\n🎉 全ての解析ワークフローが完了しました。")
    print(f"📁 結果は '{results_dir}' に保存されています。")

# --- メインプログラムのエントリーポイント ---
if __name__ == "__main__":
    # Windows環境でのマルチプロセス問題対策
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # すでに設定されている場合は無視
        pass
    
    # 解析ワークフローを実行
    run_analysis_workflow()
