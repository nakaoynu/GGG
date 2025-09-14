# temperature_dependent_bayesian_fitting.py - 温度依存ベイズ推定フィッティング

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
from typing import List, Dict, Any, Tuple, Optional
from scipy.signal import find_peaks
from scipy.spatial import KDTree
from scipy.optimize import curve_fit

# 数値計算の警告を抑制
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(all='ignore')  # NumPyの警告も抑制

# --- 0. 環境設定 ---
if __name__ == "__main__":
    print("--- 0. 環境設定を開始します ---")
    os.environ['PYTENSOR_FLAGS'] = 'optimizer=fast_compile,floatX=float64'
    try:
        import japanize_matplotlib
    except ImportError:
        print("警告: japanize_matplotlib が見つかりません。")
    plt.rcParams['figure.dpi'] = 120
    IMAGE_DIR = pathlib.Path(__file__).parent / "temperature_test"
    IMAGE_DIR.mkdir(exist_ok=True)
    print(f"画像は '{IMAGE_DIR.resolve()}' に保存されます。")

# --- 1. 物理定数とシミュレーション条件 ---
if __name__ == "__main__":
    print("--- 1. 物理定数と初期値を設定します ---")

kB = 1.380649e-23; muB = 9.274010e-24; hbar = 1.054571e-34
c = 299792458; mu0 = 4.0 * np.pi * 1e-7
s = 3.5; N_spin = 24 / 1.238 * 1e27

# 固定磁場（温度依存測定時の条件）
B_FIXED = 9.0  # Tesla

# パラメータの初期値（two_step_iterative_fitting.pyベース）
d_fixed = 157.8e-6  # 膜厚は固定値として使用
eps_bg_init = 14.20
B4_init = 0.000576; B6_init = 0.000050
gamma_init = 0.11e12; a_scale_init = 0.604971; g_factor_init = 2.003147

# --- データファイル設定（グローバル変数による設定の集約化） ---
# 温度依存透過率測定データファイル
DATA_FILE_PATH = "C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\corrected_exp_datasets\\Corrected_Transmittance_Temperature.xlsx"
DATA_SHEET_NAME = "Corrected Data"

# 測定温度条件リスト（Excelファイルの列名に対応）
TEMPERATURE_COLUMNS = ['4K', '30K', '100K', '300K']

# 周波数領域分割の閾値設定
LOW_FREQUENCY_CUTOFF = 0.361505   # THz - ベイズ推定で使用する低周波領域の上限
HIGH_FREQUENCY_CUTOFF = 0.45   # THz - eps_bgフィッティングで使用する高周波領域の下限

# --- 2. 物理モデル関数 ---
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
    
    # === 型チェック強化: gamma_arrayの型安全性確保 ===
    if np.isscalar(gamma_array):
        # 単一値の場合は7つの遷移に対応する配列に変換
        gamma_array = np.full(7, gamma_array)
        print(f"  [TYPE_SAFE] gamma_arrayが単一値でした。7要素配列に変換: {gamma_array}")
    elif not isinstance(gamma_array, np.ndarray):
        gamma_array = np.array(gamma_array)
        print(f"  [TYPE_SAFE] gamma_arrayを配列に変換: {type(gamma_array)}")
    
    # gamma_arrayが0次元配列の場合も修正
    if gamma_array.ndim == 0:
        gamma_array = np.full(7, float(gamma_array))
        print(f"  [TYPE_SAFE] gamma_arrayが0次元配列でした。7要素配列に変換: {gamma_array}")
    
    # gamma_arrayのサイズチェック
    if gamma_array.size == 1:
        gamma_array = np.full(7, gamma_array.item())
        print(f"  [TYPE_SAFE] gamma_arrayがサイズ1でした。7要素配列に変換")
    
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
        print(f"  [DEBUG] gamma_array長さ={len(gamma_array)}, delta_E長さ={len(delta_E)}")
        if len(gamma_array) > len(delta_E):
            gamma_array = gamma_array[:len(delta_E)]
            print(f"  [DEBUG] gamma_arrayを切り詰めました: 長さ={len(gamma_array)}")
        else:
            gamma_array = np.pad(gamma_array, (0, len(delta_E) - len(gamma_array)), 'edge')
            print(f"  [DEBUG] gamma_arrayを拡張しました: 長さ={len(gamma_array)}")
    
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

# --- 3. データ処理と解析ステップ ---
def load_and_split_data_three_regions_temperature(file_path: str, sheet_name: str, 
                                                 low_cutoff: float = LOW_FREQUENCY_CUTOFF, 
                                                 high_cutoff: float = HIGH_FREQUENCY_CUTOFF) -> Dict[str, List[Dict[str, Any]]]:
    """温度依存データを読み込み、低周波・中間・高周波領域に分割する。
    
    Args:
        file_path: Excelファイルのパス
        sheet_name: シート名
        low_cutoff: 低周波領域の上限 (default: 0.361505 THz)
        high_cutoff: 高周波領域の下限 (default: 0.45 THz)
    
    Returns:
        Dict containing:
        - 'low_freq': [~, 0.361505THz] - ベイズ推定用
        - 'mid_freq': [0.361505THz, 0.45THz] - 中間領域（使用しない）
        - 'high_freq': [0.45THz, ~] - eps_bgフィッティング用
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    except Exception as e:
        raise FileNotFoundError(f"Excelファイル '{file_path}' が読み込めません: {e}")
    
    freq_col = 'Frequency (THz)'
    df[freq_col] = pd.to_numeric(df[freq_col], errors='coerce')
    temp_cols = TEMPERATURE_COLUMNS
    
    low_freq_datasets, mid_freq_datasets, high_freq_datasets = [], [], []
    
    for col in temp_cols:
        if col not in df.columns:
            continue
            
        temp_value = float(col.replace('K', ''))
        df_clean = df[[freq_col, col]].dropna()
        freq, trans = df_clean[freq_col].values.astype(np.float64), df_clean[col].values.astype(np.float64)
        
        # 3つの領域にマスクを定義
        low_mask = freq <= low_cutoff
        mid_mask = (freq > low_cutoff) & (freq < high_cutoff)
        high_mask = freq >= high_cutoff
        
        base_data = {'temperature': temp_value, 'b_field': B_FIXED}
        
        # 低周波領域 [~, 0.361505THz] - ベイズ推定用
        if np.any(low_mask):
            min_low, max_low = np.min(trans[low_mask]), np.max(trans[low_mask])
            trans_norm_low = (trans[low_mask] - min_low) / (max_low - min_low) if max_low > min_low else np.full_like(trans[low_mask], 0.5)
            low_freq_datasets.append({**base_data, 'frequency': freq[low_mask], 'transmittance': trans_norm_low, 'omega': freq[low_mask] * 1e12 * 2 * np.pi})
        
        # 中間領域 [0.361505THz, 0.45THz] - 参考用（メインの解析では使用しない）
        if np.any(mid_mask):
            min_mid, max_mid = np.min(trans[mid_mask]), np.max(trans[mid_mask])
            trans_norm_mid = (trans[mid_mask] - min_mid) / (max_mid - min_mid) if max_mid > min_mid else np.full_like(trans[mid_mask], 0.5)
            mid_freq_datasets.append({**base_data, 'frequency': freq[mid_mask], 'transmittance': trans_norm_mid, 'omega': freq[mid_mask] * 1e12 * 2 * np.pi})
        
        # 高周波領域 [0.45THz, ~] - eps_bgフィッティング用
        if np.any(high_mask):
            min_high, max_high = np.min(trans[high_mask]), np.max(trans[high_mask])
            trans_norm_high = (trans[high_mask] - min_high) / (max_high - min_high) if max_high > min_high else np.full_like(trans[high_mask], 0.5)
            high_freq_datasets.append({**base_data, 'frequency': freq[high_mask], 'transmittance': trans_norm_high, 'omega': freq[high_mask] * 1e12 * 2 * np.pi, 'transmittance_full': trans})
    
    print(f"温度依存データ分割結果:")
    print(f"  低周波領域 [~, {low_cutoff}THz]: {len(low_freq_datasets)} データセット")
    print(f"  中間領域 [{low_cutoff}THz, {high_cutoff}THz]: {len(mid_freq_datasets)} データセット")
    print(f"  高周波領域 [{high_cutoff}THz, ~]: {len(high_freq_datasets)} データセット")
    
    return {
        'low_freq': low_freq_datasets, 
        'mid_freq': mid_freq_datasets,
        'high_freq': high_freq_datasets
    }

def fit_eps_bg_only_temperature(dataset: Dict[str, Any], 
                               fixed_params: Optional[Dict[str, float]] = None,
                               G0_from_bayesian: Optional[float] = None) -> Dict[str, float]:
    """各温度で高周波データからeps_bgのみをフィッティングする（他パラメータは固定）"""
    print(f"\n--- 温度 {dataset['temperature']} K の高周波eps_bgフィッティング ---")
    
    # 固定パラメータの取得
    if fixed_params is None:
        fixed_params = {
            'g_factor': g_factor_init,
            'B4': B4_init,
            'B6': B6_init,
            'gamma_scale': 1.0
        }
    
    def magnetic_cavity_model_eps_bg_only(freq_thz, eps_bg_fit):
        """eps_bgのみを変数とする温度依存磁気感受率を考慮した高周波透過率モデル"""
        try:
            omega = freq_thz * 1e12 * 2 * np.pi
            
            # 固定パラメータから値を取得
            g_factor_fit = fixed_params['g_factor']
            B4_fit = fixed_params['B4']
            B6_fit = fixed_params['B6']
            gamma_scale = fixed_params['gamma_scale']
            
            # ハミルトニアンと磁気感受率の計算
            H = get_hamiltonian(B_FIXED, g_factor_fit, B4_fit, B6_fit)
            
            # 高周波用の簡略化されたガンマ（単一値）
            gamma_array = np.full(7, gamma_scale * gamma_init)
            chi_raw = calculate_susceptibility(omega, H, dataset['temperature'], gamma_array)
            
            # 磁気感受率のスケーリング（高周波では小さくなる傾向）
            G0 = mu0 * N_spin * (g_factor_fit * muB)**2 / (2 * hbar) * 0.1  # 高周波用スケーリング係数
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
    
    # eps_bgの初期値候補（温度依存性を考慮：低温→低め、高温→高めからスタート）
    temp = dataset['temperature']
    if temp <= 10:
        # 低温では低めの初期値から開始（効率化）
        initial_eps_bg_values = [eps_bg_init * 0.95, eps_bg_init * 0.90, eps_bg_init, eps_bg_init * 1.05, 
                                12.5, 13.0, 13.5, 14.0, 14.5, 15.0]
        bounds_eps_bg = (10.0, 18.0)
    elif temp <= 100:
        # 中間温度
        initial_eps_bg_values = [eps_bg_init, eps_bg_init * 0.98, eps_bg_init * 1.02, 
                                13.0, 13.5, 14.0, 14.5]
        bounds_eps_bg = (10.0, 17.0)
    else:
        # 高温では高めの初期値から開始（効率化）
        initial_eps_bg_values = [eps_bg_init * 1.05, eps_bg_init * 1.10, eps_bg_init, eps_bg_init * 0.95, 
                                14.5, 15.0, 15.5, 14.0, 13.5, 13.0]
        bounds_eps_bg = (10.0, 18.0)  # 高温では上限を拡張
    
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
                    'd': fixed_params['d'],
                    'temperature': temp
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

# --- 3. データ処理と解析ステップ ---
def fit_single_temperature_cavity_modes(dataset: Dict[str, Any]) -> Dict[str, float]:
    """各温度で独立に高周波データからeps_bgのみをフィッティングする（two_step_iterative_fitting.pyと同様）"""
    print(f"\n--- 温度 {dataset['temperature']} K の高周波eps_bgフィッティング ---")
    
    # 固定パラメータ（two_step_iterative_fitting.pyと同じ設定）
    fixed_params = {
        'g_factor': g_factor_init,
        'B4': B4_init,
        'B6': B6_init,
        'gamma_scale': 1.0
    }
    
    def magnetic_cavity_model_eps_bg_only(freq_thz, eps_bg_fit):
        """eps_bgのみを変数とする温度依存磁気感受率を考慮した高周波透過率モデル"""
        try:
            omega = freq_thz * 1e12 * 2 * np.pi
            
            # 固定パラメータから値を取得
            g_factor_fit = fixed_params['g_factor']
            B4_fit = fixed_params['B4']
            B6_fit = fixed_params['B6']
            gamma_scale = fixed_params['gamma_scale']
            
            # ハミルトニアンと磁気感受率の計算
            H = get_hamiltonian(B_FIXED, g_factor_fit, B4_fit, B6_fit)
            
            # 高周波用の簡略化されたガンマ（単一値）
            gamma_array = np.full(7, gamma_scale * gamma_init)
            chi_raw = calculate_susceptibility(omega, H, dataset['temperature'], gamma_array)
            
            # 磁気感受率のスケーリング（高周波では小さくなる傾向）
            G0 = mu0 * N_spin * (g_factor_fit * muB)**2 / (2 * hbar) * 0.1  # 高周波用スケーリング係数
            chi = G0 * chi_raw
            
            # H_formで透磁率を計算
            mu_r = 1 + chi
            
            return calculate_normalized_transmission(omega, mu_r, d_fixed, eps_bg_fit)
        except Exception as e:
            print(f"    警告: モデル計算エラー {e}")
            return np.ones_like(freq_thz) * 0.5

    # 複数の初期値を試行（温度依存性を考慮）
    success = False
    result = {}
    
    # eps_bgの初期値候補（温度依存性を考慮した改善版：低温→低め、高温→高めからスタート）
    temp = dataset['temperature']
    if temp <= 10:
        # 低温では低めの初期値から開始（効率化）
        initial_eps_bg_values = [eps_bg_init * 0.95, eps_bg_init * 0.90, eps_bg_init, eps_bg_init * 1.05,
                                12.8, 13.2, 13.5, 13.8, 14.0, 14.2]
        bounds_eps_bg = (11.0, 18.5)  # 範囲を拡張
    elif temp <= 100:
        # 中間温度（改善された初期値選択）
        initial_eps_bg_values = [eps_bg_init, eps_bg_init * 0.98, eps_bg_init * 1.02, eps_bg_init * 1.05,
                                13.0, 13.5, 13.8, 14.0, 14.3, 14.5]
        bounds_eps_bg = (11.5, 17.5)
    else:
        # 高温では高めの初期値から開始（効率化）
        initial_eps_bg_values = [eps_bg_init * 1.08, eps_bg_init * 1.05, eps_bg_init, eps_bg_init * 0.96,
                                15.5, 15.2, 14.8, 14.5, 14.0, 13.5]
        bounds_eps_bg = (12.0, 18.0)  # 高温では上限を拡張
    
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
                    'd': d_fixed,  # 固定値を使用
                    'temperature': temp
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

class TemperatureMagneticModelOp(Op):
    """温度依存の低周波領域の磁気パラメータを推定するためのPyMC Op（温度依存gamma対応）。"""
    def __init__(self, datasets: List[Dict[str, Any]], temperature_specific_params: Dict[float, Dict[str, float]], model_type: str):
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
            
            # 温度に対応するgamma部分を抽出（7つの遷移）
            gamma_end_idx = gamma_start_idx + 7
            gamma_for_temp = gamma_concat[gamma_start_idx:gamma_end_idx]
            gamma_start_idx = gamma_end_idx
            
            H = get_hamiltonian(data['b_field'], g_factor, B4, B6)
            chi_raw = calculate_susceptibility(data['omega'], H, data['temperature'], gamma_for_temp)
            G0 = a_scale * mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
            chi = G0 * chi_raw
            
            # モデルタイプに応じて透磁率を計算
            if self.model_type == 'H_form':
                mu_r = 1 + chi
            else:  # B_form
                mu_r = 1 / (1 - chi)
            
            predicted_y = calculate_normalized_transmission(data['omega'], mu_r, d_fixed, eps_bg_fixed)
            full_predicted_y.append(predicted_y)
        
        output_storage[0][0] = np.concatenate(full_predicted_y)

def run_temperature_bayesian_fit(datasets: List[Dict[str, Any]], 
                                temperature_specific_params: Dict[float, Dict[str, float]], 
                                prior_magnetic_params: Optional[Dict[str, float]] = None, 
                                model_type: str = 'H_form') -> Optional[az.InferenceData]:
    """温度毎の固定eps_bgを使用してベイズ推定を実行する"""
    print(f"\n--- 温度別固定eps_bgによるベイズ推定 (モデル: {model_type}) ---")
    trans_observed = np.concatenate([d['transmittance'] for d in datasets])
    
    with pm.Model() as model:
        # 事前分布の設定（より制約的に）
        if prior_magnetic_params:
            # 前回のベイズ推定結果を事前分布として使用
            a_scale = pm.TruncatedNormal('a_scale', mu=prior_magnetic_params['a_scale'], sigma=0.2, lower=0.3, upper=3.0)
            g_factor = pm.TruncatedNormal('g_factor', mu=prior_magnetic_params['g_factor'], sigma=0.03, lower=1.98, upper=2.05)
            # B4、B6の事前分布をより制約的に（収束改善のため）
            B4 = pm.TruncatedNormal('B4', mu=prior_magnetic_params['B4'], 
                                   sigma=min(abs(prior_magnetic_params['B4'])*0.2 + 0.0002, 0.001), 
                                   lower=-0.002, upper=0.002)
            B6 = pm.TruncatedNormal('B6', mu=prior_magnetic_params['B6'], 
                                   sigma=min(abs(prior_magnetic_params['B6'])*0.2 + 0.00002, 0.0001), 
                                   lower=-0.0005, upper=0.0005)
            print(f"前回の推定結果を事前分布として使用:")
            print(f"  a_scale事前分布中心: {prior_magnetic_params['a_scale']:.3f}")
            print(f"  g_factor事前分布中心: {prior_magnetic_params['g_factor']:.3f}")
            print(f"  B4事前分布中心: {prior_magnetic_params['B4']:.5f}")
            print(f"  B6事前分布中心: {prior_magnetic_params['B6']:.6f}")
        else:
            # 初回のデフォルト事前分布（より制約的に）
            a_scale = pm.TruncatedNormal('a_scale', mu=a_scale_init, sigma=0.3, lower=0.3, upper=3.0)
            g_factor = pm.TruncatedNormal('g_factor', mu=g_factor_init, sigma=0.02, lower=1.98, upper=2.05)
            # 初回もB4、B6をより制約的に
            B4 = pm.TruncatedNormal('B4', mu=B4_init, sigma=0.0003, lower=-0.002, upper=0.002)
            B6 = pm.TruncatedNormal('B6', mu=B6_init, sigma=0.00003, lower=-0.0005, upper=0.0005)
            print("初回のデフォルト事前分布を使用（B4、B6制約強化）")
        
        # γパラメータの温度依存性実装（改良版）
        # 各温度条件での基準温度（ベースライン）
        base_temp = 4.0  # K
        temp_list = sorted(list(set([d['temperature'] for d in datasets])))
        n_temps = len(temp_list)
        
        # 基準温度での基本gamma分布
        log_gamma_mu_base = pm.Normal('log_gamma_mu_base', mu=np.log(gamma_init), sigma=0.8)
        log_gamma_sigma_base = pm.HalfNormal('log_gamma_sigma_base', sigma=0.5)
        log_gamma_offset_base = pm.Normal('log_gamma_offset_base', mu=0, sigma=0.5, shape=7)
        
        # 温度依存性パラメータ（温度増加に伴うgammaの変化率）
        temp_gamma_slope = pm.Normal('temp_gamma_slope', mu=0.001, sigma=0.0005)  # 温度依存性係数
        temp_gamma_nonlinear = pm.Normal('temp_gamma_nonlinear', mu=0.0, sigma=0.00001)  # 非線形項
        
        # 各温度でのgamma計算
        gamma_all_temps = []
        for temp in temp_list:
            # 温度差に基づく補正項
            temp_diff = temp - base_temp
            # 線形 + 非線形の温度依存性
            temp_correction = temp_gamma_slope * temp_diff + temp_gamma_nonlinear * temp_diff**2
            
            # 各温度での修正log_gamma_mu
            log_gamma_mu_temp = log_gamma_mu_base + temp_correction
            
            # 各温度でのgamma値を計算
            gamma_temp = pt.exp(log_gamma_mu_temp + log_gamma_offset_base * log_gamma_sigma_base)
            gamma_all_temps.append(gamma_temp)
        
        # 温度別データセットに対応するgammaを選択
        gamma_final = []
        for dataset in datasets:
            temp_idx = temp_list.index(dataset['temperature'])
            gamma_final.append(gamma_all_temps[temp_idx])
        
        # 最終的なgamma（全データセットの結合）
        gamma = pt.concatenate(gamma_final, axis=0)
        
        op = TemperatureMagneticModelOp(datasets, temperature_specific_params, model_type)
        mu = op(a_scale, gamma, g_factor, B4, B6)
        
        # 観測ノイズの事前分布（より制約的に）
        sigma = pm.HalfNormal('sigma', sigma=0.05)
        
        # 観測モデル
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=trans_observed)
        
        # サンプリング設定
        cpu_count = os.cpu_count() or 4
        try:
            print("ベイズサンプリングを開始します...")
            # より安定したサンプリング設定
            trace = pm.sample(2000,  # サンプル数を増加
                              tune=2000,  # チューニング数を大幅増加
                              chains=4,   # チェーン数を増やして収束診断の信頼性向上
                              cores=min(cpu_count, 4), 
                              target_accept=0.9,  # 受諾率を上げて数値安定性向上
                              init='jitter+adapt_diag_grad',  # より高度な初期化
                              idata_kwargs={"log_likelihood": True}, 
                              random_seed=42,
                              progressbar=True,
                              return_inferencedata=True)
            
            # 収束診断を詳細に実行
            print("\n--- 収束診断 ---")
            summary = az.summary(trace, var_names=['a_scale', 'g_factor', 'B4', 'B6'])
            max_rhat = summary['r_hat'].max()
            min_ess_bulk = summary['ess_bulk'].min()
            min_ess_tail = summary['ess_tail'].min()
            
            print(f"最大 r_hat: {max_rhat:.4f} (< 1.01 が望ましい)")
            print(f"最小 ess_bulk: {min_ess_bulk:.0f} (> 400 が望ましい)")
            print(f"最小 ess_tail: {min_ess_tail:.0f} (> 400 が望ましい)")
            
            # 収束判定
            convergence_ok = (max_rhat < 1.01 and min_ess_bulk > 400 and min_ess_tail > 400)
            
            if not convergence_ok:
                if max_rhat > 1.01:
                    print("⚠️ 警告: r_hat > 1.01 - 収束に問題があります")
                if min_ess_bulk < 400:
                    print("⚠️ 警告: ess_bulk < 400 - 有効サンプルサイズが不足")
                if min_ess_tail < 400:
                    print("⚠️ 警告: ess_tail < 400 - 分布の裾の推定が不安定")
            else:
                print("✅ 収束診断: 良好")
                
        except Exception as e:
            print(f"高精度サンプリングに失敗: {e}")
            print("中精度設定でリトライします...")
            try:
                trace = pm.sample(2000, 
                                  tune=2000, 
                                  chains=4, 
                                  cores=min(cpu_count, 2), 
                                  target_accept=0.85,
                                  init='jitter+adapt_diag_grad',  # より高度な初期化
                                  idata_kwargs={"log_likelihood": True}, 
                                  random_seed=42,
                                  progressbar=True,
                                  return_inferencedata=True)
            except Exception as e2:
                print(f"中精度設定も失敗: {e2}")
                print("最小設定でリトライします...")
                trace = pm.sample(2000, 
                                  tune=3000, 
                                  chains=2, 
                                  cores=1,  # シングルコアで実行
                                  target_accept=0.8,
                                  idata_kwargs={"log_likelihood": True}, 
                                  random_seed=42,
                                  progressbar=True,
                                  return_inferencedata=True)

        # サンプリング後にlog_likelihoodが存在するかチェックして、なければ計算
        with model:
            if "log_likelihood" not in trace:
                trace = pm.compute_log_likelihood(trace)
                # === 型アノテーション修正: 戻り値の型一貫性確保 ===
                if not isinstance(trace, az.InferenceData):
                    # Datasetの場合はInferenceDataに変換（代替方法）
                    print(f"  [TYPE_SAFE] 型変換が必要: {type(trace)}")
                assert isinstance(trace, az.InferenceData)
    
    print("----------------------------------------------------")
    print("▶ 温度依存ベイズ推定結果 (サマリー):")
    # 温度依存gamma実装により変数名を修正
    try:
        summary = az.summary(trace, var_names=['a_scale', 'g_factor', 'B4', 'B6', 'log_gamma_mu_base', 'temp_gamma_slope', 'temp_gamma_nonlinear'])
        print(summary)
    except KeyError as e:
        print(f"サマリー表示エラー: {e}")
        # 利用可能な変数名を表示（型安全版）
        try:
            print("利用可能な変数:", list(trace["posterior"].keys()))
        except Exception:
            print("posterior データにアクセスできませんでした")
    print("----------------------------------------------------")
    return trace

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
        result['log_gamma_mu_base'] = posterior['log_gamma_mu_base'].mean(dim=('chain', 'draw')).values.item()
        result['log_gamma_offset_base'] = posterior['log_gamma_offset_base'].mean(dim=('chain', 'draw')).values
        result['log_gamma_sigma_base'] = posterior['log_gamma_sigma_base'].mean(dim=('chain', 'draw')).values.item()
        result['temp_gamma_slope'] = posterior['temp_gamma_slope'].mean(dim=('chain', 'draw')).values.item()
        result['temp_gamma_nonlinear'] = posterior['temp_gamma_nonlinear'].mean(dim=('chain', 'draw')).values.item()
    except KeyError:
        # 旧版gammaパラメータとの互換性
        pass
    
    return result

def load_data_full_range_temperature(file_path: str, sheet_name: str) -> List[Dict[str, Any]]:
    """全周波数範囲の温度依存データを読み込む。"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    except Exception as e:
        raise FileNotFoundError(f"Excelファイル '{file_path}' が読み込めません: {e}")
    
    freq_col = 'Frequency (THz)'
    df[freq_col] = pd.to_numeric(df[freq_col], errors='coerce')
    temp_cols = TEMPERATURE_COLUMNS  # グローバル変数を使用
    
    all_datasets = []
    for col in temp_cols:
        if col not in df.columns:
            continue
            
        temp_value = float(col.replace('K', ''))
        df_clean = df[[freq_col, col]].dropna()
        freq, trans = df_clean[freq_col].values.astype(np.float64), df_clean[col].values.astype(np.float64)
        base_data = {'temperature': temp_value, 'b_field': B_FIXED}
        all_datasets.append({**base_data, 'frequency': freq, 'transmittance_full': trans, 'omega': freq * 1e12 * 2 * np.pi})
    
    return all_datasets

def plot_temperature_results(all_datasets: List[Dict[str, Any]], 
                           temperature_specific_params: Dict[float, Dict[str, float]], 
                           bayesian_trace: az.InferenceData,
                           model_type: str = 'H_form',
                           n_samples: int = 200):
    """温度依存フィッティング結果を95%信用区間と共に可視化する"""
    print(f"\n--- 温度依存フィッティング結果を可視化中 ({model_type}) ---")
    
    posterior = bayesian_trace["posterior"]
    mean_a_scale = float(posterior['a_scale'].mean())
    mean_g_factor = float(posterior['g_factor'].mean())
    mean_B4 = float(posterior['B4'].mean())
    mean_B6 = float(posterior['B6'].mean())
    
    # 温度依存gammaパラメータの平均値を取得
    mean_log_gamma_mu_base = float(posterior['log_gamma_mu_base'].mean())
    mean_temp_gamma_slope = float(posterior['temp_gamma_slope'].mean())
    mean_temp_gamma_nonlinear = float(posterior['temp_gamma_nonlinear'].mean())
    mean_log_gamma_sigma_base = float(posterior['log_gamma_sigma_base'].mean())
    mean_log_gamma_offset_base = posterior['log_gamma_offset_base'].mean().values
    
    G0 = mean_a_scale * mu0 * N_spin * (mean_g_factor * muB)**2 / (2 * hbar)
    
    num_conditions = len(all_datasets)
    fig, axes = plt.subplots(1, num_conditions, figsize=(12 * num_conditions, 8), sharey=True)
    if num_conditions == 1: 
        axes = [axes]

    for i, data in enumerate(all_datasets):
        ax = axes[i]
        temperature = data['temperature']
        
        # 該当温度の固定パラメータを取得
        if temperature in temperature_specific_params:
            d_fixed = temperature_specific_params[temperature]['d']
            eps_bg_fixed = temperature_specific_params[temperature]['eps_bg']
        else:
            continue
        
        # 全周波数範囲での予測計算
        freq_plot = np.linspace(data['frequency'].min(), data['frequency'].max(), 500)
        omega_plot = freq_plot * 1e12 * 2 * np.pi
        
        # --- 信用区間のための計算 ---
        total_samples = posterior.chain.size * posterior.draw.size
        indices = np.random.choice(total_samples, min(n_samples, total_samples), replace=False)
        predictions = []
        
        for idx in indices:
            chain_idx = idx // posterior.draw.size
            draw_idx = idx % posterior.draw.size
            
            a_scale_sample = float(posterior['a_scale'].isel(chain=chain_idx, draw=draw_idx))
            g_factor_sample = float(posterior['g_factor'].isel(chain=chain_idx, draw=draw_idx))
            B4_sample = float(posterior['B4'].isel(chain=chain_idx, draw=draw_idx))
            B6_sample = float(posterior['B6'].isel(chain=chain_idx, draw=draw_idx))
            
            # 該当温度のgamma配列を温度依存パラメータから計算
            log_gamma_mu_base_sample = float(posterior['log_gamma_mu_base'].isel(chain=chain_idx, draw=draw_idx))
            temp_gamma_slope_sample = float(posterior['temp_gamma_slope'].isel(chain=chain_idx, draw=draw_idx))
            temp_gamma_nonlinear_sample = float(posterior['temp_gamma_nonlinear'].isel(chain=chain_idx, draw=draw_idx))
            log_gamma_sigma_base_sample = float(posterior['log_gamma_sigma_base'].isel(chain=chain_idx, draw=draw_idx))
            log_gamma_offset_base_sample = posterior['log_gamma_offset_base'].isel(chain=chain_idx, draw=draw_idx).values
            
            base_temp = 4.0
            temp_diff = temperature - base_temp
            log_gamma_mu_temp_sample = (log_gamma_mu_base_sample + 
                                      temp_gamma_slope_sample * temp_diff + 
                                      temp_gamma_nonlinear_sample * temp_diff**2)
            gamma_sample = np.exp(log_gamma_mu_temp_sample + 
                                log_gamma_offset_base_sample * log_gamma_sigma_base_sample)
            
            H_sample = get_hamiltonian(B_FIXED, g_factor_sample, B4_sample, B6_sample)
            chi_raw_sample = calculate_susceptibility(omega_plot, H_sample, temperature, gamma_sample)
            G0_sample = a_scale_sample * mu0 * N_spin * (g_factor_sample * muB)**2 / (2 * hbar)
            chi_sample = G0_sample * chi_raw_sample
            
            # モデルタイプに応じて透磁率を計算
            if model_type == 'H_form':
                mu_r_sample = 1 + chi_sample
            else:  # B_form
                mu_r_sample = 1 / (1 - chi_sample)
            
            pred_sample = calculate_normalized_transmission(omega_plot, mu_r_sample, d_fixed, eps_bg_fixed)
            predictions.append(pred_sample)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        ci_lower, ci_upper = np.percentile(predictions, [2.5, 97.5], axis=0)
        
        # 平均パラメータでの予測計算も追加
        base_temp = 4.0
        temp_diff = temperature - base_temp
        log_gamma_mu_temp_mean = (mean_log_gamma_mu_base + 
                                 mean_temp_gamma_slope * temp_diff + 
                                 mean_temp_gamma_nonlinear * temp_diff**2)
        gamma_mean = np.exp(log_gamma_mu_temp_mean + 
                           mean_log_gamma_offset_base * mean_log_gamma_sigma_base)
        
        H_mean = get_hamiltonian(B_FIXED, mean_g_factor, mean_B4, mean_B6)
        chi_raw_mean = calculate_susceptibility(omega_plot, H_mean, temperature, gamma_mean)
        chi_mean = G0 * chi_raw_mean
        
        if model_type == 'H_form':
            mu_r_mean = 1 + chi_mean
        else:  # B_form
            mu_r_mean = 1 / (1 - chi_mean)
        
        pred_mean = calculate_normalized_transmission(omega_plot, mu_r_mean, d_fixed, eps_bg_fixed)
        
        # 実験データの正規化
        min_exp, max_exp = data['transmittance_full'].min(), data['transmittance_full'].max()
        trans_norm_full = (data['transmittance_full'] - min_exp) / (max_exp - min_exp)
        
        # プロット（平均パラメータでの予測を使用）
        color = 'red' if model_type == 'H_form' else 'blue'
        ax.plot(freq_plot, pred_mean, color=color, lw=2.5, label=f'平均予測 ({model_type})')
        ax.fill_between(freq_plot, ci_lower, ci_upper, color=color, alpha=0.3, label='95%信用区間')
        ax.scatter(data['frequency'], trans_norm_full, color='black', s=25, alpha=0.6, label='実験データ')
        
        # 低周波/高周波領域の境界線
        ax.axvline(x=0.361505, color='blue', linestyle='--', linewidth=2, alpha=0.7, 
                  label='低周波境界 (0.361505 THz)' if i == 0 else None)
        ax.axvline(x=0.45, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                  label='高周波境界 (0.45 THz)' if i == 0 else None)
        
        ax.set_title(f"温度 {temperature} K (eps_bg={eps_bg_fixed:.4f}, {model_type})", fontsize=14)
        ax.set_xlabel('周波数 (THz)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    axes[0].set_ylabel('正規化透過率', fontsize=12)
    fig.suptitle(f'温度依存フィッティング結果: 温度別背景誘電率 ({model_type})', fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(IMAGE_DIR / f'temperature_fitting_results_{model_type}.png')
    # plt.show()
    
    print("温度依存フィッティング結果の可視化が完了しました。")

def plot_combined_temperature_model_comparison(all_datasets: List[Dict[str, Any]], 
                                             temperature_specific_params: Dict[float, Dict[str, float]], 
                                             traces: Dict[str, az.InferenceData], 
                                             n_samples: int = 200):
    """H_formとB_formを2行2列レイアウトで統合比較プロット"""
    print("\n--- H_form と B_form の温度依存統合比較プロット (2x2レイアウト) ---")
    
    # 2行2列のレイアウト設定（4つの温度条件）
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharey=True, sharex=True)
    axes = axes.flatten()  # 1次元配列に変換
    
    model_results = {}
    
    # 各モデルの結果を事前計算
    for model_type, trace in traces.items():
        print(f"  {model_type}モデルの予測を計算中...")
        posterior = trace["posterior"]
        mean_a_scale = float(posterior['a_scale'].mean())
        mean_g_factor = float(posterior['g_factor'].mean())
        mean_B4 = float(posterior['B4'].mean())
        mean_B6 = float(posterior['B6'].mean())
        
        # 温度依存gammaパラメータの平均値を取得
        mean_log_gamma_mu_base = float(posterior['log_gamma_mu_base'].mean())
        mean_temp_gamma_slope = float(posterior['temp_gamma_slope'].mean())
        mean_temp_gamma_nonlinear = float(posterior['temp_gamma_nonlinear'].mean())
        mean_log_gamma_sigma_base = float(posterior['log_gamma_sigma_base'].mean())
        mean_log_gamma_offset_base = posterior['log_gamma_offset_base'].mean().values
        
        G0 = mean_a_scale * mu0 * N_spin * (mean_g_factor * muB)**2 / (2 * hbar)
        
        model_results[model_type] = {
            'mean_params': {
                'a_scale': mean_a_scale, 'g_factor': mean_g_factor, 
                'B4': mean_B4, 'B6': mean_B6, 'G0': G0
            },
            'temp_gamma_params': {
                'log_gamma_mu_base': mean_log_gamma_mu_base,
                'temp_gamma_slope': mean_temp_gamma_slope,
                'temp_gamma_nonlinear': mean_temp_gamma_nonlinear,
                'log_gamma_sigma_base': mean_log_gamma_sigma_base,
                'log_gamma_offset_base': mean_log_gamma_offset_base
            }
        }
    
    # 4つの温度条件をプロット
    for i, data in enumerate(all_datasets):
        if i >= 4:  # 最大4つまで
            break
            
        ax = axes[i]
        temperature = data['temperature']
        
        # 該当温度の固定パラメータを取得
        if temperature in temperature_specific_params:
            eps_bg_fixed = temperature_specific_params[temperature]['eps_bg']
        else:
            eps_bg_fixed = eps_bg_init
            print(f"  警告: 温度 {temperature} K のeps_bgが見つかりません。初期値を使用。")
        
        # 全周波数範囲での予測計算
        freq_plot = np.linspace(data['frequency'].min(), data['frequency'].max(), 500)
        omega_plot = freq_plot * 1e12 * 2 * np.pi
        
        # 実験データの正規化
        min_exp, max_exp = data['transmittance_full'].min(), data['transmittance_full'].max()
        trans_norm_full = (data['transmittance_full'] - min_exp) / (max_exp - min_exp)
        
        # 実験データのプロット
        ax.scatter(data['frequency'], trans_norm_full, color='black', s=35, alpha=0.8, 
                  label='実験データ', zorder=10)
        
        # 各モデルの予測をプロット
        colors = {'H_form': 'red', 'B_form': 'blue'}
        line_styles = {'H_form': '-', 'B_form': '--'}
        
        for model_type, results in model_results.items():
            mean_params = results['mean_params']
            temp_gamma_params = results['temp_gamma_params']
            trace = traces[model_type]
            posterior = trace["posterior"]
            
            # 該当温度のgamma配列を計算
            base_temp = 4.0
            temp_diff = temperature - base_temp
            log_gamma_mu_temp = (temp_gamma_params['log_gamma_mu_base'] + 
                               temp_gamma_params['temp_gamma_slope'] * temp_diff + 
                               temp_gamma_params['temp_gamma_nonlinear'] * temp_diff**2)
            gamma_array = np.exp(log_gamma_mu_temp + 
                               temp_gamma_params['log_gamma_offset_base'] * temp_gamma_params['log_gamma_sigma_base'])
            
            # 95%信用区間の計算
            total_samples = posterior['a_scale'].size
            indices = np.random.choice(total_samples, min(n_samples, total_samples), replace=False)
            
            predictions = []
            for idx in indices:
                # サンプルされたパラメータ
                a_scale_sample = float(posterior['a_scale'].values.flatten()[idx])
                g_factor_sample = float(posterior['g_factor'].values.flatten()[idx])
                B4_sample = float(posterior['B4'].values.flatten()[idx])
                B6_sample = float(posterior['B6'].values.flatten()[idx])
                
                # 該当温度のgammaを計算
                log_gamma_mu_base_sample = float(posterior['log_gamma_mu_base'].values.flatten()[idx])
                temp_gamma_slope_sample = float(posterior['temp_gamma_slope'].values.flatten()[idx])
                temp_gamma_nonlinear_sample = float(posterior['temp_gamma_nonlinear'].values.flatten()[idx])
                log_gamma_sigma_base_sample = float(posterior['log_gamma_sigma_base'].values.flatten()[idx])
                
                # === インデックス安全性: log_gamma_offset_baseの配列アクセス防護 ===
                log_gamma_offset_base_flat = posterior['log_gamma_offset_base'].values.reshape(-1, 7)
                if idx < log_gamma_offset_base_flat.shape[0]:
                    log_gamma_offset_base_sample = log_gamma_offset_base_flat[idx]
                else:
                    # 範囲外の場合は循環インデックスを使用
                    safe_idx = idx % log_gamma_offset_base_flat.shape[0]
                    log_gamma_offset_base_sample = log_gamma_offset_base_flat[safe_idx]
                    print(f"  [INDEX_SAFE] インデックス範囲外 idx={idx}, shape={log_gamma_offset_base_flat.shape[0]}, 安全インデックス={safe_idx}を使用")
                
                log_gamma_mu_temp_sample = (log_gamma_mu_base_sample + 
                                          temp_gamma_slope_sample * temp_diff + 
                                          temp_gamma_nonlinear_sample * temp_diff**2)
                gamma_sample = np.exp(log_gamma_mu_temp_sample + 
                                    log_gamma_offset_base_sample * log_gamma_sigma_base_sample)
                
                # === 型チェック強化: gamma_sampleが確実に7要素の配列になるように修正 ===
                if np.isscalar(gamma_sample):
                    gamma_sample = np.full(7, gamma_sample)
                    print(f"  [TYPE_SAFE] gamma_sampleが単一値でした。7要素配列に変換 (温度{temperature}K, idx={idx})")
                elif gamma_sample.ndim == 0:
                    gamma_sample = np.full(7, float(gamma_sample))
                    print(f"  [TYPE_SAFE] gamma_sampleが0次元配列でした。7要素配列に変換 (温度{temperature}K, idx={idx})")
                elif len(gamma_sample) != 7:
                    original_len = len(gamma_sample)
                    if len(gamma_sample) > 7:
                        gamma_sample = gamma_sample[:7]
                    else:
                        gamma_sample = np.pad(gamma_sample, (0, 7 - len(gamma_sample)), 'edge')
                    print(f"  [TYPE_SAFE] gamma_sampleサイズ調整: {original_len} → 7 (温度{temperature}K, idx={idx})")
                
                G0_sample = a_scale_sample * mu0 * N_spin * (g_factor_sample * muB)**2 / (2 * hbar)
                
                # 予測計算
                H_sample = get_hamiltonian(B_FIXED, g_factor_sample, B4_sample, B6_sample)
                chi_raw_sample = calculate_susceptibility(omega_plot, H_sample, temperature, gamma_sample)
                chi_sample = G0_sample * chi_raw_sample
                
                if model_type == 'H_form':
                    mu_r_sample = 1 + chi_sample
                else:  # B_form
                    mu_r_sample = 1 / (1 - chi_sample)
                
                predicted_trans_sample = calculate_normalized_transmission(omega_plot, mu_r_sample, d_fixed, eps_bg_fixed)
                predictions.append(predicted_trans_sample)
            
            # 統計的要約
            predictions = np.array(predictions)
            mean_prediction = np.mean(predictions, axis=0)
            ci_lower = np.percentile(predictions, 2.5, axis=0)
            ci_upper = np.percentile(predictions, 97.5, axis=0)
            
            # 平均パラメータでの予測線も計算
            # 該当温度のgamma配列を温度依存パラメータから計算
            base_temp = 4.0
            temp_diff = temperature - base_temp
            log_gamma_mu_temp = (temp_gamma_params['log_gamma_mu_base'] + 
                               temp_gamma_params['temp_gamma_slope'] * temp_diff + 
                               temp_gamma_params['temp_gamma_nonlinear'] * temp_diff**2)
            gamma_array = np.exp(log_gamma_mu_temp + 
                               temp_gamma_params['log_gamma_offset_base'] * temp_gamma_params['log_gamma_sigma_base'])
            
            # === 型チェック強化: 平均gamma_arrayが確実に7要素の配列になるように修正 ===
            if np.isscalar(gamma_array):
                gamma_array = np.full(7, gamma_array)
                print(f"  [TYPE_SAFE] 平均gamma_arrayが単一値でした。7要素配列に変換 (温度{temperature}K, {model_type})")
            elif gamma_array.ndim == 0:
                gamma_array = np.full(7, float(gamma_array))
                print(f"  [TYPE_SAFE] 平均gamma_arrayが0次元配列でした。7要素配列に変換 (温度{temperature}K, {model_type})")
            elif len(gamma_array) != 7:
                original_len = len(gamma_array)
                if len(gamma_array) > 7:
                    gamma_array = gamma_array[:7]
                else:
                    gamma_array = np.pad(gamma_array, (0, 7 - len(gamma_array)), 'edge')
                print(f"  [TYPE_SAFE] 平均gamma_arrayサイズ調整: {original_len} → 7 (温度{temperature}K, {model_type})")
            
            H_mean = get_hamiltonian(B_FIXED, mean_params['g_factor'], mean_params['B4'], mean_params['B6'])
            chi_raw_mean = calculate_susceptibility(omega_plot, H_mean, temperature, gamma_array)
            chi_mean = mean_params['G0'] * chi_raw_mean
            
            if model_type == 'H_form':
                mu_r_mean = 1 + chi_mean
            else:  # B_form
                mu_r_mean = 1 / (1 - chi_mean)
            
            predicted_trans_mean = calculate_normalized_transmission(omega_plot, mu_r_mean, d_fixed, eps_bg_fixed)
            
            # 平均値の予測線（平均パラメータから計算）
            color = colors[model_type]
            line_style = line_styles[model_type]
            ax.plot(freq_plot, predicted_trans_mean, color=color, linestyle=line_style, 
                   linewidth=3, label=f'{model_type}予測', alpha=0.9)
            
            # 95%信用区間
            ax.fill_between(freq_plot, ci_lower, ci_upper, color=color, alpha=0.2, 
                           label=f'{model_type} 95%信用区間' if i == 0 else None)
        
        # 低周波/高周波領域の境界線
        ax.axvline(x=LOW_FREQUENCY_CUTOFF, color='blue', linestyle=':', linewidth=2, alpha=0.8, 
                  label='低周波境界' if i == 0 else None)
        ax.axvline(x=HIGH_FREQUENCY_CUTOFF, color='red', linestyle=':', linewidth=2, alpha=0.8, 
                  label='高周波境界' if i == 0 else None)
        
        # 軸とタイトルの設定
        ax.set_title(f'温度 {temperature} K, (eps_bg={eps_bg_fixed:.4f})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # 凡例（最初のサブプロットのみ）
        if i == 0:
            ax.legend(loc='upper right', fontsize=16)
        
        # x軸とy軸のラベル
        if i >= 2:  # 下段
            ax.set_xlabel('周波数 (THz)', fontsize=16)
        if i % 2 == 0:  # 左列
            ax.set_ylabel('正規化透過率', fontsize=16)
    
    # 空のサブプロットを非表示（4個より少ない場合）
    for j in range(len(all_datasets), 4):
        axes[j].set_visible(False)
    
    plt.suptitle('図4 温度依存H_form vs B_formモデル比較', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=(0, 0.02, 1, 0.96))
    plt.subplots_adjust(wspace=0.15, hspace=0.25)
    
    plt.savefig(IMAGE_DIR / 'combined_temperature_model_comparison_2x2.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("H_form vs B_form 統合比較プロット (2×2レイアウト) が完成しました。")

def plot_temperature_dependencies(temperature_specific_params: Dict[float, Dict[str, float]], 
                                bayesian_trace: az.InferenceData):
    """温度依存性をプロットする"""
    print("\n--- 温度依存性の可視化 ---")
    
    temperatures = sorted(temperature_specific_params.keys())
    eps_bg_values = [temperature_specific_params[T]['eps_bg'] for T in temperatures]
    d_fixed_um = d_fixed * 1e6  # μm単位の固定膜厚
    
    # 磁気パラメータを抽出
    magnetic_params = extract_bayesian_parameters(bayesian_trace)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # eps_bg の温度依存性
    ax1.plot(temperatures, eps_bg_values, 'ro-', linewidth=2, markersize=8, label='背景誘電率')
    ax1.set_xlabel('温度 (K)', fontsize=12)
    ax1.set_ylabel('背景誘電率 eps_bg', fontsize=12)
    ax1.set_title('背景誘電率の温度依存性', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # 値をテキストで表示
    for T, eps in zip(temperatures, eps_bg_values):
        ax1.annotate(f'{eps:.3f}', (T, eps), textcoords="offset points", xytext=(0,10), ha='center')
    
    # 膜厚の表示（固定値）
    ax2.axhline(y=d_fixed_um, color='blue', linewidth=3, label=f'膜厚（固定値）')
    ax2.scatter(temperatures, [d_fixed_um]*len(temperatures), color='blue', s=80, zorder=5)
    ax2.set_xlabel('温度 (K)', fontsize=12)
    ax2.set_ylabel('膜厚 (μm)', fontsize=12)
    ax2.set_title('膜厚（固定値）', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    ax2.set_ylim(d_fixed_um*0.95, d_fixed_um*1.05)  # 固定値周辺を表示
    
    # 固定値をテキストで表示
    ax2.text(np.mean(temperatures), d_fixed_um + d_fixed_um*0.01, 
            f'{d_fixed_um:.2f} μm', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 磁気パラメータの表示（温度非依存として）
    param_names = ['g_factor', 'B4', 'B6', 'G0']
    param_values = [magnetic_params[name] for name in param_names]
    param_labels = ['g因子', 'B4 (K)', 'B6 (K)', 'G0']
    
    ax3.barh(param_labels, param_values, color=['red', 'blue', 'green', 'orange'])
    ax3.set_xlabel('パラメータ値', fontsize=12)
    ax3.set_title('磁気パラメータ（温度非依存）', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    # 値をテキストで表示
    for i, (label, value) in enumerate(zip(param_labels, param_values)):
        if label == 'G0':
            ax3.text(value*1.1, i, f'{value:.2e}', va='center', ha='left')
        else:
            ax3.text(value*1.1, i, f'{value:.5f}', va='center', ha='left')
    
    # 温度による効果の概要
    ax4.text(0.1, 0.8, f'温度範囲: {min(temperatures)} - {max(temperatures)} K', fontsize=14, transform=ax4.transAxes)
    ax4.text(0.1, 0.7, f'eps_bg変化率: {(max(eps_bg_values)-min(eps_bg_values))/min(eps_bg_values)*100:.1f}%', fontsize=14, transform=ax4.transAxes)
    ax4.text(0.1, 0.6, f'膜厚: {d_fixed_um:.2f} μm （固定値）', fontsize=14, transform=ax4.transAxes)
    ax4.text(0.1, 0.5, f'固定磁場: {B_FIXED} T', fontsize=14, transform=ax4.transAxes)
    ax4.text(0.1, 0.3, '磁気パラメータ:', fontsize=14, transform=ax4.transAxes, weight='bold')
    ax4.text(0.1, 0.2, f'g因子 = {magnetic_params["g_factor"]:.4f}', fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.1, f'B4 = {magnetic_params["B4"]:.6f} K', fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.0, f'B6 = {magnetic_params["B6"]:.6f} K', fontsize=12, transform=ax4.transAxes)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('解析結果サマリー', fontsize=14)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / 'temperature_dependencies.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_model_selection_results_temperature(traces: Dict[str, az.InferenceData]):
    """温度依存版LOO-CVの結果を横棒グラフで出力"""
    print("\n--- 温度依存モデル選択指標の評価 ---")
    
    model_names = list(traces.keys())
    loo_values = []
    loo_errors = []
    waic_values = []
    waic_errors = []
    
    # データ収集
    for model_name, trace in traces.items():
        try:
            loo_result = az.loo(trace, pointwise=True)
            loo_values.append(loo_result.elpd_loo)
            loo_errors.append(loo_result.se)
            print(f"{model_name}: elpd_loo = {loo_result.elpd_loo:.2f} ± {loo_result.se:.2f}")
        except Exception as e:
            print(f"{model_name}: LOO-CV計算に失敗 - {e}")
            loo_values.append(np.nan)
            loo_errors.append(np.nan)
        
        try:
            waic_result = az.waic(trace, pointwise=True)
            waic_values.append(waic_result.elpd_waic)
            waic_errors.append(waic_result.se)
            print(f"{model_name}: elpd_waic = {waic_result.elpd_waic:.2f} ± {waic_result.se:.2f}")
        except Exception as e:
            print(f"{model_name}: WAIC計算に失敗 - {e}")
            waic_values.append(np.nan)
            waic_errors.append(np.nan)
    
    # 横棒グラフの作成
    valid_loo_indices = [i for i, l in enumerate(loo_values) if not np.isnan(l)]
    
    if len(valid_loo_indices) >= 2:
        best_loo_idx = valid_loo_indices[np.argmax([loo_values[i] for i in valid_loo_indices])]
        best_loo = loo_values[best_loo_idx]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # LOO-CVプロット
        y_pos = np.arange(len(model_names))
        relative_loo = [loo - best_loo for loo in loo_values]
        
        for i, (model, rel_val, error) in enumerate(zip(model_names, relative_loo, loo_errors)):
            if not np.isnan(rel_val):
                color = 'lightblue' if rel_val == 0 else 'skyblue'
                ax1.barh(i, rel_val, xerr=error, capsize=8, 
                        color=color, edgecolor='navy', alpha=0.8, height=0.6)
        
        ax1.axvline(x=0, color='red', linestyle='-', linewidth=2)
        ax1.set_xlabel('elpd_loo (相対値)', fontsize=12)
        ax1.set_title('LOO-CV比較 (温度依存)', fontsize=14)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(model_names, fontsize=12)
        ax1.grid(True, axis='x', linestyle='--', alpha=0.3)
        
        # WAICプロット
        valid_waic_indices = [i for i, w in enumerate(waic_values) if not np.isnan(w)]
        if len(valid_waic_indices) >= 2:
            best_waic_idx = valid_waic_indices[np.argmax([waic_values[i] for i in valid_waic_indices])]
            best_waic = waic_values[best_waic_idx]
            relative_waic = [waic - best_waic for waic in waic_values]
            
            for i, (model, rel_val, error) in enumerate(zip(model_names, relative_waic, waic_errors)):
                if not np.isnan(rel_val):
                    color = 'lightcoral' if rel_val == 0 else 'salmon'
                    ax2.barh(i, rel_val, xerr=error, capsize=8, 
                            color=color, edgecolor='darkred', alpha=0.8, height=0.6)
            
            ax2.axvline(x=0, color='red', linestyle='-', linewidth=2)
            ax2.set_xlabel('elpd_waic (相対値)', fontsize=12)
            ax2.set_title('WAIC比較 (温度依存)', fontsize=14)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(model_names, fontsize=12)
            ax2.grid(True, axis='x', linestyle='--', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'WAIC計算に失敗', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('WAIC比較 (計算失敗)', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / 'temperature_model_comparison.png', dpi=300, bbox_inches='tight')
        # plt.show()
        
        # 定量的比較結果
        print(f"\n=== 温度依存モデル比較結果 ===")
        for i, model in enumerate(model_names):
            if not np.isnan(loo_values[i]):
                print(f"{model}: elpd_loo = {loo_values[i]:.2f} ± {loo_errors[i]:.2f}")
        
        if len(valid_loo_indices) >= 2:
            diff = loo_values[valid_loo_indices[0]] - loo_values[valid_loo_indices[1]]
            diff_error = np.sqrt(loo_errors[valid_loo_indices[0]]**2 + loo_errors[valid_loo_indices[1]]**2)
            print(f"\nモデル間差異: {abs(diff):.2f} ± {diff_error:.2f}")
            if abs(diff) > 2 * diff_error:
                best_model = model_names[valid_loo_indices[0]] if diff > 0 else model_names[valid_loo_indices[1]]
                print(f"統計的有意差: あり ({best_model} が優位)")
            else:
                print("統計的有意差: なし")
    else:
        print("❌ モデル比較に十分なデータがありません")

def run_iterative_temperature_bayesian_workflow():
    """two_step_iterative_fitting.pyを参考にした反復的温度依存ベイズ推定ワークフロー"""
    print("🚀 温度依存反復ベイズ推定ワークフローを開始します")
    print(f"膜厚固定値: {d_fixed*1e6:.2f} μm")
    print(f"固定磁場: {B_FIXED} T")
    print(f"データファイル: {DATA_FILE_PATH}")
    print(f"シート名: {DATA_SHEET_NAME}")
    print(f"測定温度: {TEMPERATURE_COLUMNS}")
    print(f"周波数分割: 低周波 ≤ {LOW_FREQUENCY_CUTOFF} THz < 中間 < {HIGH_FREQUENCY_CUTOFF} THz ≤ 高周波\n")
    
    # 第1ステップ: データの読み込みと分割
    data_dict = load_and_split_data_three_regions_temperature(
        file_path=DATA_FILE_PATH, 
        sheet_name=DATA_SHEET_NAME,
        low_cutoff=LOW_FREQUENCY_CUTOFF,
        high_cutoff=HIGH_FREQUENCY_CUTOFF
    )
    
    if not data_dict['low_freq'] or not data_dict['high_freq']:
        print("❌ 必要なデータが読み込めませんでした")
        return
    
    # 第2ステップ: 温度別eps_bg初期推定（高周波データから）
    print("\n--- 第1イテレーション: 各温度の高周波eps_bgフィッティング ---")
    
    temperature_specific_params = {}
    for i, dataset in enumerate(data_dict['high_freq']):
        temp = dataset['temperature']
        print(f"\n[{i+1}/{len(data_dict['high_freq'])}] 温度 {temp} K の処理中...")
        
        # 固定パラメータ（初期値）
        fixed_params = {
            'd': d_fixed,
            'g_factor': g_factor_init,
            'B4': B4_init,
            'B6': B6_init,
            'gamma_scale': 1.0
        }
        
        result = fit_eps_bg_only_temperature(dataset, fixed_params)
        if result:
            temperature_specific_params[temp] = {
                'eps_bg': result['eps_bg'],
                'd': d_fixed,  # 固定値を使用
                'temperature': temp
            }
            print(f"  ✅ 温度 {temp} K: eps_bg = {result['eps_bg']:.3f}")
        else:
            print(f"  ❌ 温度 {temp} K: フィッティング失敗")
            # デフォルト値を設定
            temperature_specific_params[temp] = {
                'eps_bg': eps_bg_init,
                'd': d_fixed,
                'temperature': temp
            }
    
    if not temperature_specific_params:
        print("❌ 高周波フィッティングが全て失敗しました")
        return
    
    # 第3ステップ: 反復ベイズ推定
    max_iterations = 2  # 反復回数を2回に変更
    prior_magnetic_params = None
    current_magnetic_params = None
    trace_result = None
    
    for iteration in range(max_iterations):
        print(f"\n--- 第{iteration+2}イテレーション: ベイズ推定による磁気パラメータ推定 ---")
        
        # ベイズ推定実行
        trace_result = run_temperature_bayesian_fit(
            data_dict['low_freq'], 
            temperature_specific_params,
            model_type='B_form',
            prior_magnetic_params=prior_magnetic_params
        )
        
        if trace_result is None:
            print(f"❌ 第{iteration+2}イテレーション: ベイズ推定に失敗")
            if iteration == 0:
                print("最初のベイズ推定に失敗しました。処理を終了します。")
                return
            else:
                print("前回の結果を使用して継続します。")
                break
        
        # 推定結果の抽出
        current_magnetic_params = extract_bayesian_parameters(trace_result)
        print(f"第{iteration+2}イテレーション結果:")
        for param, value in current_magnetic_params.items():
            if param == 'G0':
                print(f"  {param} = {value:.3e}")
            elif isinstance(value, np.ndarray):
                # 配列の場合は特別に処理
                if value.ndim == 0:  # 0次元配列（スカラー）
                    print(f"  {param} = {float(value):.6f}")
                else:  # 多次元配列
                    print(f"  {param} = {value}")
            else:
                print(f"  {param} = {value:.6f}")
        
        # 収束判定（g_factor、B4、B6の変化率で判定）
        if prior_magnetic_params is not None:
            g_change = abs(current_magnetic_params['g_factor'] - prior_magnetic_params['g_factor']) / prior_magnetic_params['g_factor']
            B4_change = abs(current_magnetic_params['B4'] - prior_magnetic_params['B4']) / abs(prior_magnetic_params['B4'])
            B6_change = abs(current_magnetic_params['B6'] - prior_magnetic_params['B6']) / abs(prior_magnetic_params['B6'])
            
            print(f"  パラメータ変化率: g_factor={g_change*100:.2f}%, B4={B4_change*100:.2f}%, B6={B6_change*100:.2f}%")
            
            # 収束条件: 全てのパラメータが5%未満の変化
            if g_change < 0.05 and B4_change < 0.05 and B6_change < 0.05:
                print(f"  ✅ 収束しました（第{iteration+2}イテレーション）")
                break
        
        # 次のイテレーションのための事前分布更新
        prior_magnetic_params = current_magnetic_params.copy()
        
        # eps_bgパラメータの更新（ベイズ推定結果を使用）
        print(f"\n第{iteration+3}イテレーション準備: 更新された磁気パラメータでeps_bg再推定")
        updated_temp_params = {}
        
        for temp, prev_params in temperature_specific_params.items():
            # 対応する高周波データセットを見つける
            temp_dataset = None
            for dataset in data_dict['high_freq']:
                if dataset['temperature'] == temp:
                    temp_dataset = dataset
                    break
            
            if temp_dataset is None:
                print(f"  警告: 温度 {temp} K のデータが見つかりません")
                updated_temp_params[temp] = prev_params
                continue
            
            # 更新された磁気パラメータで固定パラメータを設定
            updated_fixed_params = {
                'd': d_fixed,
                'g_factor': current_magnetic_params['g_factor'],
                'B4': current_magnetic_params['B4'],
                'B6': current_magnetic_params['B6'],
                'gamma_scale': 1.0
            }
            
            result = fit_eps_bg_only_temperature(temp_dataset, updated_fixed_params)
            if result:
                updated_temp_params[temp] = {
                    'eps_bg': result['eps_bg'],
                    'd': d_fixed,
                    'temperature': temp
                }
                change = abs(result['eps_bg'] - prev_params['eps_bg'])
                print(f"  温度 {temp} K: eps_bg {prev_params['eps_bg']:.3f} → {result['eps_bg']:.3f} (変化量: {change:.3f})")
            else:
                print(f"  温度 {temp} K: eps_bg更新失敗、前回値を使用")
                updated_temp_params[temp] = prev_params
        
        temperature_specific_params = updated_temp_params
    
    # 最終結果の表示と保存
    print("\n=== 最終結果サマリー ===")
    print("温度別光学パラメータ:")
    for temp, params in sorted(temperature_specific_params.items()):
        print(f"  {temp} K: eps_bg = {params['eps_bg']:.4f}")
    print(f"膜厚: {d_fixed*1e6:.2f} μm （全温度で固定）")
    print("\n磁気パラメータ (最終ベイズ推定結果):")
    if current_magnetic_params is not None:
        for param, value in current_magnetic_params.items():
            if param == 'G0':
                print(f"  {param} = {value:.3e}")
            elif isinstance(value, np.ndarray):
                # 配列の場合は特別に処理
                if value.ndim == 0:  # 0次元配列（スカラー）
                    print(f"  {param} = {float(value):.6f}")
                else:  # 多次元配列
                    print(f"  {param} = {value}")
            else:
                print(f"  {param} = {value:.6f}")
    
    print("🎉 温度依存反復ベイズ推定ワークフローが完了しました。")
    
    # 最終結果の可視化
    if trace_result is not None:
        # H_formとB_formの両方で解析実行
        print("\n--- H_form と B_form の比較解析を実行 ---")
        final_traces = {}
        
        # B_formの結果を保存
        final_traces['B_form'] = trace_result
        
        # H_formでも解析実行
        try:
            h_form_trace = run_temperature_bayesian_fit(
                data_dict['low_freq'], 
                temperature_specific_params,
                model_type='H_form',
                prior_magnetic_params=current_magnetic_params
            )
            if h_form_trace is not None:
                final_traces['H_form'] = h_form_trace
        except Exception as e:
            print(f"❌ H_form解析に失敗: {e}")
        
        # 統合比較プロット（2×2レイアウト）
        if len(final_traces) >= 2:
            print("\n--- H_form と B_form の統合比較プロット（2×2レイアウト）作成中 ---")
            all_data_full = load_data_full_range_temperature(DATA_FILE_PATH, DATA_SHEET_NAME)
            plot_combined_temperature_model_comparison(all_data_full, temperature_specific_params, final_traces)
        
        # 温度依存性プロット
        plot_temperature_dependencies(temperature_specific_params, trace_result)

if __name__ == "__main__":
    run_iterative_temperature_bayesian_workflow()