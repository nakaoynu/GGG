# two_step_iterative_fitting.py - 反復的背景誘電率フィッティング

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
import csv
from datetime import datetime
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
    IMAGE_DIR = pathlib.Path(__file__).parent / "B_field_iterative"
    IMAGE_DIR.mkdir(exist_ok=True)
    print(f"画像は '{IMAGE_DIR.resolve()}' に保存されます。")

# --- 1. 物理定数とシミュレーション条件 ---
if __name__ == "__main__":
    print("--- 1. 物理定数と初期値を設定します ---")

kB = 1.380649e-23; muB = 9.274010e-24; hbar = 1.054571e-34
c = 299792458; mu0 = 4.0 * np.pi * 1e-7
s = 3.5; N_spin = 24 / 1.238 * 1e27
TEMPERATURE = 1.5 # 温度 (K)

# パラメータの初期値
d_fixed = 157.8e-6  # 膜厚は固定値として使用
eps_bg_init = 14.20
B4_init = 0.001149; B6_init = -0.000010
gamma_init = 0.11e12; a_scale_init = 0.604971; g_factor_init = 2.015445

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
    
    # gamma_arrayがdelta_Eと同じ次元を持つように調整
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
    # 入力値の検証と安全な処理
    eps_bg = max(eps_bg, 0.1)  # 最小値を設定
    d = max(d, 1e-9)  # 最小値を設定
    
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
def load_and_split_data_three_regions(file_path: str, sheet_name: str, 
                                     low_cutoff: float = 0.378, 
                                     high_cutoff: float = 0.45) -> Dict[str, List[Dict[str, Any]]]:
    """データを読み込み、低周波・中間・高周波領域に分割する。
    
    Args:
        file_path: Excelファイルのパス
        sheet_name: シート名
        low_cutoff: 低周波領域の上限 (0.378 THz)
        high_cutoff: 高周波領域の下限 (0.45 THz)
    
    Returns:
        Dict containing:
        - 'low_freq': [~, 0.378THz] - ベイズ推定用
        - 'mid_freq': [0.378THz, 0.45THz] - 中間領域（使用しない）
        - 'high_freq': [0.45THz, ~] - 共振器モードフィッティング用
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    except Exception as e:
        raise FileNotFoundError(f"Excelファイル '{file_path}' が読み込めません: {e}")
    
    freq_col = 'Frequency (THz)'
    df[freq_col] = pd.to_numeric(df[freq_col], errors='coerce')
    trans_cols = [col for col in df.columns if 'Transmittance' in col and 'T)' in col]
    
    low_freq_datasets, mid_freq_datasets, high_freq_datasets = [], [], []
    
    for col in trans_cols:
        b_match = re.search(r'\((\d+\.?\d*)T\)', col)
        if not b_match: continue
        b_value = float(b_match.group(1))
        df_clean = df[[freq_col, col]].dropna()
        freq, trans = df_clean[freq_col].values.astype(np.float64), df_clean[col].values.astype(np.float64)
        
        # 3つの領域にマスクを定義
        low_mask = freq <= low_cutoff
        mid_mask = (freq > low_cutoff) & (freq < high_cutoff)
        high_mask = freq >= high_cutoff
        
        base_data = {'temperature': TEMPERATURE, 'b_field': b_value}
        
        # 低周波領域 [~, 0.378THz] - ベイズ推定用
        if np.any(low_mask):
            min_low, max_low = trans[low_mask].min(), trans[low_mask].max()
            trans_norm_low = (trans[low_mask] - min_low) / (max_low - min_low) if max_low > min_low else np.full_like(trans[low_mask], 0.5)
            low_freq_datasets.append({**base_data, 
                                    'frequency': freq[low_mask], 
                                    'transmittance': trans_norm_low, 
                                    'omega': freq[low_mask] * 1e12 * 2 * np.pi})
        
        # 中間領域 [0.378THz, 0.45THz] - 参考用（メインの解析では使用しない）
        if np.any(mid_mask):
            min_mid, max_mid = trans[mid_mask].min(), trans[mid_mask].max()
            trans_norm_mid = (trans[mid_mask] - min_mid) / (max_mid - min_mid) if max_mid > min_mid else np.full_like(trans[mid_mask], 0.5)
            mid_freq_datasets.append({**base_data, 
                                    'frequency': freq[mid_mask], 
                                    'transmittance': trans_norm_mid, 
                                    'omega': freq[mid_mask] * 1e12 * 2 * np.pi})
        
        # 高周波領域 [0.45THz, ~] - 光学パラメータフィッティング用
        if np.any(high_mask):
            min_high, max_high = trans[high_mask].min(), trans[high_mask].max()
            trans_norm_high = (trans[high_mask] - min_high) / (max_high - min_high) if max_high > min_high else np.full_like(trans[high_mask], 0.5)
            high_freq_datasets.append({**base_data, 
                                     'frequency': freq[high_mask], 
                                     'transmittance': trans_norm_high, 
                                     'omega': freq[high_mask] * 1e12 * 2 * np.pi})
    
    print(f"データ分割結果:")
    print(f"  低周波領域 [~, {low_cutoff}THz]: {len(low_freq_datasets)} データセット")
    print(f"  中間領域 [{low_cutoff}THz, {high_cutoff}THz]: {len(mid_freq_datasets)} データセット")
    print(f"  高周波領域 [{high_cutoff}THz, ~]: {len(high_freq_datasets)} データセット")
    
    return {
        'low_freq': low_freq_datasets, 
        'mid_freq': mid_freq_datasets,
        'high_freq': high_freq_datasets
    }

# 後方互換性のため、既存の関数も保持
def load_and_split_data(file_path: str, sheet_name: str, cutoff_freq: float) -> Dict[str, List[Dict[str, Any]]]:
    """データを読み込み、高周波と低周波領域に分割する（後方互換性のため保持）。"""
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

def fit_eps_bg_only(dataset: Dict[str, Any], 
                    fixed_params: Optional[Dict[str, float]] = None,
                    G0_from_bayesian: Optional[float] = None) -> Dict[str, float]:
    """各磁場で高周波データからeps_bgのみをフィッティングする（他パラメータは固定）"""
    print(f"\n--- 磁場 {dataset['b_field']} T の高周波eps_bgフィッティング ---")
    
    # 固定パラメータの取得
    if fixed_params is None:
        fixed_params = {
            'd': d_fixed,
            'g_factor': g_factor_init,
            'B4': B4_init,
            'B6': B6_init,
            'gamma_scale': 1.0
        }
    
    def magnetic_cavity_model_eps_bg_only(freq_thz, eps_bg_fit):
        """eps_bgのみを変数とする磁気感受率を考慮した高周波透過率モデル"""
        try:
            # 入力パラメータの妥当性チェック
            if not np.isfinite(eps_bg_fit):
                return np.full_like(freq_thz, 0.5)
            
            # eps_bgを安全な範囲にクリップ
            eps_bg_fit = np.clip(eps_bg_fit, 5.0, 25.0)
            
            # 固定パラメータを使用
            d_fit = fixed_params['d']
            g_factor_fit = fixed_params['g_factor']
            B4_fit = fixed_params['B4']
            B6_fit = fixed_params['B6']
            gamma_scale = fixed_params['gamma_scale']
            
            omega = freq_thz * 1e12 * 2 * np.pi
            
            # ハミルトニアンと磁気感受率の計算
            H = get_hamiltonian(dataset['b_field'], g_factor_fit, B4_fit, B6_fit)
            
            # 高周波用の簡略化されたガンマ（単一値）
            gamma_array = np.full(7, gamma_scale * gamma_init)
            chi_raw = calculate_susceptibility(omega, H, TEMPERATURE, gamma_array)
            
            # 磁気感受率のスケーリング
            if G0_from_bayesian is not None:
                G0 = G0_from_bayesian
            else:
                G0 = mu0 * N_spin * (g_factor_fit * muB)**2 / (2 * hbar) * 0.1  # 初期スケーリング係数
            
            chi = G0 * chi_raw
            
            # H_formで透磁率を計算
            mu_r = 1 + chi
            
            result = calculate_normalized_transmission(omega, mu_r, d_fit, eps_bg_fit)
            
            # NaNや無限大値をチェック
            if np.any(~np.isfinite(result)):
                return np.full_like(freq_thz, 0.5)  # フォールバック値
            
            return result
        except Exception as e:
            print(f"  ❌ モデル計算エラー: {str(e)}")
            return np.full_like(freq_thz, 0.5)  # エラー時のフォールバック

    # 複数の初期値を試行
    success = False
    result = {}
    
    # eps_bgの初期値候補（磁場依存性を考慮）
    b_field = dataset['b_field']
    if b_field >= 9.0:
        # 高磁場では若干低めの初期値から開始
        initial_eps_bg_values = [eps_bg_init * 0.95, eps_bg_init * 0.90, eps_bg_init, 
                                eps_bg_init * 1.05, 12.5, 13.5, 14.5]
        bounds_eps_bg = (10.0, 18.0)  # 範囲を若干拡張
    elif b_field >= 7.0:
        # 中間磁場
        initial_eps_bg_values = [eps_bg_init, eps_bg_init * 0.95, eps_bg_init * 1.05, 
                                12.0, 13.0, 14.0, 15.0]
        bounds_eps_bg = (10.0, 17.0)
    else:
        # 低磁場
        initial_eps_bg_values = [eps_bg_init, eps_bg_init * 0.9, eps_bg_init * 1.1, 
                                12.0, 14.0, 15.0]
        bounds_eps_bg = (10.0, 16.0)
    
    for attempt, initial_eps_bg in enumerate(initial_eps_bg_values):
        try:
            print(f"  試行 {attempt + 1}: eps_bg初期値 = {initial_eps_bg:.2f}")
            
            popt, pcov = curve_fit(
                magnetic_cavity_model_eps_bg_only,
                dataset['frequency'], 
                dataset['transmittance'], 
                p0=[initial_eps_bg],
                bounds=([bounds_eps_bg[0]], [bounds_eps_bg[1]]),
                maxfev=5000,
                ftol=1e-6,
                xtol=1e-6,
                method='trf'
            )
            
            eps_bg_fit = popt[0]
            
            # パラメータの妥当性チェック
            if bounds_eps_bg[0] <= eps_bg_fit <= bounds_eps_bg[1]:
                result = {
                    'd': fixed_params['d'],  # 固定値をそのまま返す
                    'eps_bg': eps_bg_fit,    # フィッティング結果
                    'g_factor': fixed_params['g_factor'],  # 固定値
                    'B4': fixed_params['B4'],              # 固定値
                    'B6': fixed_params['B6'],              # 固定値
                    'gamma_scale': fixed_params['gamma_scale'],  # 固定値
                    'b_field': dataset['b_field']
                }
                
                print(f"  成功 (試行 {attempt + 1}): eps_bg = {eps_bg_fit:.3f}")
                print(f"  固定パラメータ: d = {fixed_params['d']*1e6:.1f}μm, g = {fixed_params['g_factor']:.3f}")
                
                success = True
                break
            else:
                print(f"  試行 {attempt + 1}: eps_bgが物理的範囲外")
                
        except RuntimeError as e:
            print(f"  試行 {attempt + 1}: フィッティング失敗 - {e}")
            continue
        except Exception as e:
            print(f"  試行 {attempt + 1}: 予期しないエラー - {e}")
            continue
    
    if not success:
        print("  ❌ 全ての試行に失敗")
        result = {}
    
    return result

class MagneticModelOp(Op):
    """Step 2: 低周波領域の磁気パラメータを推定するためのPyMC Op。"""
    def __init__(self, datasets: List[Dict[str, Any]], field_specific_params: Dict[float, Dict[str, float]], model_type: str):
        self.datasets = datasets
        self.field_specific_params = field_specific_params
        self.model_type = model_type
        self.itypes = [pt.dscalar, pt.dvector, pt.dscalar, pt.dscalar, pt.dscalar]
        self.otypes = [pt.dvector]
    
    def perform(self, node, inputs, output_storage):
        a_scale, gamma, g_factor, B4, B6 = inputs
        full_predicted_y = []
        
        for data in self.datasets:
            # 該当する磁場の固定パラメータを取得
            b_field = data['b_field']
            if b_field in self.field_specific_params:
                d_used = self.field_specific_params[b_field]['d']
                eps_bg_fixed = self.field_specific_params[b_field]['eps_bg']
            else:
                # フォールバック：グローバル変数を使用
                d_used = 157.8e-6  # d_fixed の値を直接指定
                eps_bg_fixed = 13.14  # eps_bg_init の値を直接指定
            
            H = get_hamiltonian(data['b_field'], g_factor, B4, B6)
            chi_raw = calculate_susceptibility(data['omega'], H, data['temperature'], gamma)
            G0 = a_scale * mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
            chi = G0 * chi_raw
            
            # 数値的安定性のチェック
            if np.any(~np.isfinite(chi)):
                chi = np.where(np.isfinite(chi), chi, 0.0)
            
            # モデルタイプに応じて透磁率を計算
            if self.model_type == 'H_form':
                mu_r = 1 + chi
            else:  # B_form
                mu_r = 1 / (1 - chi)
            
            # mu_rの数値的安定性をチェック
            if np.any(~np.isfinite(mu_r)):
                mu_r = np.where(np.isfinite(mu_r), mu_r, 1.0)
            
            predicted_y = calculate_normalized_transmission(data['omega'], mu_r, d_used, eps_bg_fixed)
            
            # 予測値の最終チェック
            if np.any(~np.isfinite(predicted_y)):
                predicted_y = np.where(np.isfinite(predicted_y), predicted_y, 0.5)
            
            full_predicted_y.append(predicted_y)
        
        output_storage[0][0] = np.concatenate(full_predicted_y)

def adaptive_sampling_config(n_datasets: int, prior_magnetic_params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """データセット数に応じて適応的にサンプリング設定を調整"""
    if n_datasets <= 5:
        # 小規模：高精度設定
        if prior_magnetic_params is not None:
            return {"draws": 1500, "tune": 2000, "target_accept": 0.80}
        else:
            return {"draws": 2000, "tune": 2500, "target_accept": 0.85}
    elif n_datasets <= 10:
        # 中規模：バランス設定
        if prior_magnetic_params is not None:
            return {"draws": 1200, "tune": 1600, "target_accept": 0.78}
        else:
            return {"draws": 1600, "tune": 2000, "target_accept": 0.82}
    else:
        # 大規模（20+）：効率重視設定
        if prior_magnetic_params is not None:
            return {"draws": 800, "tune": 1200, "target_accept": 0.75}
        else:
            return {"draws": 1200, "tune": 1800, "target_accept": 0.80}

def run_bayesian_magnetic_fit_with_fixed_eps(datasets: List[Dict[str, Any]], field_specific_params: Dict[float, Dict[str, float]], 
                                           prior_magnetic_params: Optional[Dict[str, float]] = None, model_type: str = 'H_form') -> az.InferenceData:
    """Step 2: 磁場毎の固定eps_bgを使用してベイズ推定を実行する"""
    print(f"\n--- Step 2: 磁場別固定eps_bgによるベイズ推定 (モデル: {model_type}) ---")
    trans_observed = np.concatenate([d['transmittance'] for d in datasets])
    
    with pm.Model() as model:
        # 事前分布の設定
        if prior_magnetic_params:
            # 前回のベイズ推定結果を事前分布として使用
            a_scale = pm.TruncatedNormal('a_scale', mu=prior_magnetic_params['a_scale'], sigma=0.3, lower=0.1, upper=5.0)
            g_factor = pm.TruncatedNormal('g_factor', mu=prior_magnetic_params['g_factor'], sigma=0.05, lower=1.95, upper=2.05)
            B4 = pm.Normal('B4', mu=prior_magnetic_params['B4'], sigma=abs(prior_magnetic_params['B4'])*0.5 + 0.001)
            B6 = pm.Normal('B6', mu=prior_magnetic_params['B6'], sigma=abs(prior_magnetic_params['B6'])*0.5 + 0.0001)
            print(f"前回の推定結果を事前分布として使用:")
            print(f"  a_scale事前分布中心: {prior_magnetic_params['a_scale']:.3f}")
            print(f"  g_factor事前分布中心: {prior_magnetic_params['g_factor']:.3f}")
            print(f"  B4事前分布中心: {prior_magnetic_params['B4']:.5f}")
            print(f"  B6事前分布中心: {prior_magnetic_params['B6']:.6f}")
        else:
            # 初回のデフォルト事前分布
            a_scale = pm.TruncatedNormal('a_scale', mu=a_scale_init, sigma=0.5, lower=0.1, upper=5.0)
            g_factor = pm.TruncatedNormal('g_factor', mu=g_factor_init, sigma=0.03, lower=1.95, upper=2.05)
            B4 = pm.Normal('B4', mu=B4_init, sigma=abs(B4_init)*0.5)
            B6 = pm.Normal('B6', mu=B6_init, sigma=abs(B6_init)*0.5)
            print("初回のデフォルト事前分布を使用")
        
        log_gamma_mu = pm.Normal('log_gamma_mu', mu=np.log(gamma_init), sigma=1.0)
        log_gamma_sigma = pm.HalfNormal('log_gamma_sigma', sigma=0.5)
        log_gamma_offset = pm.Normal('log_gamma_offset', mu=0, sigma=0.5, shape=7)
        gamma = pm.Deterministic('gamma', pt.exp(log_gamma_mu + log_gamma_offset * log_gamma_sigma))
        
        op = MagneticModelOp(datasets, field_specific_params, model_type)
        mu = op(a_scale, gamma, g_factor, B4, B6)
        sigma = pm.HalfCauchy('sigma', beta=0.05)
        
        # 観測モデル
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=trans_observed)
        
        # サンプリング設定（データセット数に応じて適応的調整）
        cpu_count = os.cpu_count() or 4
        n_datasets = len(datasets)
        sampling_config = adaptive_sampling_config(n_datasets, prior_magnetic_params)
        
        try:
            print(f"ベイズサンプリングを開始します...（{n_datasets}データセット用最適化設定）")
            trace = pm.sample(int(sampling_config["draws"]),
                              tune=int(sampling_config["tune"]),
                              chains=2,   
                              cores=min(cpu_count, 2), 
                              target_accept=sampling_config["target_accept"],
                              max_treedepth=8 if n_datasets > 10 else 10,  # 大規模時は深度削減
                              init='jitter+adapt_diag',
                              idata_kwargs={"log_likelihood": True}, 
                              random_seed=42,
                              progressbar=True)
            
            # 収束診断
            print("\n--- 収束診断 ---")
            summary = az.summary(trace, var_names=['a_scale', 'g_factor', 'B4', 'B6'])
            max_rhat = summary['r_hat'].max()
            min_ess_bulk = summary['ess_bulk'].min()
            min_ess_tail = summary['ess_tail'].min()
            
            print(f"最大 r_hat: {max_rhat:.4f} (< 1.01 が望ましい)")
            print(f"最小 ess_bulk: {min_ess_bulk:.0f} (> 300 が望ましい)")
            print(f"最小 ess_tail: {min_ess_tail:.0f} (> 300 が望ましい)")
            
            # 元の基準に戻す
            if max_rhat > 1.01:
                print("⚠️ 警告: r_hat > 1.01 - 収束に問題があります")
            if min_ess_bulk < 300:
                print("⚠️ 警告: ess_bulk < 300 - 有効サンプルサイズが不足")
            if min_ess_tail < 300:
                print("⚠️ 警告: ess_tail < 300 - 分布の裾の推定が不安定")
                
        except Exception as e:
            print(f"高精度サンプリングに失敗: {e}")
            print("中精度設定でリトライします...")
            try:
                trace = pm.sample(2000, 
                                  tune=3000, 
                                  chains=2, 
                                  cores=min(cpu_count, 2), 
                                  target_accept=0.8,
                                  idata_kwargs={"log_likelihood": True}, 
                                  random_seed=42,
                                  progressbar=False)  # プログレスバーを無効化
            except Exception as e2:
                print(f"中精度設定も失敗: {e2}")
                print("最小設定でリトライします...")
                trace = pm.sample(1000, 
                                  tune=1500, 
                                  chains=2, 
                                  cores=1,  # シングルコアで実行
                                  idata_kwargs={"log_likelihood": True}, 
                                  random_seed=42,
                                  progressbar=False)

        # サンプリング後にlog_likelihoodが存在するかチェックして、なければ計算
        with model:
            if "log_likelihood" not in trace:
                trace = pm.compute_log_likelihood(trace)
                assert isinstance(trace, az.InferenceData)
    
    print("----------------------------------------------------")
    print("▶ Step 2 結果 (サマリー):")
    summary = az.summary(trace, var_names=['a_scale', 'g_factor', 'B4', 'B6', 'gamma'])
    print(summary)
    print("----------------------------------------------------")
    return trace

def extract_bayesian_parameters(trace: az.InferenceData) -> Dict[str, float]:
    """ベイズ推定結果から平均パラメータを抽出"""
    posterior = trace["posterior"]
    return {
        'a_scale': float(posterior['a_scale'].mean()),
        'g_factor': float(posterior['g_factor'].mean()),
        'B4': float(posterior['B4'].mean()),
        'B6': float(posterior['B6'].mean()),
        'G0': float(posterior['a_scale'].mean() * mu0 * N_spin * (posterior['g_factor'].mean() * muB)**2 / (2 * hbar))
    }

def calculate_peak_errors(all_datasets: List[Dict[str, Any]], 
                         field_specific_params: Dict[float, Dict[str, float]], 
                         bayesian_trace: az.InferenceData) -> Dict[float, float]:
    """各磁場のピーク位置相対誤差を計算"""
    peak_errors = {}
    mean_params = extract_bayesian_parameters(bayesian_trace)
    
    for data in all_datasets:
        b_field = data['b_field']
        
        # 該当磁場のパラメータを取得
        if b_field in field_specific_params:
            d_fixed = field_specific_params[b_field]['d']
            eps_bg_fixed = field_specific_params[b_field]['eps_bg']
        else:
            continue
        
        # 全周波数範囲での予測
        freq_plot = np.linspace(data['frequency'].min(), data['frequency'].max(), 500)
        omega_plot = freq_plot * 1e12 * 2 * np.pi
        
        # ベイズ推定結果による予測
        H = get_hamiltonian(b_field, mean_params['g_factor'], mean_params['B4'], mean_params['B6'])
        gamma_mean = np.full(7, gamma_init)
        chi_raw = calculate_susceptibility(omega_plot, H, TEMPERATURE, gamma_mean)
        chi = mean_params['G0'] * chi_raw
        mu_r = 1 + chi  # H_form
        
        predicted_trans = calculate_normalized_transmission(omega_plot, mu_r, d_fixed, eps_bg_fixed)
        
        # 実験データの正規化
        min_exp, max_exp = data['transmittance_full'].min(), data['transmittance_full'].max()
        trans_norm_full = (data['transmittance_full'] - min_exp) / (max_exp - min_exp)
        
        # ピーク検出（透過率の山 = 正のピーク）
        try:
            # 実験データのピーク検出（パラメータを調整）
            exp_peaks, _ = find_peaks(trans_norm_full, 
                                    height=0.25,       # 山の閾値を下げる
                                    distance=3,       # 距離をさらに短縮
                                    prominence=0.03,  # プロミネンスをさらに緩和
                                    width=1)          # 最小幅を指定
            exp_peak_freqs = data['frequency'][exp_peaks]
            
            # 予測データのピーク検出
            pred_peaks, _ = find_peaks(predicted_trans, 
                                     height=0.25,      # 山の閾値を下げる
                                     distance=3,      # 距離をさらに短縮
                                     prominence=0.03, # プロミネンスをさらに緩和
                                     width=1)
            pred_peak_freqs = freq_plot[pred_peaks]
            
            if len(exp_peak_freqs) > 0 and len(pred_peak_freqs) > 0:
                # ピーク位置のマッチング（最近傍）
                from scipy.spatial.distance import cdist
                distances = cdist(exp_peak_freqs.reshape(-1, 1), pred_peak_freqs.reshape(-1, 1))
                
                # 各実験ピークに対して最も近い予測ピークを探す
                relative_errors = []
                for i, exp_freq in enumerate(exp_peak_freqs):
                    min_idx = np.argmin(distances[i])
                    pred_freq = pred_peak_freqs[min_idx]
                    
                    # 相対誤差を計算（距離制限を緩和）
                    if exp_freq > 0 and abs(pred_freq - exp_freq) < 0.3:  # 0.3 THz以内に緩和
                        rel_error = abs((pred_freq - exp_freq) / exp_freq)
                        relative_errors.append(rel_error)
                
                if relative_errors:
                    # 最大相対誤差を記録
                    peak_errors[b_field] = max(relative_errors)
                    print(f"    磁場 {b_field} T: 最大ピーク相対誤差 = {max(relative_errors)*100:.2f}% ({len(relative_errors)}ピーク)")
                else:
                    peak_errors[b_field] = float('inf')
                    print(f"    磁場 {b_field} T: マッチするピークなし")
            else:
                peak_errors[b_field] = float('inf')
                print(f"    磁場 {b_field} T: ピーク検出不足 (実験:{len(exp_peak_freqs)}, 予測:{len(pred_peak_freqs)})")
                
        except Exception as e:
            print(f"    磁場 {b_field} T: ピーク誤差計算エラー - {e}")
            peak_errors[b_field] = float('inf')
    
    return peak_errors

def iterative_fitting_process(datasets_split: Dict[str, List[Dict[str, Any]]], 
                            all_datasets: List[Dict[str, Any]],
                            max_iterations: int = 10, 
                            peak_error_threshold: float = 0.015) -> Tuple[Dict[float, Dict[str, float]], Optional[az.InferenceData]]:
    """反復的フィッティングプロセス（データセット数に応じて自動最適化）"""
    n_datasets = len(all_datasets)
    
    # データセット数に応じて最適化
    if n_datasets > 15:
        max_iterations = min(max_iterations, 2)  # 大規模時は2回に制限
        peak_error_threshold = max(peak_error_threshold, 0.025)  # 収束条件緩和
        print(f"大規模データセット({n_datasets}個)用最適化: 反復{max_iterations}回, 閾値{peak_error_threshold*100:.1f}%")
    elif n_datasets > 8:
        max_iterations = min(max_iterations, 3)  # 中規模時は3回
        peak_error_threshold = max(peak_error_threshold, 0.020)
        print(f"中規模データセット({n_datasets}個)用最適化: 反復{max_iterations}回, 閾値{peak_error_threshold*100:.1f}%")
    
    print("\n=== 反復的フィッティングプロセスを開始します ===")
    print(f"収束条件: 全磁場のピーク位置相対誤差 < {peak_error_threshold*100:.1f}%")
    
    # 初期化
    field_specific_params = {}
    prior_magnetic_params = None
    bayesian_trace = None
    
    # 各磁場値を取得
    b_fields = sorted(list(set([d['b_field'] for d in datasets_split['high_freq']])))
    print(f"対象磁場: {b_fields} T")
    
    iteration = 0
    for iteration in range(max_iterations):
        print(f"\n--- 反復 {iteration + 1}/{max_iterations} ---")
        
        # Step 1: 各磁場で独立に高周波フィッティング（eps_bgのみ）
        print("\nStep 1: 各磁場での独立高周波eps_bgフィッティング")
        new_field_params = {}
        
        for dataset in datasets_split['high_freq']:
            b_field = dataset['b_field']
            
            # 前回のベイズ推定からG0を取得（2回目以降）
            G0_from_bayesian = None
            if prior_magnetic_params and 'G0' in prior_magnetic_params:
                G0_from_bayesian = prior_magnetic_params['G0']
            
            # 固定パラメータを準備（前回のベイズ推定結果またはデフォルト値）
            fixed_params = {}
            if prior_magnetic_params:
                fixed_params = {
                    'd': d_fixed,  # 膜厚は常に固定値
                    'g_factor': prior_magnetic_params.get('g_factor', g_factor_init),
                    'B4': prior_magnetic_params.get('B4', B4_init),
                    'B6': prior_magnetic_params.get('B6', B6_init),
                    'gamma_scale': prior_magnetic_params.get('gamma_scale', 1.0)
                }
            else:
                fixed_params = {
                    'd': d_fixed,
                    'g_factor': g_factor_init,
                    'B4': B4_init,
                    'B6': B6_init,
                    'gamma_scale': 1.0
                }
            
            result = fit_eps_bg_only(dataset, fixed_params, G0_from_bayesian)
            if result:
                new_field_params[b_field] = result
        
        if not new_field_params:
            print("❌ 高周波フィッティングに失敗しました。")
            break
        
        # Step 2: 固定eps_bgでベイズ推定
        print("\nStep 2: 固定eps_bgによるベイズ推定")
        bayesian_trace = run_bayesian_magnetic_fit_with_fixed_eps(
            datasets_split['low_freq'], 
            new_field_params, 
            prior_magnetic_params
        )
        
        # 新しい磁気パラメータを抽出
        new_magnetic_params = extract_bayesian_parameters(bayesian_trace)
        
        # ピーク位置相対誤差による収束性チェック
        print("\n--- ピーク位置相対誤差による収束診断 ---")
        peak_errors = calculate_peak_errors(all_datasets, new_field_params, bayesian_trace)
        
        # 全磁場の最大ピーク誤差を取得
        valid_errors = [err for err in peak_errors.values() if not np.isinf(err)]
        if valid_errors:
            max_peak_error = max(valid_errors)
            print(f"全磁場の最大ピーク相対誤差: {max_peak_error*100:.2f}%")
            
            # 元の収束判定に戻す
            if max_peak_error < peak_error_threshold:
                print(f"✅ 収束しました (全磁場のピーク誤差 < {peak_error_threshold*100:.1f}%)")
                field_specific_params = new_field_params
                break
            else:
                print(f"⏳ 継続中 (目標: < {peak_error_threshold*100:.1f}%)")
        else:
            print("⚠️ ピーク誤差計算に失敗 - eps_bg変化で代替判定")
            # フォールバック: eps_bgの変化を計算
            if iteration > 0 and prior_magnetic_params:
                eps_bg_changes = []
                for b_field in b_fields:
                    if b_field in field_specific_params and b_field in new_field_params:
                        old_eps = field_specific_params[b_field]['eps_bg']
                        new_eps = new_field_params[b_field]['eps_bg']
                        change = abs((new_eps - old_eps) / old_eps)
                        eps_bg_changes.append(change)
                
                max_eps_change = max(eps_bg_changes) if eps_bg_changes else float('inf')
                print(f"最大eps_bg変化率: {max_eps_change:.4f}")
                
                if max_eps_change < 0.005:  # より厳しい条件
                    print("✅ eps_bg変化により収束")
                    field_specific_params = new_field_params
                    break
        
        # パラメータ更新
        field_specific_params = new_field_params
        prior_magnetic_params = new_magnetic_params
        
        # 進捗表示
        print(f"\n反復 {iteration + 1} の結果:")
        for b_field, params in field_specific_params.items():
            error_str = f" (ピーク誤差: {peak_errors.get(b_field, float('inf'))*100:.2f}%)" if b_field in peak_errors and not np.isinf(peak_errors[b_field]) else ""
            print(f"  磁場 {b_field} T: eps_bg = {params['eps_bg']:.3f}{error_str}")
        print(f"磁気パラメータ: G0 = {new_magnetic_params['G0']:.3e}")
    
    print(f"\n=== 反復フィッティング完了 ({iteration + 1} 回) ===")
    return field_specific_params, bayesian_trace

def plot_iterative_results(all_datasets: List[Dict[str, Any]], 
                         field_specific_params: Dict[float, Dict[str, float]], 
                         bayesian_trace: az.InferenceData, 
                         cutoff_freq: float = 0.45):
    """反復フィッティング結果のプロット"""
    print("\n--- 反復フィッティング結果の可視化 ---")
    
    num_conditions = len(all_datasets)
    fig, axes = plt.subplots(1, num_conditions, figsize=(14 * num_conditions, 10), sharey=True)
    if num_conditions == 1: axes = [axes]
    
    posterior = bayesian_trace["posterior"]
    
    # ピーク誤差を計算
    peak_errors = calculate_peak_errors(all_datasets, field_specific_params, bayesian_trace)
    
    for i, data in enumerate(all_datasets):
        ax = axes[i]
        b_field = data['b_field']
        
        # 全周波数範囲での予測
        freq_plot = np.linspace(data['frequency'].min(), data['frequency'].max(), 500)
        omega_plot = freq_plot * 1e12 * 2 * np.pi
        
        # 該当磁場のパラメータを取得
        if b_field in field_specific_params:
            d_used = field_specific_params[b_field]['d']
            eps_bg_fixed = field_specific_params[b_field]['eps_bg']
        else:
            d_used = d_fixed
            eps_bg_fixed = eps_bg_init
        
        # ベイズ推定結果による予測
        mean_params = extract_bayesian_parameters(bayesian_trace)
        H = get_hamiltonian(b_field, mean_params['g_factor'], mean_params['B4'], mean_params['B6'])
        gamma_mean = np.full(7, gamma_init)
        chi_raw = calculate_susceptibility(omega_plot, H, TEMPERATURE, gamma_mean)
        chi = mean_params['G0'] * chi_raw
        mu_r = 1 + chi  # H_form
        
        predicted_trans = calculate_normalized_transmission(omega_plot, mu_r, d_fixed, eps_bg_fixed)
        
        # 実験データの正規化
        min_exp, max_exp = data['transmittance_full'].min(), data['transmittance_full'].max()
        trans_norm_full = (data['transmittance_full'] - min_exp) / (max_exp - min_exp)
        
        # プロット
        ax.scatter(data['frequency'], trans_norm_full, color='black', s=30, alpha=0.7, label='実験データ')
        ax.plot(freq_plot, predicted_trans, color='red', lw=3, label='反復フィッティング予測')
        
        # ピーク位置をマーク
        try:
            exp_peaks, _ = find_peaks(trans_norm_full, height=0.4, distance=8, prominence=0.1)
            pred_peaks, _ = find_peaks(predicted_trans, height=0.4, distance=8, prominence=0.1)

            if len(exp_peaks) > 0:
                ax.scatter(data['frequency'][exp_peaks], trans_norm_full[exp_peaks], 
                          color='blue', s=60, marker='v', label='実験ピーク', zorder=10)
            if len(pred_peaks) > 0:
                ax.scatter(freq_plot[pred_peaks], predicted_trans[pred_peaks], 
                          color='orange', s=60, marker='^', label='予測ピーク', zorder=10)
        except:
            pass
        
        # 領域境界線を表示（低周波・高周波の境界）
        ax.axvline(x=0.378, color='blue', linestyle='--', linewidth=2, alpha=0.7, 
                  label='低周波境界 (0.378 THz)' if i == 0 else None)
        ax.axvline(x=0.45, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                  label='高周波境界 (0.45 THz)' if i == 0 else None)
        
        # パラメータ情報とピーク誤差をタイトルに追加
        error_str = f"\nピーク誤差: {peak_errors.get(b_field, float('inf'))*100:.2f}%" if b_field in peak_errors and not np.isinf(peak_errors[b_field]) else "\nピーク誤差: 計算失敗"
        ax.set_title(f"磁場 {b_field} T\neps_bg = {eps_bg_fixed:.3f}, d = {d_fixed*1e6:.1f}μm{error_str}", fontsize=14)
        ax.set_xlabel('周波数 (THz)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    
    axes[0].set_ylabel('正規化透過率', fontsize=12)
    fig.suptitle('反復的フィッティング結果（磁場別独立eps_bg）', fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(IMAGE_DIR / 'iterative_fitting_results.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # eps_bgの磁場依存性プロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # eps_bg プロット
    b_fields = sorted(field_specific_params.keys())
    eps_bg_values = [field_specific_params[b]['eps_bg'] for b in b_fields]
    
    ax1.plot(b_fields, eps_bg_values, 'ro-', linewidth=2, markersize=8, label='反復フィッティング結果')
    ax1.set_xlabel('磁場 (T)', fontsize=12)
    ax1.set_ylabel('背景誘電率 eps_bg', fontsize=12)
    ax1.set_title('背景誘電率の磁場依存性（各磁場独立決定）', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # 値をテキストで表示
    for b, eps in zip(b_fields, eps_bg_values):
        ax1.annotate(f'{eps:.3f}', (b, eps), textcoords="offset points", xytext=(0,10), ha='center')
    
    # ピーク誤差プロット
    peak_error_values = [peak_errors.get(b, float('inf'))*100 for b in b_fields]
    valid_mask = [not np.isinf(err) for err in peak_error_values]
    
    if any(valid_mask):
        b_fields_valid = [b for b, valid in zip(b_fields, valid_mask) if valid]
        errors_valid = [err for err, valid in zip(peak_error_values, valid_mask) if valid]
        
        ax2.plot(b_fields_valid, errors_valid, 'go-', linewidth=2, markersize=8, label='ピーク位置相対誤差')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='1%閾値')
        ax2.set_xlabel('磁場 (T)', fontsize=12)
        ax2.set_ylabel('ピーク位置相対誤差 (%)', fontsize=12)
        ax2.set_title('各磁場でのピーク位置相対誤差', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        
        # 値をテキストで表示
        for b, err in zip(b_fields_valid, errors_valid):
            ax2.annotate(f'{err:.2f}%', (b, err), textcoords="offset points", xytext=(0,10), ha='center')
    else:
        ax2.text(0.5, 0.5, 'ピーク誤差計算に失敗', ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('ピーク位置相対誤差（計算失敗）', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / 'eps_bg_and_peak_errors.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_credible_intervals(all_datasets: List[Dict[str, Any]], 
                           field_specific_params: Dict[float, Dict[str, float]], 
                           bayesian_trace: az.InferenceData,
                           model_type: str = 'H_form',
                           n_samples: int = 100):
    """95%信用区間を含む詳細な検証プロット"""
    print(f"\n--- 95%信用区間付き検証プロット ({model_type}) ---")
    
    posterior = bayesian_trace["posterior"]
    num_conditions = len(all_datasets)
    fig, axes = plt.subplots(1, num_conditions, figsize=(12 * num_conditions, 9), sharey=True)
    if num_conditions == 1: axes = [axes]

    for i, data in enumerate(all_datasets):
        ax = axes[i]
        b_field = data['b_field']
        
        # 該当磁場の固定パラメータを取得
        if b_field in field_specific_params:
            d_fixed = field_specific_params[b_field]['d']
            eps_bg_fixed = field_specific_params[b_field]['eps_bg']
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
            gamma_sample = posterior['gamma'].isel(chain=chain_idx, draw=draw_idx).values.flatten()
            
            H_sample = get_hamiltonian(b_field, g_factor_sample, B4_sample, B6_sample)
            chi_raw_sample = calculate_susceptibility(omega_plot, H_sample, TEMPERATURE, gamma_sample)
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
        
        # 実験データの正規化
        min_exp, max_exp = data['transmittance_full'].min(), data['transmittance_full'].max()
        trans_norm_full = (data['transmittance_full'] - min_exp) / (max_exp - min_exp)
        
        # プロット
        color = 'red' if model_type == 'H_form' else 'blue'
        ax.plot(freq_plot, mean_pred, color=color, lw=2.5, label=f'平均予測 ({model_type})')
        ax.fill_between(freq_plot, ci_lower, ci_upper, color=color, alpha=0.3, label='95%信用区間')
        ax.scatter(data['frequency'], trans_norm_full, color='black', s=25, alpha=0.6, label='実験データ')
        
        # 低周波/高周波領域の境界線
        ax.axvline(x=0.378, color='blue', linestyle='--', linewidth=2, alpha=0.7, 
                  label='低周波境界 (0.378 THz)' if i == 0 else None)
        ax.axvline(x=0.45, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                  label='高周波境界 (0.45 THz)' if i == 0 else None)
        
        ax.set_title(f"磁場 {data['b_field']} T ({model_type})", fontsize=14)
        ax.set_xlabel('周波数 (THz)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    axes[0].set_ylabel('正規化透過率', fontsize=12)
    fig.suptitle(f'95%信用区間付き検証結果 ({model_type})', fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(IMAGE_DIR / f'credible_intervals_{model_type}.png')
    # plt.show()

def plot_combined_model_comparison(all_datasets: List[Dict[str, Any]], 
                                 field_specific_params: Dict[float, Dict[str, float]], 
                                 traces: Dict[str, az.InferenceData], 
                                 n_samples: int = 100):
    """H_formとB_formを1枚のグラフに統合してプロット"""
    print("\n--- H_form と B_form の統合比較プロット ---")
    
    num_conditions = len(all_datasets)
    fig, axes = plt.subplots(1, num_conditions, figsize=(16 * num_conditions, 12), sharey=True)
    if num_conditions == 1: axes = [axes]
    
    model_results = {}
    
    for model_type, trace in traces.items():
        model_results[model_type] = {}
        posterior = trace["posterior"]
        
        for i, data in enumerate(all_datasets):
            b_field = data['b_field']
            
            # 該当磁場の固定パラメータを取得
            if b_field in field_specific_params:
                d_fixed = field_specific_params[b_field]['d']
                eps_bg_fixed = field_specific_params[b_field]['eps_bg']
            else:
                continue
            
            freq_plot = np.linspace(data['frequency'].min(), data['frequency'].max(), 500)
            omega_plot = freq_plot * 1e12 * 2 * np.pi
            
            # 信用区間計算
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
                gamma_sample = posterior['gamma'].isel(chain=chain_idx, draw=draw_idx).values.flatten()
                
                H_sample = get_hamiltonian(b_field, g_factor_sample, B4_sample, B6_sample)
                chi_raw_sample = calculate_susceptibility(omega_plot, H_sample, TEMPERATURE, gamma_sample)
                G0_sample = a_scale_sample * mu0 * N_spin * (g_factor_sample * muB)**2 / (2 * hbar)
                chi_sample = G0_sample * chi_raw_sample
                
                if model_type == 'H_form':
                    mu_r_sample = 1 + chi_sample
                else:  # B_form
                    mu_r_sample = 1 / (1 - chi_sample)
                
                pred_sample = calculate_normalized_transmission(omega_plot, mu_r_sample, d_fixed, eps_bg_fixed)
                predictions.append(pred_sample)
            
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            ci_lower, ci_upper = np.percentile(predictions, [2.5, 97.5], axis=0)
            
            model_results[model_type][i] = {
                'freq_plot': freq_plot,
                'mean_pred': mean_pred,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
    
    # プロット作成
    for i, data in enumerate(all_datasets):
        ax = axes[i]
        
        # 実験データプロット
        min_exp, max_exp = data['transmittance_full'].min(), data['transmittance_full'].max()
        trans_norm_full = (data['transmittance_full'] - min_exp) / (max_exp - min_exp)
        ax.scatter(data['frequency'], trans_norm_full, color='black', s=40, alpha=0.8, label='実験データ', zorder=10)
        
        # H_form結果
        if 'H_form' in model_results and i in model_results['H_form']:
            h_results = model_results['H_form'][i]
            ax.plot(h_results['freq_plot'], h_results['mean_pred'], color='red', lw=4, label='H_form 予測', alpha=0.9)
            ax.fill_between(h_results['freq_plot'], h_results['ci_lower'], h_results['ci_upper'], 
                           color='red', alpha=0.25, label='H_form 95%信用区間')
        
        # B_form結果
        if 'B_form' in model_results and i in model_results['B_form']:
            b_results = model_results['B_form'][i]
            ax.plot(b_results['freq_plot'], b_results['mean_pred'], color='blue', lw=4, label='B_form 予測', alpha=0.9)
            ax.fill_between(b_results['freq_plot'], b_results['ci_lower'], b_results['ci_upper'], 
                           color='blue', alpha=0.25, label='B_form 95%信用区間')

        # 境界線
        ax.axvline(x=0.378, color='blue', linestyle='--', linewidth=3, alpha=0.8, 
                  label='低周波境界 (0.378 THz)' if i == 0 else None)
        ax.axvline(x=0.45, color='red', linestyle='--', linewidth=3, alpha=0.8, 
                  label='高周波境界 (0.45 THz)' if i == 0 else None)

        ax.set_title(f"磁場 {data['b_field']} T", fontsize=28, fontweight='bold')
        ax.set_xlabel('周波数 (THz)', fontsize=24, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.5)
        ax.tick_params(axis='both', which='major', labelsize=20, width=2, length=8)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=18, framealpha=0.9)

    axes[0].set_ylabel('正規化透過率', fontsize=24, fontweight='bold')
    fig.suptitle('H_form vs B_form モデル比較', fontsize=32, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=(0, 0.02, 1, 0.96))
    plt.subplots_adjust(wspace=0.15)

    plt.savefig(IMAGE_DIR / 'combined_model_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    # plt.show()

def plot_model_selection_results(traces: Dict[str, az.InferenceData]):
    """LOO-CVの結果を横棒グラフで出力"""
    print("\n--- モデル選択指標の評価 ---")
    
    model_names = list(traces.keys())
    loo_values = []
    loo_errors = []
    
    # データ収集
    for model_name, trace in traces.items():
        try:
            loo_result = az.loo(trace, pointwise=True)
            loo_value = loo_result.elpd_loo
            loo_se = loo_result.se
            loo_values.append(loo_value)
            loo_errors.append(loo_se)
            print(f"{model_name}: elpd_loo = {loo_value:.2f} ± {loo_se:.2f}")
        except Exception as e:
            print(f"{model_name}: LOO計算に失敗 - {e}")
            loo_values.append(np.nan)
            loo_errors.append(np.nan)
    
    # 横棒グラフの作成
    valid_indices = [i for i, l in enumerate(loo_values) if not np.isnan(l)]
    
    if len(valid_indices) >= 2:
        valid_loo = [loo_values[i] for i in valid_indices]
        valid_loo_err = [loo_errors[i] for i in valid_indices]
        valid_names = [model_names[i] for i in valid_indices]
        
        # 最高値を基準にした相対値を計算
        max_loo = max(valid_loo)
        relative_loo = [loo - max_loo for loo in valid_loo]
        
        # モデル名の順序を調整（B_formが上、H_formが下）
        model_order = []
        rel_values = []
        errors = []
        colors = []
        
        for name, rel_val, err in zip(valid_names, relative_loo, valid_loo_err):
            if name == 'B_form':
                model_order.insert(0, name)
                rel_values.insert(0, rel_val)
                errors.insert(0, err)
                colors.insert(0, 'skyblue')
            else:
                model_order.append(name)
                rel_values.append(rel_val)
                errors.append(err)
                colors.append('lightcoral')
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        y_pos = np.arange(len(model_order))
        
        # 横棒グラフの作成
        bars = []
        for i, (rel_val, err, color) in enumerate(zip(rel_values, errors, colors)):
            bar = ax.barh(y_pos[i], rel_val, xerr=err, color=color, alpha=0.8, 
                         height=0.6, error_kw={'capsize': 5, 'capthick': 2})
            bars.append(bar)
        
        # 基準線（0の位置）
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.8, linewidth=2)
        
        # 軸とラベルの設定
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_order, fontsize=16, fontweight='normal')
        ax.set_xlabel('elpd_loo (log_pointwise_predictive_density)', fontsize=16, fontweight='normal')
        ax.set_title('Model comparison\nhigher is better', fontsize=18, fontweight='normal', pad=20)
        
        # 軸の設定とスタイル調整
        ax.tick_params(axis='x', labelsize=14, width=1, length=5)
        ax.tick_params(axis='y', labelsize=16, width=1, length=5)
        
        # 枠線の設定
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        # グリッドの追加
        ax.grid(True, axis='both', alpha=0.3, linewidth=0.8, color='gray')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
        
        plt.savefig(IMAGE_DIR / 'model_comparison.png', dpi=600, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        # plt.show()
        
        # コンソール表示による詳細比較
        print(f"\n=== LOO-CV モデル選択結果（詳細） ===")
        for i, (name, loo, err) in enumerate(zip(valid_names, valid_loo, valid_loo_err)):
            print(f"{name}: elpd_loo = {loo:.2f} ± {err:.2f}")
        
        # 差の計算と表示
        if len(valid_indices) == 2:
            diff = abs(valid_loo[0] - valid_loo[1])
            se_diff = np.sqrt(valid_loo_err[0]**2 + valid_loo_err[1]**2)
            print(f"\nモデル間差異: {diff:.2f} ± {se_diff:.2f}")
            if diff > 2 * se_diff:
                best_model = valid_names[np.argmax(valid_loo)]
                print(f"統計的有意差: {best_model} が優位")
            else:
                print("統計的有意差: なし")
        else:
            best_idx = np.argmax(valid_loo)
            print(f"\n最良モデル: {valid_names[best_idx]} (elpd_loo = {valid_loo[best_idx]:.2f})")
                
    else:
        print("有効なLOO値が不足しているため、比較できません。")
        
        # 代替のBIC計算
        print("\n代替として、WAIC計算を実行します:")
        for model_name, trace in traces.items():
            try:
                # WAICを使用（より安全）
                waic_result = az.waic(trace, pointwise=True)
                waic_value = waic_result.waic
                waic_se = waic_result.se
                print(f"{model_name}: WAIC = {waic_value:.2f} ± {waic_se:.2f} (小さいほど良い)")
            except Exception as e:
                print(f"{model_name}: WAIC計算に失敗 - {e}")
                # 更なる代替案として簡単な情報量基準を計算
                try:
                    # ArviZ InferenceDataの適切なアクセス方法
                    posterior_data = trace["posterior"] if "posterior" in trace else None
                    if posterior_data is not None:
                        n_samples = posterior_data.chain.size * posterior_data.draw.size
                        n_params = len([v for v in posterior_data.data_vars if isinstance(v, str) and not v.startswith('_')])
                        print(f"{model_name}: パラメータ数 = {n_params}, サンプル数 = {n_samples}")
                    else:
                        print(f"{model_name}: posterior情報の取得に失敗")
                except Exception as ex:
                    print(f"{model_name}: 詳細情報の取得に失敗 - {ex}")

def run_model_comparison_analysis(split_data: Dict[str, List[Dict[str, Any]]], 
                                 all_data_raw: List[Dict[str, Any]], 
                                 field_specific_params: Dict[float, Dict[str, float]], 
                                 existing_h_form_trace: Optional[az.InferenceData] = None) -> Dict[str, az.InferenceData]:
    """H_formとB_formの両方で解析を実行してモデル比較を行う（重複実行を避ける）"""
    print("\n=== H_form と B_form の比較解析を開始 ===")
    
    traces = {}
    
    # H_formの処理：既存の結果があれば再利用、なければ新規実行
    if existing_h_form_trace is not None:
        print("\n--- H_form モデル: 既存の結果を再利用 ---")
        traces['H_form'] = existing_h_form_trace
        print("✅ H_form結果を再利用しました（計算時間短縮）")
        
        # 個別の信用区間プロット
        plot_credible_intervals(all_data_raw, field_specific_params, existing_h_form_trace, 'H_form')
    else:
        print("\n--- H_form モデルでベイズ推定実行 ---")
        try:
            trace = run_bayesian_magnetic_fit_with_fixed_eps(
                split_data['low_freq'], 
                field_specific_params, 
                prior_magnetic_params=None, 
                model_type='H_form'
            )
            traces['H_form'] = trace
            
            # 個別の信用区間プロット
            plot_credible_intervals(all_data_raw, field_specific_params, trace, 'H_form')
            
        except Exception as e:
            print(f"❌ H_form モデルの実行に失敗: {e}")
    
    # B_formの処理：常に新規実行
    print("\n--- B_form モデルでベイズ推定実行 ---")
    try:
        trace = run_bayesian_magnetic_fit_with_fixed_eps(
            split_data['low_freq'], 
            field_specific_params, 
            prior_magnetic_params=None, 
            model_type='B_form'
        )
        traces['B_form'] = trace
        
        # 個別の信用区間プロット
        plot_credible_intervals(all_data_raw, field_specific_params, trace, 'B_form')
        
    except Exception as e:
        print(f"❌ B_form モデルの実行に失敗: {e}")
    
    if len(traces) >= 2:
        # 統合比較プロット
        plot_combined_model_comparison(all_data_raw, field_specific_params, traces)
        
        # モデル選択結果の可視化
        plot_model_selection_results(traces)
        
        # ArviZによる詳細モデル比較
        print("\n=== ArviZによる詳細モデル比較 ===")
        try:
            compare_result = az.compare(traces, ic="loo")
            print(compare_result)
            
            # 比較結果の保存
            compare_result.to_csv(IMAGE_DIR / 'model_comparison_results.csv')
            print(f"比較結果をCSVファイルに保存: {IMAGE_DIR / 'model_comparison_results.csv'}")
            
        except Exception as e:
            print(f"ArviZによる比較に失敗: {e}")
        
        # パラメータ比較
        print("\n=== パラメータ推定値比較 ===")
        for param in ['a_scale', 'g_factor', 'B4', 'B6']:
            print(f"\n{param}:")
            for model_type, trace in traces.items():
                posterior = trace["posterior"]
                mean_val = float(posterior[param].mean())
                std_val = float(posterior[param].std())
                print(f"  {model_type}: {mean_val:.6f} ± {std_val:.6f}")
    
    return traces

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

# --- CSV出力機能の実装 ---
def save_peak_comparison_to_csv(all_datasets: List[Dict[str, Any]], 
                               field_specific_params: Dict[float, Dict[str, float]], 
                               traces: Dict[str, az.InferenceData],
                               results_dir: str = None) -> Dict[str, str]:
    """H形式とB形式のピーク位置の差をCSV形式で出力する"""
    if results_dir is None:
        results_dir = IMAGE_DIR
    
    print("\n--- ピーク位置比較のCSV出力 ---")
    
    # 結果保存ディレクトリの作成
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. 詳細なピーク位置比較データ
    detailed_data = []
    
    for model_type, trace in traces.items():
        print(f"{model_type}モデルのピーク位置を解析中...")
        
        # ベイズ推定パラメータの抽出
        mean_params = extract_bayesian_parameters(trace)
        
        for data in all_datasets:
            b_field = data['b_field']
            
            # 該当磁場のeps_bg取得
            eps_bg_fixed = field_specific_params.get(b_field, {}).get('eps_bg', eps_bg_init)
            
            # 予測透過率の計算
            freq_plot = np.linspace(data['frequency'].min(), data['frequency'].max(), 1000)
            omega_plot = freq_plot * 1e12 * 2 * np.pi
            
            # ハミルトニアン計算
            H = get_hamiltonian(b_field, mean_params['g_factor'], mean_params['B4'], mean_params['B6'])
            gamma_mean = np.full(7, gamma_init)
            chi_raw = calculate_susceptibility(omega_plot, H, TEMPERATURE, gamma_mean)
            chi = mean_params['G0'] * chi_raw
            
            if model_type == 'H_form':
                mu_r = 1 + chi
            else:  # B_form
                mu_r = 1 + chi / (1 + chi)
            
            predicted_trans = calculate_normalized_transmission(omega_plot, mu_r, d_fixed, eps_bg_fixed)
            
            # 実験データの正規化
            min_exp, max_exp = np.min(data['transmittance_full']), np.max(data['transmittance_full'])
            trans_norm_full = (data['transmittance_full'] - min_exp) / (max_exp - min_exp) if max_exp > min_exp else np.full_like(data['transmittance_full'], 0.5)
            
            # ピーク検出（透過率の山 = 正のピーク）
            try:
                exp_peaks, _ = find_peaks(trans_norm_full, height=0.3, distance=5)
                pred_peaks, _ = find_peaks(predicted_trans, height=0.3, distance=10)
                
                exp_peak_freqs = data['frequency'][exp_peaks] if len(exp_peaks) > 0 else np.array([])
                pred_peak_freqs = freq_plot[pred_peaks] if len(pred_peaks) > 0 else np.array([])
                
                # ピークマッチング（最近傍法）
                if len(exp_peak_freqs) > 0 and len(pred_peak_freqs) > 0:
                    tree = KDTree(pred_peak_freqs.reshape(-1, 1))
                    distances, indices = tree.query(exp_peak_freqs.reshape(-1, 1))
                    
                    for i, (exp_freq, dist, idx) in enumerate(zip(exp_peak_freqs, distances.flatten(), indices.flatten())):
                        if dist < 0.1:  # 0.1 THz以内でマッチング
                            pred_freq = pred_peak_freqs[idx]
                            detailed_data.append({
                                'model': model_type,
                                'b_field_T': b_field,
                                'peak_index': i + 1,
                                'exp_peak_freq_THz': exp_freq,
                                'pred_peak_freq_THz': pred_freq,
                                'freq_difference_THz': pred_freq - exp_freq,
                                'relative_error': abs(pred_freq - exp_freq) / exp_freq,
                                'exp_peak_height': trans_norm_full[exp_peaks[i]],
                                'pred_peak_height': predicted_trans[pred_peaks[idx]]
                            })
                        
            except Exception as e:
                print(f"  ピーク検出エラー ({model_type}, {b_field}T): {e}")
    
    # 詳細データをCSVで保存
    detailed_df = pd.DataFrame(detailed_data)
    detailed_csv_path = os.path.join(results_dir, 'peak_position_comparison_detailed.csv')
    detailed_df.to_csv(detailed_csv_path, index=False, encoding='utf-8-sig')
    print(f"  詳細ピーク比較データ: {detailed_csv_path}")
    
    # 2. H形式とB形式の統計比較
    comparison_data = []
    
    if 'H_form' in traces and 'B_form' in traces:
        # 磁場別の統計比較
        b_fields = sorted(list(set([row['b_field_T'] for row in detailed_data])))
        
        for b_field in b_fields:
            h_data = [row for row in detailed_data if row['model'] == 'H_form' and row['b_field_T'] == b_field]
            b_data = [row for row in detailed_data if row['model'] == 'B_form' and row['b_field_T'] == b_field]
            
            if h_data and b_data:
                h_rms = np.sqrt(np.mean([row['freq_difference_THz']**2 for row in h_data]))
                b_rms = np.sqrt(np.mean([row['freq_difference_THz']**2 for row in b_data]))
                h_mean = np.mean([row['freq_difference_THz'] for row in h_data])
                b_mean = np.mean([row['freq_difference_THz'] for row in b_data])
                
                comparison_data.append({
                    'b_field_T': b_field,
                    'H_form_matched_peaks': len(h_data),
                    'B_form_matched_peaks': len(b_data),
                    'H_form_rms_diff_THz': h_rms,
                    'B_form_rms_diff_THz': b_rms,
                    'H_form_mean_diff_THz': h_mean,
                    'B_form_mean_diff_THz': b_mean,
                    'rms_diff_H_vs_B': abs(h_rms - b_rms),
                    'better_model': 'H_form' if h_rms < b_rms else 'B_form'
                })
        
        # モデル比較データをCSVで保存
        comparison_df = pd.DataFrame(comparison_data)
        comparison_csv_path = os.path.join(results_dir, 'H_vs_B_form_peak_comparison.csv')
        comparison_df.to_csv(comparison_csv_path, index=False, encoding='utf-8-sig')
        print(f"  H形式 vs B形式 ピーク位置比較: {comparison_csv_path}")
        
        # 全体サマリー
        print(f"\n=== ピーク位置差分析結果サマリー ===")
        if len(comparison_data) > 0:
            print(f"解析磁場数: {len(comparison_data)}")
            print(f"詳細データ: {len(detailed_data)} 件のピークマッチング")
            avg_h_rms = comparison_df['H_form_rms_diff_THz'].mean()
            avg_b_rms = comparison_df['B_form_rms_diff_THz'].mean()
            print(f"H形式 平均RMS差: {avg_h_rms:.4f} THz")
            print(f"B形式 平均RMS差: {avg_b_rms:.4f} THz")
            
            # 優位モデルの統計
            h_wins = sum(comparison_df['better_model'] == 'H_form')
            b_wins = sum(comparison_df['better_model'] == 'B_form')
            print(f"ピーク位置精度: H形式={h_wins}磁場, B形式={b_wins}磁場で優位")
    else:
        comparison_csv_path = None
    
    print(f"✅ ピーク位置比較結果をCSVファイルに保存（保存先: {results_dir}）")
    
    return {
        'detailed_csv': detailed_csv_path,
        'comparison_csv': comparison_csv_path
    }

def save_fitting_parameters_to_csv(traces: Dict[str, az.InferenceData], 
                                 field_specific_params: Dict[float, Dict[str, float]],
                                 results_dir: str = None) -> Dict[str, str]:
    """H形式とB形式のベイズ推定パラメータをCSV形式で保存する"""
    if results_dir is None:
        results_dir = IMAGE_DIR
        
    print("\n--- フィッティングパラメータのCSV保存 ---")
    
    # 結果保存ディレクトリの作成
    os.makedirs(results_dir, exist_ok=True)
    
    # 各モデルのパラメータを抽出
    model_parameters = {}
    
    for model_type, trace in traces.items():
        print(f"{model_type}モデルのパラメータを抽出中...")
        
        # ベイズ推定パラメータの抽出
        bayesian_params = extract_bayesian_parameters(trace)
        
        # 統計情報も追加抽出
        posterior = trace["posterior"]
        
        # パラメータの統計情報を詳細に取得
        param_stats = {}
        
        # 基本パラメータの統計
        for param_name in ['a_scale', 'g_factor', 'B4', 'B6']:
            if param_name in posterior.data_vars:
                param_data = posterior[param_name]
                param_stats[f'{param_name}_mean'] = float(param_data.mean())
                param_stats[f'{param_name}_std'] = float(param_data.std())
                param_stats[f'{param_name}_hdi_3%'] = float(param_data.quantile(0.03))
                param_stats[f'{param_name}_hdi_97%'] = float(param_data.quantile(0.97))
        
        # gammaパラメータの統計
        if 'log_gamma_mu' in posterior.data_vars:
            gamma_data = posterior['log_gamma_mu']
            param_stats['log_gamma_mu_mean'] = float(gamma_data.mean())
            param_stats['log_gamma_mu_std'] = float(gamma_data.std())
            param_stats['log_gamma_mu_hdi_3%'] = float(gamma_data.quantile(0.03))
            param_stats['log_gamma_mu_hdi_97%'] = float(gamma_data.quantile(0.97))
        
        if 'log_gamma_sigma' in posterior.data_vars:
            sigma_data = posterior['log_gamma_sigma']
            param_stats['log_gamma_sigma_mean'] = float(sigma_data.mean())
            param_stats['log_gamma_sigma_std'] = float(sigma_data.std())
            param_stats['log_gamma_sigma_hdi_3%'] = float(sigma_data.quantile(0.03))
            param_stats['log_gamma_sigma_hdi_97%'] = float(sigma_data.quantile(0.97))
        
        # log_gamma_offset（7次元ベクトル）の統計
        if 'log_gamma_offset' in posterior.data_vars:
            offset_data = posterior['log_gamma_offset']
            for i in range(7):
                param_stats[f'log_gamma_offset_{i}_mean'] = float(offset_data[..., i].mean())
                param_stats[f'log_gamma_offset_{i}_std'] = float(offset_data[..., i].std())
                param_stats[f'log_gamma_offset_{i}_hdi_3%'] = float(offset_data[..., i].quantile(0.03))
                param_stats[f'log_gamma_offset_{i}_hdi_97%'] = float(offset_data[..., i].quantile(0.97))
        
        # 導出パラメータ（G0）を追加
        param_stats['G0_mean'] = bayesian_params.get('G0', 0)
        
        # 不確実性評価指標
        if 'sigma' in posterior.data_vars:
            sigma_data = posterior['sigma']
            param_stats['observation_noise_sigma_mean'] = float(sigma_data.mean())
            param_stats['observation_noise_sigma_std'] = float(sigma_data.std())
        
        model_parameters[model_type] = param_stats
    
    # CSV形式での保存
    # 1. 各モデル別の詳細パラメータファイル
    saved_files = {}
    
    for model_type, params in model_parameters.items():
        df_model = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
        model_csv_path = os.path.join(results_dir, f'fitting_parameters_{model_type.lower()}.csv')
        df_model.to_csv(model_csv_path, index=False, encoding='utf-8-sig')
        print(f"  {model_type}パラメータを保存: {model_csv_path}")
        saved_files[f'{model_type}_params'] = model_csv_path
    
    # 2. モデル比較用の統合ファイル（主要パラメータのみ）
    comparison_data = []
    main_params = ['a_scale_mean', 'g_factor_mean', 'B4_mean', 'B6_mean', 'G0_mean', 
                   'log_gamma_mu_mean', 'log_gamma_sigma_mean', 'observation_noise_sigma_mean']
    
    for model_type, params in model_parameters.items():
        row = {'Model': model_type}
        for param in main_params:
            row[param] = params.get(param, np.nan)
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    comparison_csv_path = os.path.join(results_dir, 'model_comparison_parameters.csv')
    df_comparison.to_csv(comparison_csv_path, index=False, encoding='utf-8-sig')
    print(f"  モデル比較パラメータを保存: {comparison_csv_path}")
    saved_files['model_comparison'] = comparison_csv_path
    
    # 3. 磁場別光学パラメータ（eps_bg）の保存
    field_data = []
    for b_field, params in sorted(field_specific_params.items()):
        field_data.append({
            'B_Field_T': b_field,
            'eps_bg': params.get('eps_bg', np.nan),
            'Fixed_thickness_um': d_fixed * 1e6,  # μm単位
            'Temperature_K': TEMPERATURE
        })
    
    df_field = pd.DataFrame(field_data)
    field_csv_path = os.path.join(results_dir, 'magnetic_field_optical_parameters.csv')
    df_field.to_csv(field_csv_path, index=False, encoding='utf-8-sig')
    print(f"  磁場別光学パラメータを保存: {field_csv_path}")
    saved_files['field_params'] = field_csv_path
    
    print(f"✅ 全てのパラメータをCSVファイルに保存しました（保存先: {results_dir}）")
    
    return saved_files

if __name__ == '__main__':
    print("\n--- 反復的2段階フィッティング解析を開始します ---")
    
    # データ読み込み（新しい3区間分割を使用）
    file_path = "C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\corrected_exp_datasets\\Corrected_Transmittance_B_Field.xlsx"
    sheet_name = 'Corrected Data'
    all_data_raw = load_data_full_range(file_path, sheet_name)
    split_data = load_and_split_data_three_regions(file_path, sheet_name, 
                                                   low_cutoff=0.378,   # 低周波領域: [~, 0.378THz]
                                                   high_cutoff=0.45)   # 高周波領域: [0.45THz, ~]
    
    # 反復的フィッティング実行（高速化版）
    field_specific_params, final_bayesian_trace = iterative_fitting_process(
        split_data, 
        all_data_raw,
        max_iterations=3,        # 反復回数削減（結果から判断すると3回で十分）
        peak_error_threshold=0.020  # 2.0%に緩和（現実的な目標）
    )
    
    if field_specific_params and final_bayesian_trace:
        # 結果の可視化（従来版）
        plot_iterative_results(all_data_raw, field_specific_params, final_bayesian_trace)
        
        # 新機能: H_formとB_formの比較解析（H_form結果を再利用）
        print("\n=== H_form と B_form の比較解析を開始 ===")
        comparison_traces = run_model_comparison_analysis(split_data, all_data_raw, field_specific_params, 
                                                        existing_h_form_trace=final_bayesian_trace)
        
        # 最終結果のサマリー
        print("\n=== 最終結果サマリー ===")
        print("磁場別背景誘電率:")
        for b_field, params in sorted(field_specific_params.items()):
            print(f"  {b_field} T: eps_bg = {params['eps_bg']:.4f}, d = {params['d']*1e6:.2f}μm")
        
        print("\n磁気パラメータ (ベイズ推定):")
        final_magnetic_params = extract_bayesian_parameters(final_bayesian_trace)
        for param, value in final_magnetic_params.items():
            if param == 'G0':
                print(f"  {param} = {value:.3e}")
            else:
                print(f"  {param} = {value:.6f}")
        
        # 最終的なピーク誤差評価
        print("\n=== 最終ピーク位置精度評価 ===")
        final_peak_errors = calculate_peak_errors(all_data_raw, field_specific_params, final_bayesian_trace)
        valid_errors = [(b, err*100) for b, err in final_peak_errors.items() if not np.isinf(err)]
        
        if valid_errors:
            print("各磁場のピーク位置相対誤差:")
            for b_field, error_percent in valid_errors:
                status = "✅" if error_percent < 1.5 else "⚠️"
                print(f"  {b_field} T: {error_percent:.2f}% {status}")
            
            max_error = max([err for _, err in valid_errors])
            avg_error = np.mean([err for _, err in valid_errors])
            print(f"\n統計:")
            print(f"  最大誤差: {max_error:.2f}%")
            print(f"  平均誤差: {avg_error:.2f}%")
            print(f"  目標達成: {'✅ はい' if max_error < 1.5 else '❌ いいえ'} (< 1.5%)")
        else:
            print("❌ ピーク誤差計算に失敗しました。")
        
        print("\n🎉 反復的フィッティング解析が完了しました。")
        
        # H形式とB形式の両方が解析されている場合、CSV出力を実行
        if len(comparison_traces) >= 2:
            print("\n=== CSV形式での結果出力を開始します ===")
            
            # 1. ピーク位置比較データの出力
            peak_csv_files = save_peak_comparison_to_csv(all_data_raw, field_specific_params, comparison_traces)
            print(f"ピーク位置比較結果:")
            for key, path in peak_csv_files.items():
                if path:
                    print(f"  {key}: {path}")
            
            # 2. ベイズ推定パラメータの出力  
            param_csv_files = save_fitting_parameters_to_csv(comparison_traces, field_specific_params)
            print(f"\nベイズ推定パラメータ結果:")
            for key, path in param_csv_files.items():
                print(f"  {key}: {path}")
            
            print(f"\n✅ 全てのCSV出力が完了しました。保存先: {IMAGE_DIR}")
        else:
            print("\n⚠️  H形式とB形式の比較が不完全のため、CSV出力をスキップしました。")
    else:
        print("❌ 反復的フィッティングに失敗しました。")
