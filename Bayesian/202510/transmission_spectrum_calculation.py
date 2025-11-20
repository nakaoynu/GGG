# transmission_spectrum_calculation.py - 透過率スペクトルT(ω, B, T)計算プログラム
#
# 【概要】
# このプログラムは、重み付きベイズ推定で得られた磁気パラメータを用いて、
# 2つの実験パターンに対応した透過率スペクトルを計算します:
#   ① 温度:変数, 磁場:固定
#   ② 温度:固定, 磁場:変数
#
# 【特徴】
# - γは線形の温度依存性を示す設定
# - ε_bgはそれぞれの磁場・温度で独立に非線形最小二乗法で決定
# - その他の磁気パラメータは重み付きベイズ推定の結果を使用
#
# 【使用方法】
# 1. weighted_bayesian_fitting_completed.py で解析を実行し、結果を保存
# 2. このスクリプトで結果ディレクトリとデータパターンを指定して実行
# 3. 透過率スペクトルが計算され、結果が保存されます

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import pathlib
import yaml
import warnings
from typing import List, Dict, Any, Tuple, Optional, Union
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
kB = 1.380649e-23  # ボルツマン定数 [J/K]
muB = 9.274010e-24  # ボーア磁子 [J/T]
hbar = 1.054571e-34  # 換算プランク定数 [J·s]
c = 299792458  # 光速 [m/s]
mu0 = 4.0 * np.pi * 1e-7  # 真空の透磁率 [H/m]

# --- 設定ファイル読み込み ---
def load_config(config_path: Optional[Union[str, pathlib.Path]] = None) -> Dict[str, Any]:
    """設定ファイル(YAML)を読み込み"""
    if config_path is None:
        config_path = pathlib.Path(__file__).parent / "config.yml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 設定ファイル '{config_path}' を読み込みました。")
        return config
    except Exception as e:
        print(f"❌ 設定ファイルの読み込みに失敗: {e}")
        return {}

# --- ハミルトニアン計算 ---
def get_hamiltonian(B_ext_z: float, g_factor: float, B4: float, B6: float, s: float = 3.5) -> np.ndarray:
    """ハミルトニアンを計算する"""
    n_states = int(2 * s + 1)
    m_values = np.arange(s, -s - 1, -1)
    Sz = np.diag(m_values)
    
    # 結晶場演算子（s=7/2の場合）
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
    
    # ハミルトニアン
    H_cf = (B4 * kB) * (O40 + 5 * O44) + (B6 * kB) * (O60 - 21 * O64)
    H_zee = g_factor * muB * B_ext_z * Sz
    return H_cf + H_zee

# --- 磁気感受率計算 ---
def calculate_susceptibility(omega_array: np.ndarray, H: np.ndarray, T: float, gamma_array: np.ndarray) -> np.ndarray:
    """磁気感受率を計算する（温度依存gamma対応）"""
    # gamma配列の正規化（7要素に統一）
    if np.isscalar(gamma_array):
        gamma_array = np.full(7, gamma_array)
    elif hasattr(gamma_array, 'ndim') and gamma_array.ndim == 0:
        gamma_array = np.full(7, float(gamma_array))
    elif hasattr(gamma_array, '__len__'):
        gamma_array = np.array(gamma_array)
        if len(gamma_array) < 7:
            gamma_array = np.pad(gamma_array, (0, 7 - len(gamma_array)), 'edge')
        elif len(gamma_array) > 7:
            gamma_array = gamma_array[:7]
    
    # 固有値と固有ベクトルを計算
    eigenvalues, _ = np.linalg.eigh(H)
    eigenvalues -= np.min(eigenvalues)
    
    # 数値的安定性のためのクリッピング
    eigenvalues = np.clip(eigenvalues / (kB * T), -700, 700)
    
    # 分配関数と占有確率
    Z = np.sum(np.exp(-eigenvalues))
    populations = np.exp(-eigenvalues) / Z
    delta_E = (eigenvalues[1:] - eigenvalues[:-1]) * kB * T
    delta_pop = populations[1:] - populations[:-1]
    
    # 無効な値をチェック
    valid_mask = np.isfinite(delta_E) & (np.abs(delta_E) > 1e-30)
    if not np.any(valid_mask):
        return np.zeros_like(omega_array, dtype=complex)
    
    omega_0 = delta_E / hbar
    s_val = 3.5  # スピン量子数
    m_vals = np.arange(s_val, -s_val, -1)
    transition_strength = (s_val + m_vals) * (s_val - m_vals + 1)
    
    # gamma配列をdelta_Eと同じ次元に調整
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
    
    # 磁気感受率の計算
    chi_array = np.zeros_like(omega_array, dtype=complex)
    for i, omega in enumerate(omega_array):
        if not np.isfinite(omega):
            continue
        denominator = omega_0_filtered - omega - 1j * gamma_filtered
        denominator[np.abs(denominator) < 1e-20] = 1e-20 + 1j * 1e-20
        chi_array[i] = np.sum(numerator / denominator)
    
    return -chi_array

# --- 透過率計算 ---
def calculate_normalized_transmission(omega_array: np.ndarray, mu_r_array: np.ndarray, 
                                     d: float, eps_bg: float) -> np.ndarray:
    """正規化透過率を計算する"""
    # 入力値の検証
    eps_bg = max(eps_bg, 0.1)
    d = max(d, 1e-6)
    
    # 複素屈折率と impedance の計算
    mu_r_safe = np.where(np.isfinite(mu_r_array), mu_r_array, 1.0)
    eps_mu_product = eps_bg * mu_r_safe
    eps_mu_product = np.where(eps_mu_product.real > 0, eps_mu_product, 0.1 + 1j * eps_mu_product.imag)
    
    n_complex = np.sqrt(eps_mu_product + 0j)
    impe = np.sqrt(mu_r_safe / eps_bg + 0j)
    
    # 波長計算
    lambda_0 = np.full_like(omega_array, np.inf, dtype=float)
    nonzero_mask = omega_array > 1e-12
    lambda_0[nonzero_mask] = (2 * np.pi * c) / omega_array[nonzero_mask]
    
    # 位相計算
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
    transmission = np.clip(transmission, 0, 2)
    
    # 正規化
    min_trans, max_trans = np.min(transmission), np.max(transmission)
    if max_trans > min_trans and np.isfinite(max_trans) and np.isfinite(min_trans):
        return (transmission - min_trans) / (max_trans - min_trans)
    else:
        return np.full_like(transmission, 0.5)

# --- γの温度依存性計算 ---
def calculate_gamma_for_temperature(T: float, bayesian_params: Dict[str, float], 
                                   base_temp: float = 4.0) -> np.ndarray:
    """温度Tでのγ値を計算（線形温度依存性）"""
    log_gamma_mu_base = bayesian_params['log_gamma_mu_base']
    log_gamma_sigma_base = bayesian_params.get('log_gamma_sigma_base', 0.0)
    log_gamma_offset_base = bayesian_params.get('log_gamma_offset_base', np.zeros(7))
    temp_gamma_slope = bayesian_params['temp_gamma_slope']
    
    # 温度補正（線形項のみ）
    temp_diff = T - base_temp
    temp_correction = temp_gamma_slope * temp_diff
    log_gamma_mu_temp = log_gamma_mu_base + temp_correction
    
    # γ配列の計算
    gamma_array = np.exp(log_gamma_mu_temp + log_gamma_offset_base * log_gamma_sigma_base)
    return gamma_array

# --- eps_bgフィッティング ---
def fit_eps_bg(freq_thz: np.ndarray, trans_obs: np.ndarray, 
              B: float, T: float, d: float,
              bayesian_params: Dict[str, float], 
              physical_params: Dict[str, float],
              model_type: str = 'H_form') -> float:
    """高周波データからeps_bgをフィッティング"""
    
    def model_func(freq_thz, eps_bg_fit):
        """eps_bgのみを変数とするモデル関数"""
        try:
            omega = freq_thz * 1e12 * 2 * np.pi
            
            # パラメータ取得
            g_factor = bayesian_params['g_factor']
            B4 = bayesian_params['B4']
            B6 = bayesian_params['B6']
            a_scale = bayesian_params['a_scale']
            N_spin = physical_params['N_spin']
            
            # ハミルトニアンと磁気感受率
            H = get_hamiltonian(B, g_factor, B4, B6)
            gamma_array = calculate_gamma_for_temperature(T, bayesian_params)
            chi_raw = calculate_susceptibility(omega, H, T, gamma_array)
            
            # スケーリング
            G0 = a_scale * mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
            chi = G0 * chi_raw
            
            # モデル形式に応じた透磁率
            if model_type == 'B_form':
                mu_r = 1 / (1 - chi)
            else:  # H_form
                mu_r = 1 + chi
            
            return calculate_normalized_transmission(omega, mu_r, d, eps_bg_fit)
        except:
            return np.ones_like(freq_thz) * 0.5
    
    # 初期値と境界の設定（温度依存）
    if T <= 10:
        eps_bg_init_vals = [14.20 * 0.90, 14.20 * 0.95, 14.20, 13.0, 13.5]
        bounds = (11.0, 16.0)
    elif T <= 100:
        eps_bg_init_vals = [14.20, 14.20 * 1.02, 13.8, 14.2, 13.5]
        bounds = (11.5, 16.5)
    else:
        eps_bg_init_vals = [14.20 * 1.05, 14.20 * 1.10, 14.5, 15.0, 14.0]
        bounds = (12.0, 17.0)
    
    # 複数の初期値で試行
    for eps_bg_init in eps_bg_init_vals:
        try:
            popt, _ = curve_fit(model_func, freq_thz, trans_obs,
                               p0=[eps_bg_init], bounds=([bounds[0]], [bounds[1]]),
                               maxfev=3000, method='trf')
            eps_bg_fit = popt[0]
            if bounds[0] <= eps_bg_fit <= bounds[1]:
                return eps_bg_fit
        except:
            continue
    
    # 失敗時はデフォルト値
    return 14.20

# --- 透過率スペクトル計算（パターン①: 温度変数、磁場固定） ---
def calculate_transmission_spectrum_temp_variable(
    freq_range: np.ndarray,
    temp_range: np.ndarray,
    B_fixed: float,
    d_fixed: float,
    bayesian_params: Dict[str, float],
    physical_params: Dict[str, float],
    model_type: str = 'H_form',
    high_freq_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    パターン①: 温度変数、磁場固定の透過率スペクトルを計算
    
    Args:
        freq_range: 周波数配列 [THz]
        temp_range: 温度配列 [K]
        B_fixed: 固定磁場 [T]
        d_fixed: 膜厚 [m]
        bayesian_params: ベイズ推定で得られた磁気パラメータ
        physical_params: 物理定数 (N_spin, s)
        model_type: 'H_form' or 'B_form'
        high_freq_data: 高周波データ（eps_bgフィッティング用）、なければ固定値使用
    
    Returns:
        結果辞書
    """
    print(f"\n{'='*60}")
    print(f"パターン①: 温度変数、磁場固定 (B={B_fixed}T, モデル:{model_type})")
    print(f"{'='*60}")
    
    omega_range = freq_range * 1e12 * 2 * np.pi
    results = {
        'frequency': freq_range,
        'temperature': temp_range,
        'B_fixed': B_fixed,
        'model_type': model_type,
        'spectra': {},
        'eps_bg_values': {}
    }
    
    # パラメータ取得
    g_factor = bayesian_params['g_factor']
    B4 = bayesian_params['B4']
    B6 = bayesian_params['B6']
    a_scale = bayesian_params['a_scale']
    N_spin = physical_params['N_spin']
    
    # 各温度でのスペクトル計算
    for T in temp_range:
        print(f"\n温度 {T}K での計算中...")
        
        # eps_bgの決定
        if high_freq_data is not None:
            # 高周波データからフィッティング
            temp_data = high_freq_data[high_freq_data['temperature'] == T]
            if not temp_data.empty:
                freq_hf = temp_data['frequency'].values
                trans_hf = temp_data['transmittance'].values
                eps_bg = fit_eps_bg(freq_hf, trans_hf, B_fixed, T, d_fixed,
                                   bayesian_params, physical_params, model_type)
                print(f"  eps_bg (フィッティング): {eps_bg:.4f}")
            else:
                eps_bg = 14.20
                print(f"  eps_bg (デフォルト): {eps_bg:.4f}")
        else:
            eps_bg = 14.20
            print(f"  eps_bg (デフォルト): {eps_bg:.4f}")
        
        results['eps_bg_values'][T] = eps_bg
        
        # ハミルトニアンと磁気感受率の計算
        H = get_hamiltonian(B_fixed, g_factor, B4, B6)
        gamma_array = calculate_gamma_for_temperature(T, bayesian_params)
        chi_raw = calculate_susceptibility(omega_range, H, T, gamma_array)
        
        # スケーリング
        G0 = a_scale * mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
        chi = G0 * chi_raw
        
        # モデル形式に応じた透磁率
        if model_type == 'B_form':
            mu_r = 1 / (1 - chi)
        else:  # H_form
            mu_r = 1 + chi
        
        # 透過率計算
        transmission = calculate_normalized_transmission(omega_range, mu_r, d_fixed, eps_bg)
        results['spectra'][T] = transmission
        
        print(f"  スペクトル計算完了 (データ点数: {len(transmission)})")
    
    return results

# --- 透過率スペクトル計算（パターン②: 温度固定、磁場変数） ---
def calculate_transmission_spectrum_field_variable(
    freq_range: np.ndarray,
    B_range: np.ndarray,
    T_fixed: float,
    d_fixed: float,
    bayesian_params: Dict[str, float],
    physical_params: Dict[str, float],
    model_type: str = 'H_form',
    high_freq_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    パターン②: 温度固定、磁場変数の透過率スペクトルを計算
    
    Args:
        freq_range: 周波数配列 [THz]
        B_range: 磁場配列 [T]
        T_fixed: 固定温度 [K]
        d_fixed: 膜厚 [m]
        bayesian_params: ベイズ推定で得られた磁気パラメータ
        physical_params: 物理定数 (N_spin, s)
        model_type: 'H_form' or 'B_form'
        high_freq_data: 高周波データ（eps_bgフィッティング用）、なければ固定値使用
    
    Returns:
        結果辞書
    """
    print(f"\n{'='*60}")
    print(f"パターン②: 温度固定、磁場変数 (T={T_fixed}K, モデル:{model_type})")
    print(f"{'='*60}")
    
    omega_range = freq_range * 1e12 * 2 * np.pi
    results = {
        'frequency': freq_range,
        'B_field': B_range,
        'T_fixed': T_fixed,
        'model_type': model_type,
        'spectra': {},
        'eps_bg_values': {}
    }
    
    # パラメータ取得
    g_factor = bayesian_params['g_factor']
    B4 = bayesian_params['B4']
    B6 = bayesian_params['B6']
    a_scale = bayesian_params['a_scale']
    N_spin = physical_params['N_spin']
    
    # γ値の計算（温度固定なので1回だけ）
    gamma_array = calculate_gamma_for_temperature(T_fixed, bayesian_params)
    
    # 各磁場でのスペクトル計算
    for B in B_range:
        print(f"\n磁場 {B}T での計算中...")
        
        # eps_bgの決定
        if high_freq_data is not None:
            # 高周波データからフィッティング
            field_data = high_freq_data[high_freq_data['B_field'] == B]
            if not field_data.empty:
                freq_hf = field_data['frequency'].values
                trans_hf = field_data['transmittance'].values
                eps_bg = fit_eps_bg(freq_hf, trans_hf, B, T_fixed, d_fixed,
                                   bayesian_params, physical_params, model_type)
                print(f"  eps_bg (フィッティング): {eps_bg:.4f}")
            else:
                eps_bg = 14.20
                print(f"  eps_bg (デフォルト): {eps_bg:.4f}")
        else:
            eps_bg = 14.20
            print(f"  eps_bg (デフォルト): {eps_bg:.4f}")
        
        results['eps_bg_values'][B] = eps_bg
        
        # ハミルトニアンと磁気感受率の計算
        H = get_hamiltonian(B, g_factor, B4, B6)
        chi_raw = calculate_susceptibility(omega_range, H, T_fixed, gamma_array)
        
        # スケーリング
        G0 = a_scale * mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
        chi = G0 * chi_raw
        
        # モデル形式に応じた透磁率
        if model_type == 'B_form':
            mu_r = 1 / (1 - chi)
        else:  # H_form
            mu_r = 1 + chi
        
        # 透過率計算
        transmission = calculate_normalized_transmission(omega_range, mu_r, d_fixed, eps_bg)
        results['spectra'][B] = transmission
        
        print(f"  スペクトル計算完了 (データ点数: {len(transmission)})")
    
    return results

# --- 結果保存 ---
def save_results(results: Dict[str, Any], output_dir: pathlib.Path, pattern_name: str):
    """結果をCSVとグラフで保存"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSVファイル保存
    if 'temperature' in results:
        # パターン①
        df_list = []
        for T, spectrum in results['spectra'].items():
            df_temp = pd.DataFrame({
                'frequency_THz': results['frequency'],
                'temperature_K': T,
                'B_field_T': results['B_fixed'],
                'transmittance': spectrum,
                'eps_bg': results['eps_bg_values'][T]
            })
            df_list.append(df_temp)
        df = pd.concat(df_list, ignore_index=True)
    else:
        # パターン②
        df_list = []
        for B, spectrum in results['spectra'].items():
            df_temp = pd.DataFrame({
                'frequency_THz': results['frequency'],
                'temperature_K': results['T_fixed'],
                'B_field_T': B,
                'transmittance': spectrum,
                'eps_bg': results['eps_bg_values'][B]
            })
            df_list.append(df_temp)
        df = pd.concat(df_list, ignore_index=True)
    
    csv_path = output_dir / f"{pattern_name}_{results['model_type']}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 結果をCSVに保存: {csv_path}")
    
    # グラフ作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    if 'temperature' in results:
        # パターン①: 温度変数
        for T, spectrum in results['spectra'].items():
            ax1.plot(results['frequency'], spectrum, label=f'{T}K', linewidth=2)
        ax1.set_xlabel('周波数 (THz)', fontsize=12)
        ax1.set_ylabel('正規化透過率', fontsize=12)
        ax1.set_title(f'{pattern_name} - {results["model_type"]}\n(B={results["B_fixed"]}T)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # eps_bg vs 温度
        temps = sorted(results['eps_bg_values'].keys())
        eps_bgs = [results['eps_bg_values'][T] for T in temps]
        ax2.plot(temps, eps_bgs, 'o-', linewidth=2, markersize=8)
        ax2.set_xlabel('温度 (K)', fontsize=12)
        ax2.set_ylabel('ε_bg', fontsize=12)
        ax2.set_title('背景誘電率の温度依存性', fontsize=14)
        ax2.grid(True, alpha=0.3)
    else:
        # パターン②: 磁場変数
        for B, spectrum in results['spectra'].items():
            ax1.plot(results['frequency'], spectrum, label=f'{B}T', linewidth=2)
        ax1.set_xlabel('周波数 (THz)', fontsize=12)
        ax1.set_ylabel('正規化透過率', fontsize=12)
        ax1.set_title(f'{pattern_name} - {results["model_type"]}\n(T={results["T_fixed"]}K)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # eps_bg vs 磁場
        fields = sorted(results['eps_bg_values'].keys())
        eps_bgs = [results['eps_bg_values'][B] for B in fields]
        ax2.plot(fields, eps_bgs, 's-', linewidth=2, markersize=8)
        ax2.set_xlabel('磁場 (T)', fontsize=12)
        ax2.set_ylabel('ε_bg', fontsize=12)
        ax2.set_title('背景誘電率の磁場依存性', fontsize=14)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = output_dir / f"{pattern_name}_{results['model_type']}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✅ グラフを保存: {fig_path}")
    plt.close()

# --- メイン実行関数 ---
def main():
    """メイン実行関数"""
    print("="*70)
    print("透過率スペクトル T(ω, B, T) 計算プログラム")
    print("="*70)
    
    # 設定ファイル読み込み
    config = load_config()
    if not config:
        print("❌ 設定ファイルが読み込めませんでした。終了します。")
        return
    
    # 物理パラメータ
    physical_params = {
        'N_spin': config['physical_parameters']['N_spin'],
        's': config['physical_parameters']['s'],
        'd_fixed': config['physical_parameters']['d_fixed'],
        'B_fixed': config['physical_parameters']['B_fixed']
    }
    
    # 結果ディレクトリの指定（最新のrunを使用）
    results_parent = pathlib.Path(__file__).parent / config['file_paths']['results_parent_dir']
    if not results_parent.exists():
        print(f"❌ 結果ディレクトリが見つかりません: {results_parent}")
        return
    
    # 最新のrun_XXXXXXディレクトリを取得
    run_dirs = sorted([d for d in results_parent.iterdir() if d.is_dir() and d.name.startswith('run_')])
    if not run_dirs:
        print(f"❌ run_ディレクトリが見つかりません: {results_parent}")
        return
    
    latest_run = run_dirs[-1]
    print(f"\n📁 使用する結果ディレクトリ: {latest_run.name}")
    
    # ベイズ推定結果の読み込み
    model_types = ['H_form', 'B_form']
    bayesian_results = {}
    
    for model_type in model_types:
        trace_file = latest_run / f"trace_{model_type}.nc"
        if not trace_file.exists():
            print(f"⚠️ {model_type}のtraceファイルが見つかりません: {trace_file}")
            continue
        
        try:
            trace = az.from_netcdf(trace_file)
            posterior = trace["posterior"]
            
            # パラメータ抽出
            bayesian_results[model_type] = {
                'a_scale': posterior['a_scale'].mean().item(),
                'g_factor': posterior['g_factor'].mean().item(),
                'B4': posterior['B4'].mean().item(),
                'B6': posterior['B6'].mean().item(),
                'log_gamma_mu_base': posterior['log_gamma_mu_base'].mean().item(),
                'log_gamma_sigma_base': posterior.get('log_gamma_sigma_base', az.xr.DataArray([0.0])).mean().item(),
                'log_gamma_offset_base': posterior.get('log_gamma_offset_base', az.xr.DataArray(np.zeros(7))).mean(dim=['chain', 'draw']).values,
                'temp_gamma_slope': posterior['temp_gamma_slope'].mean().item()
            }
            print(f"✅ {model_type}のパラメータを読み込みました")
            
        except Exception as e:
            print(f"❌ {model_type}のパラメータ読み込みに失敗: {e}")
    
    if not bayesian_results:
        print("❌ ベイズ推定結果が読み込めませんでした。終了します。")
        return
    
    # 周波数範囲の設定
    freq_range = np.linspace(0.3, 0.5, 500)  # 0.3-0.5 THz, 500点
    
    # 出力ディレクトリ
    output_dir = latest_run / "transmission_spectra"
    output_dir.mkdir(exist_ok=True)
    
    # パターン①: 温度変数、磁場固定
    print("\n" + "="*70)
    print("パターン①: 温度変数、磁場固定 の計算")
    print("="*70)
    
    temp_range = np.array([4, 30, 100, 300])  # 測定温度
    B_fixed = physical_params['B_fixed']
    d_fixed = physical_params['d_fixed']
    
    for model_type, params in bayesian_results.items():
        results_temp = calculate_transmission_spectrum_temp_variable(
            freq_range=freq_range,
            temp_range=temp_range,
            B_fixed=B_fixed,
            d_fixed=d_fixed,
            bayesian_params=params,
            physical_params=physical_params,
            model_type=model_type,
            high_freq_data=None  # 必要に応じて実データを渡す
        )
        save_results(results_temp, output_dir, "pattern1_temp_variable")
    
    # パターン②: 温度固定、磁場変数
    print("\n" + "="*70)
    print("パターン②: 温度固定、磁場変数 の計算")
    print("="*70)
    
    B_range = np.linspace(0, 15, 16)  # 0-15 T, 16点
    T_fixed = 4.0  # 固定温度 [K]
    
    for model_type, params in bayesian_results.items():
        results_field = calculate_transmission_spectrum_field_variable(
            freq_range=freq_range,
            B_range=B_range,
            T_fixed=T_fixed,
            d_fixed=d_fixed,
            bayesian_params=params,
            physical_params=physical_params,
            model_type=model_type,
            high_freq_data=None  # 必要に応じて実データを渡す
        )
        save_results(results_field, output_dir, "pattern2_field_variable")
    
    print("\n" + "="*70)
    print("🎉 全ての計算が完了しました")
    print(f"📁 結果は '{output_dir}' に保存されています")
    print("="*70)

if __name__ == "__main__":
    main()
