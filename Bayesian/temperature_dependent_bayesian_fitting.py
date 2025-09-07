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
from typing import List, Dict, Any, Tuple, Optional
from scipy.signal import find_peaks
from scipy.spatial import KDTree
from scipy.optimize import curve_fit

# --- 0. 環境設定 ---
if __name__ == "__main__":
    print("--- 0. 環境設定を開始します ---")
    os.environ['PYTENSOR_FLAGS'] = 'optimizer=fast_compile,floatX=float64'
    try:
        import japanize_matplotlib
    except ImportError:
        print("警告: japanize_matplotlib が見つかりません。")
    plt.rcParams['figure.dpi'] = 120
    IMAGE_DIR = pathlib.Path(__file__).parent / "temperature_analysis_results"
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

# パラメータの初期値
d_init = 157.8e-6; eps_bg_init = 13.14
B4_init = 0.002; B6_init = -0.00003
gamma_init = 0.11e12; a_scale_init = 1.5; g_factor_init = 2.02

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
    """磁気感受率を計算する（数値安定性を向上）"""
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    eigenvalues -= np.min(eigenvalues)
    
    # 温度が非常に低い場合の数値安定性を考慮
    beta = 1.0 / (kB * T)
    max_exp_arg = 700  # オーバーフロー防止
    
    # エネルギーが大きすぎる状態は除外
    valid_mask = eigenvalues * beta < max_exp_arg
    valid_eigenvalues = eigenvalues[valid_mask]
    
    # ボルツマン分布の計算
    exp_values = np.exp(-valid_eigenvalues * beta)
    Z = np.sum(exp_values)
    
    if Z < 1e-300:  # アンダーフロー防止
        print(f"警告: 分配関数が非常に小さい値です (Z={Z:.2e}, T={T}K)")
        Z = 1e-300
    
    populations = exp_values / Z
    
    # 遷移の計算（有効な状態のみ）
    n_valid = len(valid_eigenvalues)
    if n_valid < 2:
        print(f"警告: 有効な状態が不足しています (T={T}K)")
        return np.zeros_like(omega_array, dtype=complex)
    
    # 隣接状態間の遷移のみを考慮（簡略化）
    delta_E = valid_eigenvalues[1:] - valid_eigenvalues[:-1]
    delta_pop = populations[1:] - populations[:-1]
    omega_0 = delta_E / hbar
    
    # 遷移強度の計算
    m_vals = np.arange(s, -s, -1)[:n_valid-1]
    transition_strength = (s + m_vals) * (s - m_vals + 1)
    
    # gamma_arrayの調整
    if len(gamma_array) != len(delta_E):
        if len(gamma_array) > len(delta_E):
            gamma_array = gamma_array[:len(delta_E)]
        else:
            gamma_array = np.pad(gamma_array, (0, len(delta_E) - len(gamma_array)), 'edge')
    
    # 磁気感受率の計算
    numerator = delta_pop * transition_strength
    denominator = omega_0[:, np.newaxis] - omega_array - 1j * gamma_array[:, np.newaxis] 
    
    # ゼロ除算防止（より厳密）
    denominator_mag = np.abs(denominator)
    small_denom_mask = denominator_mag < 1e-15
    denominator[small_denom_mask] = 1e-15 * np.exp(1j * np.angle(denominator[small_denom_mask]))
    
    # NaN/Inf チェック
    if np.any(np.isnan(denominator)) or np.any(np.isinf(denominator)):
        print("警告: 分母にNaNまたはInfが含まれています")
        denominator = np.where(np.isfinite(denominator), denominator, 1e-15)
    
    chi_array = np.sum(numerator[:, np.newaxis] / denominator, axis=0)
    
    # 最終結果の数値チェック
    if np.any(np.isnan(chi_array)) or np.any(np.isinf(chi_array)):
        print("警告: 磁気感受率の計算結果にNaNまたはInfが含まれています")
        chi_array = np.where(np.isfinite(chi_array), chi_array, 0)
    
    return -chi_array

def calculate_normalized_transmission(omega_array: np.ndarray, mu_r_array: np.ndarray, d: float, eps_bg: float) -> np.ndarray:
    """正規化透過率を計算する"""
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
def load_temperature_data(file_path: str, sheet_name: str, 
                         low_cutoff: float = 0.378, 
                         high_cutoff: float = 0.45) -> Dict[str, List[Dict[str, Any]]]:
    """温度依存データを読み込み、低周波・中間・高周波領域に分割する。
    
    Args:
        file_path: Excelファイルのパス
        sheet_name: シート名
        low_cutoff: 低周波領域の上限 (0.378 THz)
        high_cutoff: 高周波領域の下限 (0.45 THz)
    
    Returns:
        Dict containing:
        - 'low_freq': [~, 0.378THz] - ベイズ推定用
        - 'mid_freq': [0.378THz, 0.45THz] - 中間領域（使用しない）
        - 'high_freq': [0.45THz, ~] - 光学パラメータフィッティング用
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    except Exception as e:
        raise FileNotFoundError(f"Excelファイル '{file_path}' が読み込めません: {e}")
    
    freq_col = 'Frequency (THz)'
    df[freq_col] = pd.to_numeric(df[freq_col], errors='coerce')
    temp_cols = ['4K', '30K', '100K', '300K']  # 温度条件の列名
    
    low_freq_datasets, mid_freq_datasets, high_freq_datasets = [], [], []
    
    for col in temp_cols:
        if col not in df.columns:
            print(f"警告: 列 '{col}' が見つかりません。スキップします。")
            continue
            
        # 温度値を抽出（列名から）
        temp_value = float(col.replace('K', ''))
        
        df_clean = df[[freq_col, col]].dropna()
        freq, trans = df_clean[freq_col].values.astype(np.float64), df_clean[col].values.astype(np.float64)
        
        # 3つの領域にマスクを定義
        low_mask = freq <= low_cutoff
        mid_mask = (freq > low_cutoff) & (freq < high_cutoff)
        high_mask = freq >= high_cutoff
        
        base_data = {'temperature': temp_value, 'b_field': B_FIXED}
        
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
    
    print(f"温度依存データ分割結果:")
    print(f"  低周波領域 [~, {low_cutoff}THz]: {len(low_freq_datasets)} データセット")
    print(f"  中間領域 [{low_cutoff}THz, {high_cutoff}THz]: {len(mid_freq_datasets)} データセット")
    print(f"  高周波領域 [{high_cutoff}THz, ~]: {len(high_freq_datasets)} データセット")
    
    return {
        'low_freq': low_freq_datasets, 
        'mid_freq': mid_freq_datasets,
        'high_freq': high_freq_datasets
    }

def fit_single_temperature_cavity_modes(dataset: Dict[str, Any]) -> Dict[str, float]:
    """各温度で独立に高周波データから光学的・磁気パラメータを決定する"""
    print(f"\n--- 温度 {dataset['temperature']} K の高周波フィッティング ---")
    
    def magnetic_cavity_model(freq_thz, d_fit, eps_bg_fit, g_factor_fit, B4_fit, B6_fit, gamma_scale):
        """磁気感受率を考慮した高周波透過率モデル"""
        try:
            omega = freq_thz * 1e12 * 2 * np.pi
            
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
            
            return calculate_normalized_transmission(omega, mu_r, d_fit, eps_bg_fit)
        except:
            return np.ones_like(freq_thz) * 0.5

    # 複数の初期値と境界条件を試行
    success = False
    result = {}
    
    # 初期値のセット（複数パターン）
    initial_sets = [
        ([d_init, eps_bg_init, g_factor_init, B4_init, B6_init, 1.0],
         ([120e-6, 10.0, 1.9, -0.01, -0.001, 0.1],
          [180e-6, 18.0, 2.1,  0.01,  0.001, 10.0])),
        ([d_init*0.9, eps_bg_init*1.1, g_factor_init, B4_init*2, B6_init*2, 1.5],
         ([100e-6, 8.0, 1.8, -0.015, -0.002, 0.05],
          [200e-6, 25.0, 2.2,  0.015,  0.002, 15.0])),
        ([d_init, eps_bg_init*0.95, 2.0, 0.001, -0.00001, 0.5],
         ([140e-6, 12.0, 1.98, -0.003, -0.0003, 0.3],
          [170e-6, 15.0, 2.02,  0.003,  0.0003, 3.0]))
    ]
    
    for attempt, (p0, bounds) in enumerate(initial_sets):
        try:
            popt, pcov = curve_fit(
                magnetic_cavity_model,
                dataset['frequency'],
                dataset['transmittance'],
                p0=p0,
                bounds=bounds,
                maxfev=5000,
                method='trf'
            )
            
            d_fit, eps_bg_fit, g_factor_fit, B4_fit, B6_fit, gamma_scale_fit = popt
            
            # パラメータが物理的に妥当かチェック
            if (100e-6 <= d_fit <= 200e-6 and 8.0 <= eps_bg_fit <= 25.0 and 
                1.8 <= g_factor_fit <= 2.2 and abs(B4_fit) <= 0.02 and abs(B6_fit) <= 0.005):
                
                result = {
                    'd': d_fit,
                    'eps_bg': eps_bg_fit,
                    'g_factor': g_factor_fit,
                    'B4': B4_fit,
                    'B6': B6_fit,
                    'gamma_scale': gamma_scale_fit,
                    'temperature': dataset['temperature']
                }
                
                print(f"  成功 (試行 {attempt + 1}): d = {d_fit*1e6:.2f} um, eps_bg = {eps_bg_fit:.3f}")
                print(f"  磁気パラメータ: g = {g_factor_fit:.3f}, B4 = {B4_fit:.5f}, B6 = {B6_fit:.6f}")
                print(f"  gamma_scale = {gamma_scale_fit:.3f}")
                
                success = True
                break
            else:
                print(f"  試行 {attempt + 1}: パラメータが物理的範囲外")
                
        except RuntimeError as e:
            print(f"  試行 {attempt + 1}: フィッティング失敗 - {e}")
            continue
        except Exception as e:
            print(f"  試行 {attempt + 1}: 予期しないエラー - {e}")
            continue
    
    if not success:
        print("  ❌ 全ての試行に失敗 - 非磁性モデルを試行")
        # フォールバック: 非磁性モデル
        try:
            def simple_cavity_model(freq_thz, d_fit, eps_bg_fit):
                """非磁性共振器モデル"""
                omega = freq_thz * 1e12 * 2 * np.pi
                mu_r = np.ones_like(omega)  # 磁気効果なし
                return calculate_normalized_transmission(omega, mu_r, d_fit, eps_bg_fit)
            
            p0_simple = [d_init, eps_bg_init]
            bounds_simple = ([120e-6, 10.0], [180e-6, 20.0])
            
            popt_simple, _ = curve_fit(
                simple_cavity_model,
                dataset['frequency'],
                dataset['transmittance'],
                p0=p0_simple,
                bounds=bounds_simple,
                maxfev=5000,
                method='trf'
            )
            
            d_fit, eps_bg_fit = popt_simple
            result = {
                'd': d_fit,
                'eps_bg': eps_bg_fit,
                'g_factor': g_factor_init,  # デフォルト値
                'B4': B4_init,
                'B6': B6_init,
                'gamma_scale': 1.0,
                'temperature': dataset['temperature']
            }
            print(f"  非磁性モデル成功: d = {d_fit*1e6:.2f} um, eps_bg = {eps_bg_fit:.3f}")
            
        except Exception as e:
            print(f"  非磁性モデルも失敗: {e}")
            result = {
                'd': d_init, 'eps_bg': eps_bg_init, 'g_factor': g_factor_init,
                'B4': B4_init, 'B6': B6_init, 'gamma_scale': 1.0,
                'temperature': dataset['temperature']
            }
    
    return result

class TemperatureMagneticModelOp(Op):
    """温度依存の低周波領域の磁気パラメータを推定するためのPyMC Op。"""
    def __init__(self, datasets: List[Dict[str, Any]], temperature_specific_params: Dict[float, Dict[str, float]], model_type: str):
        self.datasets = datasets
        self.temperature_specific_params = temperature_specific_params
        self.model_type = model_type
        self.itypes = [pt.dscalar, pt.dvector, pt.dscalar, pt.dscalar, pt.dscalar]
        self.otypes = [pt.dvector]
    
    def perform(self, node, inputs, output_storage):
        a_scale, gamma, g_factor, B4, B6 = inputs
        full_predicted_y = []
        
        for data in self.datasets:
            # 該当する温度の固定パラメータを取得
            temperature = data['temperature']
            if temperature in self.temperature_specific_params:
                d_fixed = self.temperature_specific_params[temperature]['d']
                eps_bg_fixed = self.temperature_specific_params[temperature]['eps_bg']
            else:
                # フォールバック
                d_fixed = d_init
                eps_bg_fixed = eps_bg_init
            
            H = get_hamiltonian(data['b_field'], g_factor, B4, B6)
            chi_raw = calculate_susceptibility(data['omega'], H, data['temperature'], gamma)
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
                                model_type: str = 'H_form') -> az.InferenceData:
    """温度毎の固定eps_bgを使用してベイズ推定を実行する"""
    print(f"\n--- 温度別固定eps_bgによるベイズ推定 (モデル: {model_type}) ---")
    trans_observed = np.concatenate([d['transmittance'] for d in datasets])
    
    with pm.Model() as model:
        # 事前分布の設定（より制約的に）
        if prior_magnetic_params:
            # 前回のベイズ推定結果を事前分布として使用
            a_scale = pm.TruncatedNormal('a_scale', mu=prior_magnetic_params['a_scale'], sigma=0.2, lower=0.3, upper=3.0)
            g_factor = pm.TruncatedNormal('g_factor', mu=prior_magnetic_params['g_factor'], sigma=0.03, lower=1.98, upper=2.05)
            B4 = pm.Normal('B4', mu=prior_magnetic_params['B4'], sigma=abs(prior_magnetic_params['B4'])*0.3 + 0.0005)
            B6 = pm.Normal('B6', mu=prior_magnetic_params['B6'], sigma=abs(prior_magnetic_params['B6'])*0.3 + 0.00005)
            print(f"前回の推定結果を事前分布として使用:")
            print(f"  a_scale事前分布中心: {prior_magnetic_params['a_scale']:.3f}")
            print(f"  g_factor事前分布中心: {prior_magnetic_params['g_factor']:.3f}")
            print(f"  B4事前分布中心: {prior_magnetic_params['B4']:.5f}")
            print(f"  B6事前分布中心: {prior_magnetic_params['B6']:.6f}")
        else:
            # 初回のデフォルト事前分布（より制約的に）
            a_scale = pm.TruncatedNormal('a_scale', mu=a_scale_init, sigma=0.3, lower=0.3, upper=3.0)
            g_factor = pm.TruncatedNormal('g_factor', mu=g_factor_init, sigma=0.02, lower=1.98, upper=2.05)
            B4 = pm.Normal('B4', mu=B4_init, sigma=abs(B4_init)*0.3 + 0.0005)
            B6 = pm.Normal('B6', mu=B6_init, sigma=abs(B6_init)*0.3 + 0.00005)
            print("初回のデフォルト事前分布を使用")
        
        # γパラメータの事前分布（より安定化）
        log_gamma_mu = pm.Normal('log_gamma_mu', mu=np.log(gamma_init), sigma=0.5)
        log_gamma_sigma = pm.HalfNormal('log_gamma_sigma', sigma=0.3)
        log_gamma_offset = pm.Normal('log_gamma_offset', mu=0, sigma=0.3, shape=7)
        gamma = pm.Deterministic('gamma', pt.exp(log_gamma_mu + log_gamma_offset * log_gamma_sigma))
        
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
            trace = pm.sample(4000,  # サンプル数を増加
                              tune=5000,  # チューニング数を大幅増加
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
                
                # 収束が悪い場合は追加サンプリング
                print("\n収束が不十分なため、追加サンプリングを実行します...")
                trace_extended = pm.sample(2000, 
                                         tune=1000,
                                         chains=4,
                                         cores=min(cpu_count, 4),
                                         target_accept=0.95,
                                         trace=trace,
                                         idata_kwargs={"log_likelihood": True},
                                         random_seed=43,
                                         progressbar=True)
                trace = trace_extended
                
                # 再度収束診断
                print("\n--- 追加サンプリング後の収束診断 ---")
                summary_extended = az.summary(trace, var_names=['a_scale', 'g_factor', 'B4', 'B6'])
                max_rhat_ext = summary_extended['r_hat'].max()
                min_ess_bulk_ext = summary_extended['ess_bulk'].min()
                min_ess_tail_ext = summary_extended['ess_tail'].min()
                
                print(f"最大 r_hat: {max_rhat_ext:.4f}")
                print(f"最小 ess_bulk: {min_ess_bulk_ext:.0f}")
                print(f"最小 ess_tail: {min_ess_tail_ext:.0f}")
            else:
                print("✅ 収束診断: 良好")
                
        except Exception as e:
            print(f"高精度サンプリングに失敗: {e}")
            print("中精度設定でリトライします...")
            try:
                trace = pm.sample(3000, 
                                  tune=4000, 
                                  chains=4, 
                                  cores=min(cpu_count, 4), 
                                  target_accept=0.85,
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
                assert isinstance(trace, az.InferenceData)
    
    print("----------------------------------------------------")
    print("▶ 温度依存ベイズ推定結果 (サマリー):")
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

def load_data_full_range_temperature(file_path: str, sheet_name: str) -> List[Dict[str, Any]]:
    """全周波数範囲の温度依存データを読み込む。"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    except Exception as e:
        raise FileNotFoundError(f"Excelファイル '{file_path}' が読み込めません: {e}")
    
    freq_col = 'Frequency (THz)'
    df[freq_col] = pd.to_numeric(df[freq_col], errors='coerce')
    temp_cols = ['4K', '30K', '100K', '300K']
    
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
                           n_samples: int = 100):
    """温度依存フィッティング結果を95%信用区間と共に可視化する"""
    print(f"\n--- 温度依存フィッティング結果を可視化中 ({model_type}) ---")
    
    posterior = bayesian_trace["posterior"]
    mean_a_scale = float(posterior['a_scale'].mean())
    mean_g_factor = float(posterior['g_factor'].mean())
    mean_B4 = float(posterior['B4'].mean())
    mean_B6 = float(posterior['B6'].mean())
    mean_gamma = posterior['gamma'].mean().values.flatten()
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
            gamma_sample = posterior['gamma'].isel(chain=chain_idx, draw=draw_idx).values.flatten()
            
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
        
        ax.set_title(f"温度 {temperature} K (eps_bg={eps_bg_fixed:.4f}, {model_type})", fontsize=14)
        ax.set_xlabel('周波数 (THz)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    axes[0].set_ylabel('正規化透過率', fontsize=12)
    fig.suptitle(f'温度依存フィッティング結果: 温度別背景誘電率 ({model_type})', fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(IMAGE_DIR / f'temperature_fitting_results_{model_type}.png')
    plt.show()
    
    print("温度依存フィッティング結果の可視化が完了しました。")

def plot_temperature_dependencies(temperature_specific_params: Dict[float, Dict[str, float]], 
                                bayesian_trace: az.InferenceData):
    """温度依存性をプロットする"""
    print("\n--- 温度依存性の可視化 ---")
    
    temperatures = sorted(temperature_specific_params.keys())
    eps_bg_values = [temperature_specific_params[T]['eps_bg'] for T in temperatures]
    d_values = [temperature_specific_params[T]['d']*1e6 for T in temperatures]  # μm単位
    
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
    
    # 膜厚の温度依存性
    ax2.plot(temperatures, d_values, 'bo-', linewidth=2, markersize=8, label='膜厚')
    ax2.set_xlabel('温度 (K)', fontsize=12)
    ax2.set_ylabel('膜厚 (μm)', fontsize=12)
    ax2.set_title('膜厚の温度依存性', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    # 値をテキストで表示
    for T, d in zip(temperatures, d_values):
        ax2.annotate(f'{d:.2f}', (T, d), textcoords="offset points", xytext=(0,10), ha='center')
    
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
    ax4.text(0.1, 0.6, f'膜厚変化率: {(max(d_values)-min(d_values))/min(d_values)*100:.1f}%', fontsize=14, transform=ax4.transAxes)
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
    plt.show()

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
        plt.show()
        
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

if __name__ == '__main__':
    print("\n--- 温度依存ベイズ推定解析を開始します ---")
    
    # データ読み込み
    file_path = "C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\Circular_Polarization_Temparature.xlsx"
    all_data_raw = load_data_full_range_temperature(file_path, 'limited')
    split_data = load_temperature_data(file_path, 'limited', 
                                     low_cutoff=0.378,   # 低周波領域: [~, 0.378THz]
                                     high_cutoff=0.45)   # 高周波領域: [0.45THz, ~]
    
    # Step 1: 各温度で独立に高周波フィッティング
    print("\n=== Step 1: 各温度での独立高周波フィッティング ===")
    temperature_specific_params = {}
    
    for dataset in split_data['high_freq']:
        result = fit_single_temperature_cavity_modes(dataset)
        if result:
            temperature_specific_params[result['temperature']] = result
    
    if not temperature_specific_params:
        print("❌ 高周波フィッティングに失敗しました。プログラムを終了します。")
        exit()
    
    print("\n高周波フィッティング結果:")
    for temp, params in sorted(temperature_specific_params.items()):
        print(f"  {temp} K: d = {params['d']*1e6:.2f} μm, eps_bg = {params['eps_bg']:.4f}")
    
    # Step 2: H_formとB_formの両方でベイズ推定
    print("\n=== Step 2: H_form と B_form のベイズ推定 ===")
    traces = {}
    
    for model_type in ['H_form', 'B_form']:
        print(f"\n--- {model_type} モデルでベイズ推定実行 ---")
        try:
            trace = run_temperature_bayesian_fit(
                split_data['low_freq'], 
                temperature_specific_params, 
                prior_magnetic_params=None, 
                model_type=model_type
            )
            traces[model_type] = trace
            
            # 個別の結果プロット
            plot_temperature_results(all_data_raw, temperature_specific_params, trace, model_type)
            
        except Exception as e:
            print(f"❌ {model_type} モデルの実行に失敗: {e}")
            continue
    
    if traces:
        # 温度依存性のプロット
        # 最初に成功したtraceを使用
        first_trace = list(traces.values())[0]
        plot_temperature_dependencies(temperature_specific_params, first_trace)
        
        # モデル比較（複数のモデルがある場合）
        if len(traces) >= 2:
            plot_model_selection_results_temperature(traces)
            
            print("\n=== モデル比較 ===")
            try:
                compare_result = az.compare(traces, ic="loo")
                print(compare_result)
                
                # WAICによる比較も実行
                print("\n=== WAIC比較 ===")
                for model_name, trace in traces.items():
                    try:
                        waic_result = az.waic(trace, pointwise=True)
                        print(f"{model_name}: WAIC = {waic_result.elpd_waic:.2f} ± {waic_result.se:.2f}")
                    except Exception as e:
                        print(f"{model_name}: WAIC計算に失敗 - {e}")
                        
            except Exception as e:
                print(f"モデル比較に失敗: {e}")
        
        # 最終結果のサマリー
        print("\n=== 最終結果サマリー ===")
        print("温度別光学パラメータ:")
        for temp, params in sorted(temperature_specific_params.items()):
            print(f"  {temp} K: eps_bg = {params['eps_bg']:.4f}, d = {params['d']*1e6:.2f}μm")
        
        print("\n磁気パラメータ (ベイズ推定):")
        final_magnetic_params = extract_bayesian_parameters(first_trace)
        for param, value in final_magnetic_params.items():
            if param == 'G0':
                print(f"  {param} = {value:.3e}")
            else:
                print(f"  {param} = {value:.6f}")
        
        print(f"\n固定磁場: {B_FIXED} T")
        print("🎉 温度依存ベイズ推定解析が完了しました。")
    else:
        print("❌ ベイズ推定に失敗しました。")
