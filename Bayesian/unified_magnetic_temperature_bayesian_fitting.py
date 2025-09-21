# unified_magnetic_temperature_bayesian_fitting.py - 磁場・温度依存性統合ベイズ推定フィッティング

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
IMAGE_DIR = pathlib.Path(__file__).parent / "unified_magnetic_temperature_results"

if __name__ == "__main__":
    print("--- 0. 環境設定を開始します ---")
    os.environ['PYTENSOR_FLAGS'] = 'optimizer=fast_compile,floatX=float64'
    try:
        import japanize_matplotlib
    except ImportError:
        print("警告: japanize_matplotlib が見つかりません。")
    plt.rcParams['figure.dpi'] = 120
    IMAGE_DIR.mkdir(exist_ok=True)
    print(f"画像は '{IMAGE_DIR.resolve()}' に保存されます。")

# --- 1. 物理定数とシミュレーション条件 ---
if __name__ == "__main__":
    print("--- 1. 物理定数と初期値を設定します ---")

kB = 1.380649e-23; muB = 9.274010e-24; hbar = 1.054571e-34
c = 299792458; mu0 = 4.0 * np.pi * 1e-7
s = 3.5; N_spin = 24 / 1.238 * 1e27

# パラメータの初期値
d_fixed = 157.8e-6  # 膜厚は固定値として使用
eps_bg_init = 14.20
B4_init = 0.001149; B6_init = -0.000010
gamma_init = 0.11e12; a_scale_init = 0.604971; g_factor_init = 2.015445

# --- データファイル設定 ---
# 磁場依存データ
B_FIELD_DATA_FILE = "C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\corrected_exp_datasets\\Corrected_Transmittance_B_Field.xlsx"
B_FIELD_SHEET_NAME = "Corrected Data"

# 温度依存データ
TEMP_DATA_FILE = "C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\corrected_exp_datasets\\Corrected_Transmittance_Temperature.xlsx"
TEMP_SHEET_NAME = "Corrected Data"

# 測定条件
TEMPERATURE_COLUMNS = ['4K', '30K', '100K', '300K']
B_FIXED_FOR_TEMP = 9.0  # 温度依存測定時の固定磁場 (T)
TEMP_FIXED_FOR_B = 1.5   # 磁場依存測定時の固定温度 (K)

# 周波数領域分割の閾値設定
LOW_FREQUENCY_CUTOFF_B = 0.378     # THz - 磁場依存データのベイズ推定で使用する低周波領域の上限
LOW_FREQUENCY_CUTOFF_T = 0.361505  # THz - 温度依存データのベイズ推定で使用する低周波領域の上限
HIGH_FREQUENCY_CUTOFF = 0.45       # THz - eps_bgフィッティングで使用する高周波領域の下限

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

def calculate_susceptibility_robust(omega_array: np.ndarray, H: np.ndarray, T: float, gamma_value: float) -> np.ndarray:
    """磁気感受率を計算する（堅牢版 - NaN/inf対策強化）"""
    
    # 入力値の検証
    if not np.isfinite(gamma_value) or gamma_value <= 0:
        return np.zeros_like(omega_array, dtype=complex)
    
    if not np.isfinite(T) or T <= 0:
        return np.zeros_like(omega_array, dtype=complex)
    
    # gamma_valueを安全な範囲に制限
    gamma_value = np.clip(gamma_value, 1e9, 1e15)
    
    # ハミルトニアンの固有値計算（エラーハンドリング付き）
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eigenvalues, _ = np.linalg.eigh(H)
    except:
        return np.zeros_like(omega_array, dtype=complex)
    
    # 固有値の有効性チェック
    if not np.all(np.isfinite(eigenvalues)):
        return np.zeros_like(omega_array, dtype=complex)
    
    # エネルギーの最小値を0に設定
    eigenvalues -= np.min(eigenvalues)
    
    # ボルツマン因子の計算（オーバーフロー防止）
    exp_arg = -eigenvalues / (kB * T)
    exp_arg = np.clip(exp_arg, -700, 700)  # 数値オーバーフロー防止
    
    try:
        populations = np.exp(exp_arg)
        Z = np.sum(populations)
        
        if Z == 0 or not np.isfinite(Z):
            return np.zeros_like(omega_array, dtype=complex)
        
        populations /= Z
        
    except:
        return np.zeros_like(omega_array, dtype=complex)
    
    # 遷移エネルギーと占有数差の計算
    delta_E = eigenvalues[1:] - eigenvalues[:-1]
    delta_pop = populations[1:] - populations[:-1]
    omega_0 = delta_E / hbar
    
    # 遷移強度の計算
    m_vals = np.arange(s, -s, -1)
    transition_strength = (s + m_vals) * (s - m_vals + 1)
    
    # 分子の計算
    numerator = delta_pop * transition_strength
    
    # 有効な遷移のみをフィルタリング
    valid_mask = (np.isfinite(numerator) & 
                  np.isfinite(omega_0) & 
                  (np.abs(numerator) > 1e-30) &
                  (np.abs(omega_0) > 1e-30))
    
    if not np.any(valid_mask):
        return np.zeros_like(omega_array, dtype=complex)
    
    numerator_valid = numerator[valid_mask]
    omega_0_valid = omega_0[valid_mask]
    
    # 各周波数点での磁化率計算（ベクトル化）
    chi_array = np.zeros_like(omega_array, dtype=complex)
    
    try:
        # ベクトル化された計算で高速化
        omega_mesh, omega0_mesh = np.meshgrid(omega_array, omega_0_valid)
        denominator = omega0_mesh - omega_mesh - 1j * gamma_value
        
        # 分母が0に近い場合の処理
        small_denom_mask = np.abs(denominator) < 1e-20
        denominator[small_denom_mask] = 1e-20 + 1j * 1e-20
        
        # 磁化率の計算
        contributions = numerator_valid[:, np.newaxis] / denominator
        chi_array = np.sum(contributions, axis=0)
        
        # 結果の有効性チェック
        if not np.all(np.isfinite(chi_array)):
            return np.zeros_like(omega_array, dtype=complex)
        
    except:
        return np.zeros_like(omega_array, dtype=complex)
    
    return -chi_array

def calculate_normalized_transmission(omega_array: np.ndarray, mu_r_array: np.ndarray, d: float, eps_bg: float) -> np.ndarray:
    """正規化透過率を計算する（改良版：数値安定性とピーク位置精度の向上）"""
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

# --- 3. データ読み込み関数 ---
# --- 3. 統一データ読み込み・正規化関数 ---
def load_and_normalize_all_data() -> List[Dict[str, Any]]:
    """全データセットを読み込み、統一的に正規化する"""
    print("\n--- データセットの統一読み込みと正規化 ---")
    
    all_data_frames = []
    
    # 磁場依存データの読み込み
    try:
        df_b = pd.read_excel(B_FIELD_DATA_FILE, sheet_name=B_FIELD_SHEET_NAME, header=0)
        freq_col = 'Frequency (THz)'
        df_b[freq_col] = pd.to_numeric(df_b[freq_col], errors='coerce')
        b_trans_cols = [col for col in df_b.columns if 'Transmittance' in col and 'T)' in col]
        
        for col in b_trans_cols:
            b_match = re.search(r'\((\d+\.?\d*)T\)', col)
            if not b_match: 
                continue
            b_value = float(b_match.group(1))
            
            temp_df = df_b[[freq_col, col]].copy().dropna()
            temp_df.columns = ['frequency', 'transmittance']
            temp_df['b_field'] = b_value
            temp_df['temperature'] = TEMP_FIXED_FOR_B
            temp_df['data_type'] = 'magnetic_field'
            
            # 磁場依存データは LOW_FREQUENCY_CUTOFF_B を使用
            temp_df = temp_df[temp_df['frequency'] <= LOW_FREQUENCY_CUTOFF_B]
            if len(temp_df) > 0:
                all_data_frames.append(temp_df)
        
        print(f"磁場依存データ: {len([df for df in all_data_frames if df['data_type'].iloc[0] == 'magnetic_field'])} データセット")
    
    except Exception as e:
        print(f"磁場依存データの読み込みエラー: {e}")
        return []
    
    # 温度依存データの読み込み
    try:
        df_t = pd.read_excel(TEMP_DATA_FILE, sheet_name=TEMP_SHEET_NAME, header=0)
        df_t[freq_col] = pd.to_numeric(df_t[freq_col], errors='coerce')
        
        for col in TEMPERATURE_COLUMNS:
            if col not in df_t.columns:
                continue
            temp_value = float(col.replace('K', ''))
            
            temp_df = df_t[[freq_col, col]].copy().dropna()
            temp_df.columns = ['frequency', 'transmittance']
            temp_df['b_field'] = B_FIXED_FOR_TEMP
            temp_df['temperature'] = temp_value
            temp_df['data_type'] = 'temperature'
            
            # 温度依存データは LOW_FREQUENCY_CUTOFF_T を使用
            temp_df = temp_df[temp_df['frequency'] <= LOW_FREQUENCY_CUTOFF_T]
            if len(temp_df) > 0:
                all_data_frames.append(temp_df)
        
        print(f"温度依存データ: {len([df for df in all_data_frames if df['data_type'].iloc[0] == 'temperature'])} データセット")
    
    except Exception as e:
        print(f"温度依存データの読み込みエラー: {e}")
        return []
    
    if not all_data_frames:
        print("エラー: データが読み込めませんでした")
        return []
    
    # 全データを結合
    full_df = pd.concat(all_data_frames, ignore_index=True)
    
    # 統一正規化: 全データの最大値・最小値を使用
    min_trans_global = full_df['transmittance'].min()
    max_trans_global = full_df['transmittance'].max()
    
    if max_trans_global > min_trans_global:
        full_df['transmittance_norm'] = ((full_df['transmittance'] - min_trans_global) / 
                                        (max_trans_global - min_trans_global))
    else:
        full_df['transmittance_norm'] = 0.5  # 全て同じ値の場合
    
    print(f"透過率の正規化: min={min_trans_global:.4f}, max={max_trans_global:.4f}")
    
    # データセットリストに再構成
    datasets = []
    for (b_field, temperature, data_type), group in full_df.groupby(['b_field', 'temperature', 'data_type']):
        freq = group['frequency'].values.astype(np.float64)
        trans_norm = group['transmittance_norm'].values.astype(np.float64)
        
        # NaN値のチェック
        valid_mask = np.isfinite(freq) & np.isfinite(trans_norm)
        if not np.any(valid_mask):
            continue
        
        freq = freq[valid_mask]
        trans_norm = trans_norm[valid_mask]
        
        # 周波数でソート
        sort_idx = np.argsort(freq)
        freq = freq[sort_idx]
        trans_norm = trans_norm[sort_idx]
        
        datasets.append({
            'data_type': data_type,
            'b_field': float(b_field),
            'temperature': float(temperature),
            'frequency': freq,
            'transmittance': trans_norm,
            'omega': freq * 1e12 * 2 * np.pi
        })
    
    print(f"統一正規化完了: 合計 {len(datasets)} データセット")
    return datasets

# --- 4. 統合PyMC Op ---
class UnifiedMagneticTemperatureModelOp(Op):
    """磁場・温度依存性を統合したPyMC Op（緩和係数の磁場・温度2次依存性対応）"""
    
    def __init__(self, datasets: List[Dict[str, Any]], model_type: str = 'H_form'):
        self.datasets = datasets
        self.model_type = model_type
        
        # 入力パラメータの定義
        # [a_scale, g_factor, B4, B6, gamma_base, gamma_B, gamma_B2, gamma_T, gamma_T2, gamma_BT]
        self.itypes = [pt.dscalar] * 10
        self.otypes = [pt.dvector]
    
    def perform(self, node, inputs, output_storage):
        a_scale, g_factor, B4, B6, gamma_base, gamma_B, gamma_B2, gamma_T, gamma_T2, gamma_BT = inputs
        
        full_predicted_y = []
        
        for data in self.datasets:
            b_field = data['b_field']
            temperature = data['temperature']
            omega_array = data['omega']
            
            try:
                # ハミルトニアンの計算
                if self.model_type == 'B_form':
                    # B_formでは磁場をB4, B6に含めて扱う
                    H = get_hamiltonian(b_field, g_factor, B4 * b_field, B6 * b_field)
                else:
                    # H_formでは従来通り
                    H = get_hamiltonian(b_field, g_factor, B4, B6)
                
                # 緩和係数の磁場・温度2次依存性計算
                # gamma = gamma_base + gamma_B*B + gamma_B2*B^2 + gamma_T*T + gamma_T2*T^2 + gamma_BT*B*T
                gamma_value = (gamma_base + 
                              gamma_B * b_field + 
                              gamma_B2 * b_field**2 + 
                              gamma_T * temperature + 
                              gamma_T2 * temperature**2 + 
                              gamma_BT * b_field * temperature)
                
                # 物理的制約（正値）
                gamma_value = max(gamma_value, 1e10)  # 最小値設定
                
                # 各遷移に対する緩和係数（単一値）
                # gamma_array = np.full(7, gamma_value)  # 削除 - 堅牢版では不要
                
                # 磁気感受率の計算
                chi = calculate_susceptibility_robust(omega_array, H, temperature, gamma_value)
                
                # 相対透磁率の計算（モデル依存・物理的修正版）
                G0 = a_scale * mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
                
                if self.model_type == 'B_form':
                    # B_form: mu_r = 1 / (1 - G0 * chi)（物理的修正版）
                    denominator = 1 - G0 * chi
                    # 数値安定性のため、分母が0に近い場合の処理
                    denominator = np.where(np.abs(denominator) < 1e-10, 
                                         np.sign(denominator.real) * 1e-10 + 1j * denominator.imag, 
                                         denominator)
                    mu_r = 1 / denominator
                else:
                    # H_form: mu_r = 1 + G0 * chi
                    mu_r = 1 + G0 * chi
                
                # 正規化透過率の計算（背景誘電率は固定）
                predicted_y = calculate_normalized_transmission(omega_array, mu_r, d_fixed, eps_bg_init)
                
            except Exception as e:
                print(f"計算エラー (B={b_field}T, T={temperature}K): {e}")
                predicted_y = np.full(len(omega_array), 0.5)
            
            full_predicted_y.append(predicted_y)
        
        output_storage[0][0] = np.concatenate(full_predicted_y)

# --- 5. 統合ベイズ推定関数 ---
def run_unified_bayesian_fit(magnetic_datasets: List[Dict[str, Any]], 
                           temperature_datasets: List[Dict[str, Any]], 
                           model_type: str = 'H_form') -> az.InferenceData:
    """磁場・温度依存性を統合したベイズ推定を実行"""
    print(f"\n--- 磁場・温度統合ベイズ推定 (モデル: {model_type}) ---")
    
    # データセットを結合
    all_datasets = magnetic_datasets + temperature_datasets
    trans_observed = np.concatenate([d['transmittance'] for d in all_datasets])
    
    print(f"統合データセット: 磁場依存 {len(magnetic_datasets)} + 温度依存 {len(temperature_datasets)} = 合計 {len(all_datasets)}")
    print(f"観測データ点数: {len(trans_observed)}")
    
    with pm.Model() as model:
        # 基本磁気パラメータの事前分布（安定化）
        a_scale = pm.TruncatedNormal('a_scale', mu=a_scale_init, sigma=0.2, lower=0.1, upper=2.0)
        g_factor = pm.TruncatedNormal('g_factor', mu=g_factor_init, sigma=0.015, lower=1.99, upper=2.04)
        B4 = pm.Normal('B4', mu=B4_init, sigma=abs(B4_init)*0.3 + 0.0003)
        B6 = pm.Normal('B6', mu=B6_init, sigma=abs(B6_init)*0.3 + 0.00003)
        
        # 緩和係数の磁場・温度依存性パラメータ（安定化）
        # gamma = gamma_base + gamma_B*B + gamma_B2*B^2 + gamma_T*T + gamma_T2*T^2 + gamma_BT*B*T
        gamma_base = pm.TruncatedNormal('gamma_base', mu=5e11, sigma=2e11, lower=1e10, upper=5e12)
        
        # 磁場依存性（1次・2次）- より保守的な範囲
        gamma_B = pm.Normal('gamma_B', mu=0.0, sigma=5e10)  # B線形項
        gamma_B2 = pm.Normal('gamma_B2', mu=0.0, sigma=5e9)  # B^2項
        
        # 温度依存性（1次・2次）- より保守的な範囲
        gamma_T = pm.Normal('gamma_T', mu=0.0, sigma=5e8)   # T線形項
        gamma_T2 = pm.Normal('gamma_T2', mu=0.0, sigma=1e6)  # T^2項
        
        # 磁場・温度相互作用項
        gamma_BT = pm.Normal('gamma_BT', mu=0.0, sigma=1e7)  # B*T項
        
        # 物理モデルのOp
        op = UnifiedMagneticTemperatureModelOp(all_datasets, model_type)
        mu = op(a_scale, g_factor, B4, B6, gamma_base, gamma_B, gamma_B2, gamma_T, gamma_T2, gamma_BT)
        
        # 観測ノイズ（やや大きめに設定して安定化）
        sigma = pm.HalfNormal('sigma', sigma=0.1)
        
        # 観測モデル
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=trans_observed)
        
        # サンプリング設定
        cpu_count = os.cpu_count() or 4
        try:
            print("統合ベイズサンプリングを開始します...")
            trace = pm.sample(1500,  # サンプル数を減らして安定化
                              tune=1500,
                              chains=4,
                              cores=min(cpu_count, 2),  # 過負荷防止
                              target_accept=0.80,  # より保守的な設定
                              init='jitter+adapt_diag',  # 安定した初期化
                              idata_kwargs={"log_likelihood": True},
                              random_seed=42,
                              progressbar=True,
                              return_inferencedata=True)
            
            # 収束診断
            print("\n--- 収束診断 ---")
            summary = az.summary(trace, var_names=['a_scale', 'g_factor', 'B4', 'B6', 'gamma_base', 'gamma_B', 'gamma_B2', 'gamma_T', 'gamma_T2', 'gamma_BT'])
            max_rhat = summary['r_hat'].max()
            min_ess_bulk = summary['ess_bulk'].min()
            
            print(f"最大 r_hat: {max_rhat:.4f}")
            print(f"最小 ess_bulk: {min_ess_bulk:.0f}")
            
            if max_rhat < 1.01 and min_ess_bulk > 400:
                print("✅ 収束診断: 良好")
            else:
                print("⚠️ 警告: 収束に問題がある可能性があります")
                
        except Exception as e:
            print(f"高精度サンプリングに失敗: {e}")
            print("低精度設定でリトライします...")
            trace = pm.sample(2000, 
                              tune=2000, 
                              chains=2, 
                              cores=1,
                              target_accept=0.8,
                              idata_kwargs={"log_likelihood": True}, 
                              random_seed=42,
                              progressbar=True,
                              return_inferencedata=True)

        # log_likelihoodの計算確認
        with model:
            if "log_likelihood" not in trace:
                computed = pm.compute_log_likelihood(trace)
                if isinstance(computed, az.InferenceData):
                    trace = computed
                else:
                    # Datasetの場合はInferenceDataに変換
                    trace = az.InferenceData(computed)
    
    print("----------------------------------------------------")
    print("▶ 統合ベイズ推定結果 (サマリー):")
    try:
        summary = az.summary(trace, var_names=['a_scale', 'g_factor', 'B4', 'B6', 'gamma_base', 'gamma_B', 'gamma_B2', 'gamma_T', 'gamma_T2', 'gamma_BT'])
        print(summary)
    except Exception as e:
        print(f"サマリー表示エラー: {e}")
    print("----------------------------------------------------")
    
    return trace

# --- 6. 結果可視化関数 ---
def plot_unified_results(magnetic_datasets: List[Dict[str, Any]], 
                        temperature_datasets: List[Dict[str, Any]], 
                        trace: az.InferenceData, 
                        model_type: str = 'H_form',
                        n_samples: int = 100):
    """統合フィッティング結果の可視化"""
    print(f"\n--- 統合フィッティング結果の可視化 ({model_type}) ---")
    
    posterior = trace["posterior"]
    all_datasets = magnetic_datasets + temperature_datasets
    
    # 磁場依存と温度依存に分けてプロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # --- 磁場依存プロット ---
    ax1.set_title('磁場依存性フィッティング結果', fontsize=14, fontweight='bold')
    colors_b = plt.get_cmap('viridis')(np.linspace(0, 1, len(magnetic_datasets)))
    
    for i, (data, color) in enumerate(zip(magnetic_datasets, colors_b)):
        freq = data['frequency']
        trans_exp = data['transmittance']
        b_field = data['b_field']
        
        # 実験データプロット
        ax1.plot(freq, trans_exp, 'o', color=color, markersize=4, alpha=0.7, 
                label=f'実験 {b_field}T')
        
        # 理論曲線（平均）
        freq_plot = np.linspace(freq.min(), freq.max(), 200)
        omega_plot = freq_plot * 1e12 * 2 * np.pi
        
        # 予測計算のためのダミーデータセット
        pred_data = {**data, 'frequency': freq_plot, 'omega': omega_plot}
        
        # サンプルからの予測
        predictions = []
        total_samples = posterior.chain.size * posterior.draw.size
        indices = np.random.choice(total_samples, min(n_samples, total_samples), replace=False)
        
        for idx in indices:
            chain_idx = idx // posterior.draw.size
            draw_idx = idx % posterior.draw.size
            
            params = {
                'a_scale': float(posterior['a_scale'][chain_idx, draw_idx]),
                'g_factor': float(posterior['g_factor'][chain_idx, draw_idx]),
                'B4': float(posterior['B4'][chain_idx, draw_idx]),
                'B6': float(posterior['B6'][chain_idx, draw_idx]),
                'gamma_base': float(posterior['gamma_base'][chain_idx, draw_idx]),
                'gamma_B': float(posterior['gamma_B'][chain_idx, draw_idx]),
                'gamma_B2': float(posterior['gamma_B2'][chain_idx, draw_idx]),
                'gamma_T': float(posterior['gamma_T'][chain_idx, draw_idx]),
                'gamma_T2': float(posterior['gamma_T2'][chain_idx, draw_idx]),
                'gamma_BT': float(posterior['gamma_BT'][chain_idx, draw_idx])
            }
            
            pred_y = calculate_single_prediction(pred_data, params, model_type)
            predictions.append(pred_y)
        
        # 95%信用区間
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        lower_95 = np.percentile(predictions, 2.5, axis=0)
        upper_95 = np.percentile(predictions, 97.5, axis=0)
        
        ax1.plot(freq_plot, mean_pred, '-', color=color, linewidth=2)
        ax1.fill_between(freq_plot, lower_95, upper_95, color=color, alpha=0.2)
    
    ax1.set_xlabel('周波数 (THz)', fontsize=12)
    ax1.set_ylabel('正規化透過率', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # --- 温度依存プロット ---
    ax2.set_title('温度依存性フィッティング結果', fontsize=14, fontweight='bold')
    colors_t = plt.get_cmap('plasma')(np.linspace(0, 1, len(temperature_datasets)))

    for i, (data, color) in enumerate(zip(temperature_datasets, colors_t)):
        freq = data['frequency']
        trans_exp = data['transmittance']
        temperature = data['temperature']
        
        # 実験データプロット
        ax2.plot(freq, trans_exp, 's', color=color, markersize=4, alpha=0.7, 
                label=f'実験 {temperature}K')
        
        # 理論曲線（同様の処理）
        freq_plot = np.linspace(freq.min(), freq.max(), 200)
        omega_plot = freq_plot * 1e12 * 2 * np.pi
        pred_data = {**data, 'frequency': freq_plot, 'omega': omega_plot}
        
        predictions = []
        total_samples = posterior.chain.size * posterior.draw.size
        indices = np.random.choice(total_samples, min(n_samples, total_samples), replace=False)
        
        for idx in indices:
            chain_idx = idx // posterior.draw.size
            draw_idx = idx % posterior.draw.size
            
            params = {
                'a_scale': float(posterior['a_scale'][chain_idx, draw_idx]),
                'g_factor': float(posterior['g_factor'][chain_idx, draw_idx]),
                'B4': float(posterior['B4'][chain_idx, draw_idx]),
                'B6': float(posterior['B6'][chain_idx, draw_idx]),
                'gamma_base': float(posterior['gamma_base'][chain_idx, draw_idx]),
                'gamma_B': float(posterior['gamma_B'][chain_idx, draw_idx]),
                'gamma_B2': float(posterior['gamma_B2'][chain_idx, draw_idx]),
                'gamma_T': float(posterior['gamma_T'][chain_idx, draw_idx]),
                'gamma_T2': float(posterior['gamma_T2'][chain_idx, draw_idx]),
                'gamma_BT': float(posterior['gamma_BT'][chain_idx, draw_idx])
            }
            
            pred_y = calculate_single_prediction(pred_data, params, model_type)
            predictions.append(pred_y)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        lower_95 = np.percentile(predictions, 2.5, axis=0)
        upper_95 = np.percentile(predictions, 97.5, axis=0)
        
        ax2.plot(freq_plot, mean_pred, '-', color=color, linewidth=2)
        ax2.fill_between(freq_plot, lower_95, upper_95, color=color, alpha=0.2)
    
    ax2.set_xlabel('周波数 (THz)', fontsize=12)
    ax2.set_ylabel('正規化透過率', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / f'unified_fitting_results_{model_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_single_prediction(data: Dict[str, Any], params: Dict[str, float], model_type: str) -> np.ndarray:
    """単一データセットに対する予測計算（可視化用）"""
    b_field = data['b_field']
    temperature = data['temperature']
    omega_array = data['omega']
    
    try:
        # ハミルトニアンの計算
        if model_type == 'B_form':
            H = get_hamiltonian(b_field, params['g_factor'], params['B4'] * b_field, params['B6'] * b_field)
        else:
            H = get_hamiltonian(b_field, params['g_factor'], params['B4'], params['B6'])
        
        # 緩和係数の磁場・温度2次依存性計算
        gamma_value = (params['gamma_base'] + 
                      params['gamma_B'] * b_field + 
                      params['gamma_B2'] * b_field**2 + 
                      params['gamma_T'] * temperature + 
                      params['gamma_T2'] * temperature**2 + 
                      params['gamma_BT'] * b_field * temperature)
        gamma_value = max(gamma_value, 1e10)
        # gamma_array = np.full(7, gamma_value)  # 削除 - 堅牢版では不要
        
        # 磁気感受率の計算
        chi = calculate_susceptibility_robust(omega_array, H, temperature, gamma_value)
        
        # 相対透磁率の計算（モデル依存・物理的修正版）
        G0 = params['a_scale'] * mu0 * N_spin * (params['g_factor'] * muB)**2 / (2 * hbar)
        
        if model_type == 'B_form':
            # B_form: mu_r = 1 / (1 - G0 * chi)（物理的修正版）
            denominator = 1 - G0 * chi
            # 数値安定性のため、分母が0に近い場合の処理
            denominator = np.where(np.abs(denominator) < 1e-10, 
                                 np.sign(denominator.real) * 1e-10 + 1j * denominator.imag, 
                                 denominator)
            mu_r = 1 / denominator
        else:
            # H_form: mu_r = 1 + G0 * chi
            mu_r = 1 + G0 * chi
        
        # 正規化透過率の計算
        predicted_y = calculate_normalized_transmission(omega_array, mu_r, d_fixed, eps_bg_init)
        
    except Exception as e:
        print(f"予測計算エラー (B={b_field}T, T={temperature}K): {e}")
        predicted_y = np.full(len(omega_array), 0.5)
    
    return predicted_y

def plot_gamma_dependencies(trace: az.InferenceData):
    """緩和係数の磁場・温度依存性を可視化"""
    print("\n--- 緩和係数の磁場・温度依存性を可視化 ---")
    
    posterior = trace["posterior"]
    
    # パラメータの平均値を取得
    gamma_base_mean = float(posterior['gamma_base'].mean())
    gamma_B_mean = float(posterior['gamma_B'].mean())
    gamma_B2_mean = float(posterior['gamma_B2'].mean())
    gamma_T_mean = float(posterior['gamma_T'].mean())
    gamma_T2_mean = float(posterior['gamma_T2'].mean())
    gamma_BT_mean = float(posterior['gamma_BT'].mean())
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- 磁場依存性（温度固定） ---
    B_range = np.linspace(0, 15, 100)
    T_fixed = 1.5  # K
    
    gamma_B_dep = (gamma_base_mean + 
                   gamma_B_mean * B_range + 
                   gamma_B2_mean * B_range**2 + 
                   gamma_T_mean * T_fixed + 
                   gamma_T2_mean * T_fixed**2 + 
                   gamma_BT_mean * B_range * T_fixed)
    
    ax1.plot(B_range, gamma_B_dep / 1e12, 'b-', linewidth=2)
    ax1.set_xlabel('磁場 (T)', fontsize=12)
    ax1.set_ylabel('緩和係数 γ (THz)', fontsize=12)
    ax1.set_title(f'磁場依存性 (T = {T_fixed}K)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # --- 温度依存性（磁場固定） ---
    T_range = np.linspace(1, 300, 100)
    B_fixed = 9.0  # T
    
    gamma_T_dep = (gamma_base_mean + 
                   gamma_B_mean * B_fixed + 
                   gamma_B2_mean * B_fixed**2 + 
                   gamma_T_mean * T_range + 
                   gamma_T2_mean * T_range**2 + 
                   gamma_BT_mean * B_fixed * T_range)
    
    ax2.plot(T_range, gamma_T_dep / 1e12, 'r-', linewidth=2)
    ax2.set_xlabel('温度 (K)', fontsize=12)
    ax2.set_ylabel('緩和係数 γ (THz)', fontsize=12)
    ax2.set_title(f'温度依存性 (B = {B_fixed}T)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # --- 2D依存性（ヒートマップ） ---
    B_2d = np.linspace(0, 15, 50)
    T_2d = np.linspace(1, 300, 50)
    B_mesh, T_mesh = np.meshgrid(B_2d, T_2d)
    
    gamma_2d = (gamma_base_mean + 
                gamma_B_mean * B_mesh + 
                gamma_B2_mean * B_mesh**2 + 
                gamma_T_mean * T_mesh + 
                gamma_T2_mean * T_mesh**2 + 
                gamma_BT_mean * B_mesh * T_mesh)
    
    im = ax3.contourf(B_mesh, T_mesh, gamma_2d / 1e12, levels=20, cmap='viridis')
    ax3.set_xlabel('磁場 (T)', fontsize=12)
    ax3.set_ylabel('温度 (K)', fontsize=12)
    ax3.set_title('緩和係数の2D依存性', fontsize=14)
    plt.colorbar(im, ax=ax3, label='γ (THz)')
    
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / 'gamma_dependencies.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # パラメータ値を表示
    print("緩和係数依存性パラメータ:")
    print(f"  γ_base = {gamma_base_mean/1e12:.3f} THz")
    print(f"  γ_B = {gamma_B_mean/1e12:.3f} THz/T")
    print(f"  γ_B2 = {gamma_B2_mean/1e12:.4f} THz/T²")
    print(f"  γ_T = {gamma_T_mean/1e12:.4f} THz/K")
    print(f"  γ_T2 = {gamma_T2_mean/1e12:.6f} THz/K²")
    print(f"  γ_BT = {gamma_BT_mean/1e12:.4f} THz/(T·K)")

# --- 7. モデル比較とLOO-CV関数 ---
def run_model_comparison(magnetic_datasets: List[Dict[str, Any]], 
                        temperature_datasets: List[Dict[str, Any]]) -> Dict[str, az.InferenceData]:
    """H_formとB_formの比較とLOO-CVによる模型選択"""
    print("\n=== モデル比較とLOO-CV解析 ===")
    
    traces = {}
    
    # H_formでの推定
    print("\n--- H_formモデルでの統合ベイズ推定 ---")
    traces['H_form'] = run_unified_bayesian_fit(magnetic_datasets, temperature_datasets, 'H_form')
    
    # B_formでの推定
    print("\n--- B_formモデルでの統合ベイズ推定 ---")
    traces['B_form'] = run_unified_bayesian_fit(magnetic_datasets, temperature_datasets, 'B_form')
    
    # LOO-CV比較
    print("\n--- LOO-CV (Leave-One-Out Cross-Validation) 比較 ---")
    loo_results = {}
    
    for model_name, trace in traces.items():
        try:
            loo = az.loo(trace, pointwise=True)
            loo_results[model_name] = loo
            print(f"{model_name} LOO:")
            print(f"  ELPD LOO: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
            print(f"  p_loo: {loo.p_loo:.2f}")
            if hasattr(loo, 'pareto_k'):
                n_high_k = np.sum(loo.pareto_k > 0.7)
                print(f"  High Pareto k (>0.7): {n_high_k}/{len(loo.pareto_k)}")
        except Exception as e:
            print(f"{model_name} LOO計算エラー: {e}")
    
    # モデル比較
    if len(loo_results) == 2:
        try:
            comparison = az.compare(loo_results)
            print("\n--- モデル比較結果 ---")
            print(comparison)
            
            # 最良モデルの特定
            best_model = comparison.index[0]
            delta_loo = abs(float(comparison.loc[comparison.index[1], 'elpd_diff']))
            se_diff = abs(float(comparison.loc[comparison.index[1], 'se']))
            
            print(f"\n最良モデル: {best_model}")
            print(f"ELPD差: {delta_loo:.2f} ± {se_diff:.2f}")
            
            if abs(delta_loo) > 2 * se_diff:
                print("→ 統計的に有意な差があります")
            else:
                print("→ 統計的に有意な差はありません")
                
        except Exception as e:
            print(f"モデル比較エラー: {e}")
    
    return traces

def plot_model_comparison(magnetic_datasets: List[Dict[str, Any]], 
                         temperature_datasets: List[Dict[str, Any]], 
                         traces: Dict[str, az.InferenceData], 
                         n_samples: int = 50):
    """H_formとB_formの予測曲線比較（95%信用区間付き）"""
    print(f"\n--- モデル比較可視化 ({list(traces.keys())}) ---")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    colors = {'H_form': 'blue', 'B_form': 'red'}
    line_styles = {'H_form': '-', 'B_form': '--'}
    
    # --- 磁場依存比較 ---
    ax1.set_title('磁場依存性: モデル比較', fontsize=14, fontweight='bold')
    
    for i, data in enumerate(magnetic_datasets[:3]):  # 最初の3データセットのみ表示
        freq = data['frequency']
        trans_exp = data['transmittance']
        b_field = data['b_field']
        
        # 実験データプロット
        ax1.plot(freq, trans_exp, 'o', color='black', markersize=4, alpha=0.6, 
                label=f'実験 {b_field}T' if i == 0 else "")
        
        for model_name, trace in traces.items():
            color = colors[model_name]
            style = line_styles[model_name]
            
            # 予測計算
            freq_plot = np.linspace(freq.min(), freq.max(), 100)
            omega_plot = freq_plot * 1e12 * 2 * np.pi
            pred_data = {**data, 'frequency': freq_plot, 'omega': omega_plot}
            
            predictions = calculate_predictions_for_comparison(pred_data, trace, model_name, n_samples)
            
            if len(predictions) > 0:
                mean_pred = np.mean(predictions, axis=0)
                lower_95 = np.percentile(predictions, 2.5, axis=0)
                upper_95 = np.percentile(predictions, 97.5, axis=0)
                
                label = f'{model_name} {b_field}T' if i == 0 else ""
                ax1.plot(freq_plot, mean_pred, style, color=color, linewidth=2, label=label)
                ax1.fill_between(freq_plot, lower_95, upper_95, color=color, alpha=0.15)
    
    ax1.set_xlabel('周波数 (THz)', fontsize=12)
    ax1.set_ylabel('正規化透過率', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- 温度依存比較 ---
    ax2.set_title('温度依存性: モデル比較', fontsize=14, fontweight='bold')
    
    for i, data in enumerate(temperature_datasets):
        freq = data['frequency']
        trans_exp = data['transmittance']
        temperature = data['temperature']
        
        # 実験データプロット
        ax2.plot(freq, trans_exp, 's', color='black', markersize=4, alpha=0.6, 
                label=f'実験 {temperature}K' if i == 0 else "")
        
        for model_name, trace in traces.items():
            color = colors[model_name]
            style = line_styles[model_name]
            
            # 予測計算
            freq_plot = np.linspace(freq.min(), freq.max(), 100)
            omega_plot = freq_plot * 1e12 * 2 * np.pi
            pred_data = {**data, 'frequency': freq_plot, 'omega': omega_plot}
            
            predictions = calculate_predictions_for_comparison(pred_data, trace, model_name, n_samples)
            
            if len(predictions) > 0:
                mean_pred = np.mean(predictions, axis=0)
                lower_95 = np.percentile(predictions, 2.5, axis=0)
                upper_95 = np.percentile(predictions, 97.5, axis=0)
                
                label = f'{model_name} {temperature}K' if i == 0 else ""
                ax2.plot(freq_plot, mean_pred, style, color=color, linewidth=2, label=label)
                ax2.fill_between(freq_plot, lower_95, upper_95, color=color, alpha=0.15)
    
    ax2.set_xlabel('周波数 (THz)', fontsize=12)
    ax2.set_ylabel('正規化透過率', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # --- パラメータ比較 ---
    ax3.set_title('パラメータ事後分布比較', fontsize=14, fontweight='bold')
    
    param_names = ['a_scale', 'g_factor', 'B4', 'B6']
    x_pos = np.arange(len(param_names))
    width = 0.35
    
    for i, (model_name, trace) in enumerate(traces.items()):
        posterior = trace["posterior"]
        means = [float(posterior[param].mean()) for param in param_names]
        stds = [float(posterior[param].std()) for param in param_names]
        
        color = colors[model_name]
        ax3.bar(x_pos + i*width, means, width, yerr=stds, 
               label=model_name, color=color, alpha=0.7)
    
    ax3.set_xlabel('パラメータ', fontsize=12)
    ax3.set_ylabel('値', fontsize=12)
    ax3.set_xticks(x_pos + width/2)
    ax3.set_xticklabels(param_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # --- 残差比較 ---
    ax4.set_title('モデル残差比較', fontsize=14, fontweight='bold')
    
    all_datasets = magnetic_datasets + temperature_datasets
    residuals = {}
    
    for model_name, trace in traces.items():
        model_residuals = []
        for data in all_datasets:
            freq = data['frequency']
            trans_exp = data['transmittance']
            
            # 予測計算
            predictions = calculate_predictions_for_comparison(data, trace, model_name, 10)
            if len(predictions) > 0:
                mean_pred = np.mean(predictions, axis=0)
                residual = trans_exp - mean_pred
                model_residuals.extend(residual)
        
        if model_residuals:
            residuals[model_name] = np.array(model_residuals)
    
    for model_name, res in residuals.items():
        color = colors[model_name]
        ax4.hist(res, bins=30, alpha=0.6, label=f'{model_name} (σ={np.std(res):.4f})', 
                color=color, density=True)
    
    ax4.set_xlabel('残差', fontsize=12)
    ax4.set_ylabel('密度', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / 'model_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_predictions_for_comparison(data: Dict[str, Any], trace: az.InferenceData, 
                                       model_type: str, n_samples: int = 50) -> np.ndarray:
    """モデル比較用の予測計算"""
    try:
        posterior = trace["posterior"]
        predictions = []
        
        total_samples = posterior.chain.size * posterior.draw.size
        indices = np.random.choice(total_samples, min(n_samples, total_samples), replace=False)
        
        for idx in indices:
            chain_idx = idx // posterior.draw.size
            draw_idx = idx % posterior.draw.size
            
            params = {
                'a_scale': float(posterior['a_scale'][chain_idx, draw_idx]),
                'g_factor': float(posterior['g_factor'][chain_idx, draw_idx]),
                'B4': float(posterior['B4'][chain_idx, draw_idx]),
                'B6': float(posterior['B6'][chain_idx, draw_idx]),
                'gamma_base': float(posterior['gamma_base'][chain_idx, draw_idx]),
                'gamma_B': float(posterior['gamma_B'][chain_idx, draw_idx]),
                'gamma_B2': float(posterior['gamma_B2'][chain_idx, draw_idx]),
                'gamma_T': float(posterior['gamma_T'][chain_idx, draw_idx]),
                'gamma_T2': float(posterior['gamma_T2'][chain_idx, draw_idx]),
                'gamma_BT': float(posterior['gamma_BT'][chain_idx, draw_idx])
            }
            
            pred_y = calculate_single_prediction(data, params, model_type)
            predictions.append(pred_y)
        
        return np.array(predictions)
    
    except Exception as e:
        print(f"予測計算エラー ({model_type}): {e}")
        return np.array([])

# --- 8. メイン実行関数 ---
if __name__ == '__main__':
    print("\n=== 磁場・温度依存性統合ベイズ推定解析を開始します ===")
    
    # 統一データ読み込み
    print("\n--- 統一データ読み込み ---")
    all_datasets = load_and_normalize_all_data()
    
    if not all_datasets:
        print("エラー: データが読み込めませんでした。")
        exit(1)
    
    # データセットを磁場依存と温度依存に分類
    magnetic_datasets = [d for d in all_datasets if d['data_type'] == 'magnetic_field']
    temperature_datasets = [d for d in all_datasets if d['data_type'] == 'temperature']
    
    print(f"磁場依存データセット: {len(magnetic_datasets)}")
    print(f"温度依存データセット: {len(temperature_datasets)}")
    
    # モデル比較実行（H_formとB_formの両方）
    print("\n--- H_formとB_formの比較解析 ---")
    traces = run_model_comparison(magnetic_datasets, temperature_datasets)
    
    # 比較結果の可視化
    print("\n--- 比較結果可視化 ---")
    plot_model_comparison(magnetic_datasets, temperature_datasets, traces)
    
    # 個別モデルの詳細可視化
    for model_name, trace in traces.items():
        print(f"\n--- {model_name}モデルの詳細結果 ---")
        plot_unified_results(magnetic_datasets, temperature_datasets, trace, model_name)
        
        # 緩和係数依存性の可視化
        plot_gamma_dependencies(trace)
    
    # 結果保存
    print("\n--- 結果保存 ---")
    for model_name, trace in traces.items():
        try:
            # パラメータサマリーの保存
            summary = az.summary(trace, var_names=['a_scale', 'g_factor', 'B4', 'B6', 'gamma_base', 'gamma_B', 'gamma_B2', 'gamma_T', 'gamma_T2', 'gamma_BT'])
            summary.to_csv(str(IMAGE_DIR / f'{model_name}_bayesian_summary.csv'))
            
            # トレースデータの保存
            trace.to_netcdf(str(IMAGE_DIR / f'{model_name}_bayesian_trace.nc'))
            
        except Exception as e:
            print(f"{model_name}の結果保存エラー: {e}")
    
    print(f"\n=== 統合ベイズ推定解析完了 ===")
    print(f"結果は '{IMAGE_DIR.resolve()}' に保存されました。")
