"""
Weighted Least Squares Mixed Fitting (v8.1-WLS)
領域別残差: ポラリトン→透過率スペクトル形状, 共振器→FWHM + ピーク位置

【設計方針】bayesian_v8_mixed_likelihood.py v8.1 と同一の物理モデル・領域分類を使用し、
推定手法のみを「ベイズ推定 (SMC)」→「重み付き最小二乗法 (scipy.optimize.least_squares)」
に変更。

【混合残差構造】
  - ポラリトン領域 (f < 0.361 THz): 透過率スペクトル形状の重み付き残差
  - 高次共振器領域 (f > 0.45 THz): FWHM + ピーク位置の残差
  - 背景領域: 低重みスペクトル残差
【パラメータ不確かさ】ヤコビアン行列から共分散行列を推定、モンテカルロ伝播で信頼帯を計算

【パラメータ (12個)】
  g, a, B4, B6, eps_bg, gamma_1 ... gamma_7

【継承元】bayesian_v8_mixed_likelihood.py v8.1
  - 同一の物理関数群 (Hamiltonian, susceptibility, transmission)
  - 同一の領域分類 (detect_peaks_and_classify, create_weight_array)
  - 同一のデータセット (TARGET_DATA)
  - 同一の FWHM + ピーク位置計算 (compute_cavity_peak_info)
"""

# ========== 設定 ==========
SIGMA_FWHM      = 0.005    # THz (5 GHz) — FWHM 残差の正規化用
SIGMA_PEAK_FREQ = 0.005    # THz (5 GHz) — ピーク位置残差の正規化用
USE_BACKGROUND_RESIDUAL = True
N_MONTE_CARLO = 500        # パラメータ不確かさ伝播用モンテカルロサンプル数

import os
import json
import time
import pathlib
import datetime
import warnings
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'

from scipy.optimize import least_squares, differential_evolution
from scipy.signal import find_peaks, peak_widths

# ============================================================================
# 物理定数
# ============================================================================
kB   = 1.380649e-23
muB  = 9.274010e-24
hbar = 1.054571e-34
c    = 299792458
mu0  = 4.0 * np.pi * 1e-7
eps0 = 8.854187817e-12

THZ_TO_HZ    = 1e12
THZ_TO_RAD_S = 2.0 * np.pi * THZ_TO_HZ
RAD_S_TO_THZ = 1.0 / THZ_TO_RAD_S

N_SPIN   = 1.9386e+28
d_fixed  = 157.8e-6

TARGET_DATA = [
    {'B': 9.0, 'T':  4.0, 'file': 'BayesianInput_Raw_Transmittance_Temperature.xlsx', 'sheet': 'Normalized Data', 'col': '4K'},
    {'B': 9.0, 'T': 10.0, 'file': 'BayesianInput_Raw_Transmittance_Temperature.xlsx', 'sheet': 'Normalized Data', 'col': '10K'},
    {'B': 9.0, 'T': 20.0, 'file': 'BayesianInput_Raw_Transmittance_Temperature.xlsx', 'sheet': 'Normalized Data', 'col': '20K'},
    {'B': 9.0, 'T': 30.0, 'file': 'BayesianInput_Raw_Transmittance_Temperature.xlsx', 'sheet': 'Normalized Data', 'col': '30K'},
    {'B': 4.2, 'T':  1.5, 'file': 'BayesianInput_Raw_Transmittance_Field.xlsx',       'sheet': 'Normalized Data', 'col': '4.2T'},
    {'B': 5.0, 'T':  1.5, 'file': 'BayesianInput_Raw_Transmittance_Field.xlsx',       'sheet': 'Normalized Data', 'col': '5T'},
    {'B': 6.0, 'T':  1.5, 'file': 'BayesianInput_Raw_Transmittance_Field.xlsx',       'sheet': 'Normalized Data', 'col': '6T'},
    {'B': 7.0, 'T':  1.5, 'file': 'BayesianInput_Raw_Transmittance_Field.xlsx',       'sheet': 'Normalized Data', 'col': '7T'},
    {'B': 8.0, 'T':  1.5, 'file': 'BayesianInput_Raw_Transmittance_Field.xlsx',       'sheet': 'Normalized Data', 'col': '8T'},
    {'B': 9.0, 'T':  1.5, 'file': 'BayesianInput_Raw_Transmittance_Field.xlsx',       'sheet': 'Normalized Data', 'col': '9T'},
]

S_VALUE       = 3.5
N_TRANSITIONS = 7

POLARITON_UPPER = 0.361505   # THz — ポラリトン領域上限
CAVITY_LOWER    = 0.45       # THz — 共振器領域下限

# パラメータ名とインデックス
PARAM_NAMES = ['g', 'a', 'B4', 'B6', 'eps_bg',
               'gamma_1', 'gamma_2', 'gamma_3', 'gamma_4',
               'gamma_5', 'gamma_6', 'gamma_7']

# パラメータ境界 (物理的制約)
PARAM_BOUNDS_LOWER = np.array([
    1.5,    # g
    0.1,    # a
    -0.075, # B4 (cm^-1)
    -0.025, # B6 (cm^-1)
    13.0,   # eps_bg
    0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,  # gamma_1-7 (THz)
])
PARAM_BOUNDS_UPPER = np.array([
    2.8,    # g
    12.0,   # a
    0.075,  # B4
    0.025,  # B6
    16.0,   # eps_bg
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # gamma_1-7 (THz)
])

# 初期値 (v6 最適化結果を使用、フォールバック)
PARAM_INITIAL_DEFAULT = np.array([
    2.0,    # g
    3.0,    # a
    0.0,    # B4
    0.0,    # B6
    14.5,   # eps_bg
    0.074, 0.074, 0.074, 0.074, 0.074, 0.074, 0.074,  # gamma_1-7
])


# ============================================================================
# 物理関数群 (bayesian_v8_mixed_likelihood.py と同一)
# ============================================================================
def get_hamiltonian(B_ext_z, g_factor, B4, B6, s=S_VALUE):
    n_states = int(2 * s + 1)
    m_values = np.arange(s, -s - 1, -1)
    Sz = np.diag(m_values)
    if n_states == 8:
        O40 = np.diag([7, -13, -3, 9, 9, -3, -13, 7]) / 60
        X_O44 = np.zeros((8, 8))
        X_O44[3, 7] = X_O44[4, 0] = np.sqrt(35) / 12
        X_O44[2, 6] = X_O44[5, 1] = 5 * np.sqrt(3) / 12
        O44 = X_O44 + X_O44.T
        O60 = np.diag([1, -5, 9, -5, -5, 9, -5, 1]) / 1260
        X_O64 = np.zeros((8, 8))
        X_O64[3, 7] = X_O64[4, 0] = 3 * np.sqrt(35) / 60
        X_O64[2, 6] = X_O64[5, 1] = -7 * np.sqrt(3) / 60
        O64 = X_O64 + X_O64.T
    else:
        raise ValueError(f"s={s} は未実装")
    H_cf  = B4 * (O40 + 5 * O44) + B6 * (O60 - 21 * O64)
    H_zee = g_factor * muB * B_ext_z * Sz / kB
    return H_cf + H_zee


def construct_spin_operators():
    s_val    = 3.5
    n_states = int(2 * s_val + 1)
    m_values = np.arange(s_val, -s_val - 1, -1)
    Sz = np.diag(m_values)
    Sx = np.zeros((n_states, n_states), dtype=float)
    Sy = np.zeros((n_states, n_states), dtype=float)
    for i in range(n_states - 1):
        m_lower = m_values[i + 1]
        coeff = np.sqrt((s_val - m_lower) * (s_val + m_lower + 1))
        Sx[i, i + 1] += coeff / 2.0
        Sy[i, i + 1] += -coeff / (2.0j)
    for i in range(1, n_states):
        m_upper = m_values[i - 1]
        coeff = np.sqrt((s_val + m_upper) * (s_val - m_upper + 1))
        Sx[i, i - 1] += coeff / 2.0
        Sy[i, i - 1] += coeff / (2.0j)
    Sy_real = np.imag(Sy)
    return Sx, Sy_real, Sz


def calculate_susceptibility(freq_thz, H, T, gamma_thz):
    gamma_uniform = 0.1
    gamma_array_7 = np.full(7, 0.1)
    if np.isscalar(gamma_thz):
        gamma_mode    = 'uniform'
        gamma_uniform = float(np.real(gamma_thz))
    elif hasattr(gamma_thz, '__len__'):
        gamma_array = np.atleast_1d(gamma_thz).real.astype(float)
        if len(gamma_array) == 7:
            gamma_mode    = '7gamma'
            gamma_array_7 = gamma_array
        else:
            gamma_mode    = 'uniform'
            gamma_uniform = float(gamma_array[0])
    else:
        gamma_mode    = 'uniform'
        gamma_uniform = float(np.real(gamma_thz))

    eigenvalues_K, eigenvectors = np.linalg.eigh(H)
    E_min       = np.min(eigenvalues_K)
    E_shifted_K = eigenvalues_K - E_min
    boltzmann_exp = np.clip(E_shifted_K / T, -700, 700)
    Z           = np.sum(np.exp(-boltzmann_exp))
    populations = np.exp(-boltzmann_exp) / Z
    E_shifted_J = E_shifted_K * kB

    Sx_zeeman, Sy_zeeman, _Sz_zeeman = construct_spin_operators()
    Sx_eb = eigenvectors.T.conj() @ Sx_zeeman @ eigenvectors
    Sy_eb = eigenvectors.T.conj() @ Sy_zeeman @ eigenvectors

    transition_xx   = np.abs(Sx_eb) ** 2
    transition_yy   = np.abs(Sy_eb) ** 2
    delta_E_matrix  = E_shifted_J[None, :] - E_shifted_J[:, None]
    omega_0_rad     = delta_E_matrix / hbar
    freq_0_matrix   = omega_0_rad * RAD_S_TO_THZ
    transition_perp = (transition_xx + transition_yy) / 2.0
    pop_diff_matrix = populations[:, None] - populations[None, :]
    strength_matrix = pop_diff_matrix * transition_perp
    non_diag_mask   = ~np.eye(8, dtype=bool)
    population_threshold = 1e-3
    occupied_mask   = populations[:, None] > population_threshold
    finite_mask     = (
        np.isfinite(freq_0_matrix) &
        np.isfinite(strength_matrix) &
        (np.abs(strength_matrix) > 1e-20) &
        occupied_mask &
        non_diag_mask
    )
    if not np.any(finite_mask):
        return np.zeros_like(freq_thz, dtype=complex)

    freq_0_valid   = freq_0_matrix[finite_mask]
    strength_valid = strength_matrix[finite_mask]
    n_indices, n_prime_indices = np.where(finite_mask)
    energy_order   = np.argsort(E_shifted_J)

    if gamma_mode == 'uniform':
        gamma_per_transition = np.full(len(freq_0_valid), gamma_uniform)
    elif gamma_mode == '7gamma':
        gamma_per_transition = np.zeros(len(freq_0_valid))
        for trans_idx in range(len(freq_0_valid)):
            n       = n_indices[trans_idx]
            n_prime = n_prime_indices[trans_idx]
            E_n       = E_shifted_J[n]
            E_n_prime = E_shifted_J[n_prime]
            lower_state = n if E_n <= E_n_prime else n_prime
            lower_state_energy_idx = np.where(energy_order == lower_state)[0][0]
            gamma_idx = min(lower_state_energy_idx, 6)
            gamma_per_transition[trans_idx] = gamma_array_7[gamma_idx]
    else:
        gamma_per_transition = np.full(len(freq_0_valid), 0.1)

    freq_diff  = freq_0_valid[None, :] - freq_thz[:, None]
    denominator = freq_diff - 1j * gamma_per_transition[None, :]
    safe_mask   = np.abs(denominator) > 1e-10
    denominator = np.where(safe_mask, denominator, 1e-10 + 1j * 1e-10)
    chi_array   = np.sum(strength_valid[None, :] / denominator, axis=1)
    return chi_array


def calculate_transmission(freq_thz, mu_r, d, eps_bg):
    eps_bg    = max(eps_bg, 0.1)
    d         = max(d, 1e-6)
    omega     = freq_thz * THZ_TO_RAD_S
    mu_r_safe = np.where(np.isfinite(mu_r), mu_r, 1.0 + 0j)
    eps_mu    = eps_bg * mu_r_safe
    n_ref     = np.sqrt(eps_mu.astype(complex))
    impe      = np.where(np.abs(n_ref) > 1e-15, 1.0 / n_ref, 1.0)
    delta     = n_ref * omega * d / c
    numerator = 4.0 * impe
    exp_pos   = np.exp(-1j * delta)
    exp_neg   = np.exp(1j * delta)
    denom_fp  = (1 + impe) ** 2 * exp_pos - (1 - impe) ** 2 * exp_neg
    safe_mask = np.abs(denom_fp) > 1e-15
    t         = np.zeros_like(denom_fp, dtype=complex)
    t[safe_mask] = numerator[safe_mask] / denom_fp[safe_mask]
    transmission = np.abs(t) ** 2
    transmission = np.where(np.isfinite(transmission), transmission, 0.0)
    transmission = np.clip(transmission, 0, 2)
    t_min, t_max = np.min(transmission), np.max(transmission)
    if t_max > t_min and np.isfinite(t_max) and np.isfinite(t_min):
        return np.clip((transmission - t_min) / (t_max - t_min), 0.0, 1.0)
    else:
        return np.full_like(transmission, 0.5)


def calculate_transmission_for_params(freq, B, T, g, a, B4, B6, eps, gamma_array, model_form='H'):
    H_ham   = get_hamiltonian(B, g, B4, B6)
    chi_raw = calculate_susceptibility(freq, H_ham, T, gamma_array)
    G0      = a * mu0 * N_SPIN * (g * muB) ** 2 / (2 * hbar) / THZ_TO_RAD_S
    chi     = G0 * chi_raw
    if model_form == 'H':
        mu_r = 1.0 + chi
    else:
        mu_r = 1.0 / (1.0 - chi)
    return calculate_transmission(freq, mu_r, d_fixed, eps)


# ============================================================================
# FWHM・ピーク位置計算ユーティリティ (bayesian_v8 v8.1 と同一)
# ============================================================================
def compute_cavity_peak_info(freq, trans, cavity_lower=CAVITY_LOWER):
    """共振器領域の最も顕著なピークの FWHM とピーク周波数を返す。"""
    mask = freq >= cavity_lower
    if not np.any(mask):
        return None, None

    freq_cav  = freq[mask]
    trans_cav = trans[mask]

    peaks, props = find_peaks(trans_cav, prominence=0.03, width=2)
    if len(peaks) == 0:
        return None, None

    best_idx = np.argmax(props['prominences'])
    peak_idx = peaks[best_idx]

    widths_samples, _, _, _ = peak_widths(trans_cav, [peak_idx], rel_height=0.5)
    df = freq[1] - freq[0] if len(freq) > 1 else 1.0

    return float(widths_samples[0] * df), float(freq_cav[peak_idx])


def compute_fwhm_from_spectrum(freq, trans, cavity_lower=CAVITY_LOWER):
    """共振器ピークの FWHM リストとピーク周波数リストを返す。"""
    df   = freq[1] - freq[0] if len(freq) > 1 else 1.0
    mask = freq >= cavity_lower
    if not np.any(mask):
        return [], []
    freq_cav  = freq[mask]
    trans_cav = trans[mask]
    peaks, _props = find_peaks(trans_cav, prominence=0.03, width=2)
    if len(peaks) == 0:
        return [], []
    widths_samples, _, _, _ = peak_widths(trans_cav, peaks, rel_height=0.5)
    fwhm_list      = [float(w * df) for w in widths_samples]
    peak_freq_list = [float(freq_cav[p]) for p in peaks]
    return fwhm_list, peak_freq_list


# ============================================================================
# ピーク検出・重み配列
# ============================================================================
def detect_peaks_and_classify(freq, trans, polariton_upper=POLARITON_UPPER, cavity_lower=CAVITY_LOWER):
    peaks, properties = find_peaks(trans, prominence=0.05, width=3)
    if len(peaks) == 0:
        return [], []
    peak_freqs  = freq[peaks]
    peak_widths_arr = properties['widths'] * (freq[1] - freq[0])
    sort_idx    = np.argsort(peak_freqs)
    peak_freqs  = peak_freqs[sort_idx]
    peak_widths_arr = peak_widths_arr[sort_idx]
    polariton_regions = []
    cavity_regions    = []
    for pf, pw in zip(peak_freqs, peak_widths_arr):
        f_start = max(freq[0], pf - 1.5 * pw)
        f_end   = min(freq[-1], pf + 1.5 * pw)
        if pf <= polariton_upper:
            f_end_clipped = min(f_end, polariton_upper)
            if f_end_clipped > f_start:
                polariton_regions.append((f_start, f_end_clipped))
        elif pf >= cavity_lower:
            f_start_clipped = max(f_start, cavity_lower)
            if f_end > f_start_clipped:
                cavity_regions.append((f_start_clipped, f_end))
    return polariton_regions, cavity_regions


def create_weight_array(freq, _trans, polariton_regions, cavity_regions):
    weight_array = np.full_like(freq, 0.01)
    for f_start, f_end in polariton_regions:
        mask = (freq >= f_start) & (freq <= f_end)
        weight_array[mask] = 2.0
    for f_start, f_end in cavity_regions:
        mask = (freq >= f_start) & (freq <= f_end)
        weight_array[mask] = 1.0
    return weight_array


# ============================================================================
# データ読み込み
# ============================================================================
def load_all_datasets(target_data_list):
    print("\n--- データ読み込み ---")
    datasets  = []
    base_dir  = pathlib.Path(__file__).parent / 'bayesian_inputs'

    for config in target_data_list:
        excel_path = base_dir / config['file']
        if not excel_path.exists():
            print(f"❌ {excel_path} が見つかりません")
            continue
        try:
            df = pd.read_excel(excel_path, sheet_name=config['sheet'])
            if 'Frequency (THz)' not in df.columns or config['col'] not in df.columns:
                print(f"❌ {config['col']}: 必要な列が見つかりません")
                continue
            df_clean = df[['Frequency (THz)', config['col']]].dropna()
            freq  = df_clean['Frequency (THz)'].values.astype(np.float64)
            trans = df_clean[config['col']].values.astype(np.float64)

            polariton_regions, cavity_regions = detect_peaks_and_classify(freq, trans)
            weight_array = create_weight_array(freq, trans, polariton_regions, cavity_regions)
            fwhm_obs, peak_freq_obs = compute_cavity_peak_info(freq, trans)

            label = f"{config['B']:.1f}T" if config['T'] == 1.5 else f"{config['T']:.0f}K"

            dataset = {
                'freq':             freq,
                'trans':            trans,
                'weight':           weight_array,
                'B':                config['B'],
                'T':                config['T'],
                'label':            label,
                'polariton_regions': polariton_regions,
                'cavity_regions':   cavity_regions,
                'sigma':            np.full_like(freq, 0.01),
                'fwhm_obs':         fwhm_obs,
                'peak_freq_obs':    peak_freq_obs,
            }
            datasets.append(dataset)

            fwhm_str = f"{fwhm_obs*1000:.1f} GHz" if fwhm_obs is not None else "N/A"
            print(f"✓ {label} (B={config['B']}T, T={config['T']}K): {len(freq)} points, "
                  f"cav FWHM={fwhm_str}")
        except Exception as e:
            print(f"❌ {config['col']} 読み込みエラー: {e}")

    print(f"\n✅ 合計 {len(datasets)} データセット読み込み完了")
    n_with_fwhm = sum(1 for d in datasets if d['fwhm_obs'] is not None)
    print(f"   FWHM 観測値あり: {n_with_fwhm} / {len(datasets)} データセット")
    return datasets


# ============================================================================
# v6 結果読み込み（初期値に使用）
# ============================================================================
def load_v6_optimized_params(model_form='H'):
    json_path = pathlib.Path(__file__).parent / f"global_fitting_results_{model_form}_v6" / "shared_gamma_params.json"
    if not json_path.exists():
        print(f"❌ {json_path} が見つかりません")
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        global_params = data['global_params']
        params = {
            'g':   global_params['g'],
            'a':   global_params['a'],
            'B4':  global_params['B4'],
            'B6':  global_params['B6'],
            'eps': global_params['eps'],
            'gamma': np.array(data['shared_gamma']),
        }
        print(f"\n✓ {model_form}-form v6:")
        for k, v in params.items():
            if k != 'gamma':
                print(f"  {k} = {v:.6f}")
        print(f"  gamma = {params['gamma']}")
        return params
    except Exception as e:
        print(f"❌ {model_form}-form 読み込みエラー: {e}")
        return None


def v6_to_param_vector(v6_params):
    """v6 パラメータ辞書を 12 要素のベクトルに変換"""
    gamma = np.atleast_1d(v6_params['gamma'])
    if len(gamma) < 7:
        gamma = np.full(7, gamma[0] if len(gamma) > 0 else 0.074)
    return np.array([
        v6_params['g'], v6_params['a'],
        v6_params['B4'], v6_params['B6'], v6_params['eps'],
        gamma[0], gamma[1], gamma[2], gamma[3], gamma[4], gamma[5], gamma[6]
    ])


# ============================================================================
# 混合残差関数 (WLS のコア)
# ============================================================================
def compute_mixed_residuals(params, datasets, model_form):
    """
    パラメータ params (12要素) から全データセットの混合残差ベクトルを構築する。

    残差構成:
    1. ポラリトン領域: (trans_pred - trans_obs) * sqrt(weight) / sigma
    2. FWHM:          (fwhm_pred - fwhm_obs) / sigma_fwhm
    3. ピーク位置:     (peak_freq_pred - peak_freq_obs) / sigma_peak_freq
    4. 背景領域:       (trans_pred - trans_obs) * sqrt(weight) / sigma
    """
    g       = params[0]
    a       = params[1]
    B4      = params[2]
    B6      = params[3]
    eps_bg  = params[4]
    gamma_array = params[5:12]

    all_residuals = []

    for data in datasets:
        freq      = data['freq']
        trans_obs = data['trans']
        weight    = data['weight']
        B_ext     = data['B']
        T_val     = data['T']

        # 透過率スペクトル計算
        trans_pred = calculate_transmission_for_params(
            freq, B_ext, T_val, g, a, B4, B6, eps_bg, gamma_array, model_form)

        # ポラリトン領域 (weight == 2.0) のスペクトル残差
        pol_mask = weight == 2.0
        if np.any(pol_mask):
            sigma_pol = 0.01 / np.sqrt(weight[pol_mask])
            residual_pol = (trans_pred[pol_mask] - trans_obs[pol_mask]) / sigma_pol
            all_residuals.append(residual_pol)

        # 背景領域 (weight == 0.01) のスペクトル残差
        if USE_BACKGROUND_RESIDUAL:
            bg_mask = weight == 0.01
            if np.any(bg_mask):
                sigma_bg = 0.01 / np.sqrt(weight[bg_mask])
                residual_bg = (trans_pred[bg_mask] - trans_obs[bg_mask]) / sigma_bg
                all_residuals.append(residual_bg)

        # 共振器領域: FWHM + ピーク位置残差
        fwhm_obs      = data['fwhm_obs']
        peak_freq_obs = data['peak_freq_obs']

        if fwhm_obs is not None:
            fwhm_pred, peak_freq_pred = compute_cavity_peak_info(freq, trans_pred)
            if fwhm_pred is None:
                # ペナルティ: 大きな残差を発生
                fwhm_pred = fwhm_obs * 2.0
                peak_freq_pred = peak_freq_obs + 0.1

            residual_fwhm = (fwhm_pred - fwhm_obs) / SIGMA_FWHM
            all_residuals.append(np.array([residual_fwhm]))

            if peak_freq_obs is not None:
                residual_peak = (peak_freq_pred - peak_freq_obs) / SIGMA_PEAK_FREQ
                all_residuals.append(np.array([residual_peak]))

    return np.concatenate(all_residuals)


# ============================================================================
# 最適化実行
# ============================================================================
def run_wls_fitting(datasets, model_form, initial_params):
    """
    重み付き最小二乗法で混合残差を最小化する。

    Returns: OptimizeResult (scipy)
    """
    print(f"\n  初期パラメータ: {dict(zip(PARAM_NAMES, initial_params))}")

    # まず初期残差を確認
    try:
        r0 = compute_mixed_residuals(initial_params, datasets, model_form)
        cost0 = 0.5 * np.sum(r0 ** 2)
        print(f"  初期コスト: {cost0:.4f} (残差 {len(r0)} 点)")
    except Exception as e:
        print(f"  ⚠️ 初期残差計算エラー: {e}")

    result = least_squares(
        compute_mixed_residuals,
        initial_params,
        args=(datasets, model_form),
        bounds=(PARAM_BOUNDS_LOWER, PARAM_BOUNDS_UPPER),
        method='trf',
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=5000,
        verbose=1,
    )

    print(f"\n  最適化結果:")
    print(f"    成功: {result.success}")
    print(f"    コスト: {result.cost:.6f}")
    print(f"    関数評価回数: {result.nfev}")
    print(f"    メッセージ: {result.message}")

    return result


def estimate_parameter_uncertainty(result):
    """
    最小二乗法のヤコビアン行列から共分散行列とパラメータ不確かさを推定する。

    Returns: (cov_matrix, param_std, correlation_matrix)
    """
    J = result.jac
    n_residuals = len(result.fun)
    n_params    = len(result.x)

    # 残差の分散 (reduced chi-square)
    s2 = 2.0 * result.cost / max(n_residuals - n_params, 1)

    # 共分散行列: C = s² * (J^T J)^{-1}
    try:
        JtJ = J.T @ J
        cov_matrix = s2 * np.linalg.inv(JtJ)
        param_std  = np.sqrt(np.abs(np.diag(cov_matrix)))

        # 相関行列
        d = np.sqrt(np.abs(np.diag(cov_matrix)))
        d[d == 0] = 1e-20
        correlation_matrix = cov_matrix / np.outer(d, d)
    except np.linalg.LinAlgError:
        print("  ⚠️ 共分散行列の計算に失敗（特異行列）")
        cov_matrix = np.full((n_params, n_params), np.nan)
        param_std  = np.full(n_params, np.nan)
        correlation_matrix = np.full((n_params, n_params), np.nan)

    return cov_matrix, param_std, correlation_matrix


# ============================================================================
# プロット関数
# ============================================================================
def plot_fit_spectra(params, datasets, model_form, save_dir=None,
                     cov_matrix=None, n_mc=N_MONTE_CARLO):
    """
    最適フィットスペクトルを観測データと共にプロット。
    共分散行列が与えられた場合、モンテカルロ伝播で 95% 信頼帯を表示。
    """
    print(f"\n{'='*80}\nフィットスペクトルプロット ({model_form}-form)\n{'='*80}")

    # モンテカルロサンプルの生成
    mc_params = None
    if cov_matrix is not None and np.all(np.isfinite(cov_matrix)):
        try:
            mc_params = np.random.multivariate_normal(params, cov_matrix, size=n_mc)
            # 境界でクリップ
            mc_params = np.clip(mc_params, PARAM_BOUNDS_LOWER, PARAM_BOUNDS_UPPER)
        except (np.linalg.LinAlgError, ValueError):
            print("  ⚠️ モンテカルロサンプル生成失敗、信頼帯なし")
            mc_params = None

    n_datasets = len(datasets)
    ncols = 2
    nrows = (n_datasets + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    fig.suptitle(f'WLS Fit Spectra ({model_form}-form) — v8.1-WLS Mixed Residual',
                 fontsize=12, y=0.995)
    axes = axes.flatten()

    g, a, B4, B6, eps_bg = params[0], params[1], params[2], params[3], params[4]
    gamma_array = params[5:12]

    for idx, data in enumerate(datasets):
        ax        = axes[idx]
        freq      = data['freq']
        trans_obs = data['trans']
        B, T      = data['B'], data['T']
        label     = data['label']
        fwhm_obs  = data['fwhm_obs']
        peak_freq_obs = data.get('peak_freq_obs')

        # 最適フィット
        trans_fit = calculate_transmission_for_params(
            freq, B, T, g, a, B4, B6, eps_bg, gamma_array, model_form)

        # 信頼帯 (モンテカルロ伝播)
        has_ci = False
        if mc_params is not None:
            trans_mc = np.zeros((n_mc, len(freq)))
            for i in range(n_mc):
                p = mc_params[i]
                trans_mc[i] = calculate_transmission_for_params(
                    freq, B, T, p[0], p[1], p[2], p[3], p[4], p[5:12], model_form)
            ci_lo = np.percentile(trans_mc, 2.5, axis=0)
            ci_hi = np.percentile(trans_mc, 97.5, axis=0)
            has_ci = True

        fwhm_pred, peak_freq_pred = compute_cavity_peak_info(freq, trans_fit)

        # 領域別 RMSE
        pol_mask = freq < POLARITON_UPPER
        cav_mask = freq >= CAVITY_LOWER
        rmse_all = np.sqrt(np.mean((trans_obs - trans_fit) ** 2))
        rmse_pol = np.sqrt(np.mean((trans_obs[pol_mask] - trans_fit[pol_mask]) ** 2)) if np.any(pol_mask) else np.nan
        rmse_cav = np.sqrt(np.mean((trans_obs[cav_mask] - trans_fit[cav_mask]) ** 2)) if np.any(cav_mask) else np.nan

        # ピーク位置誤差
        peak_err_ghz = abs(peak_freq_pred - peak_freq_obs) * 1000 if (peak_freq_pred and peak_freq_obs) else np.nan

        # 領域ハイライト
        for f_s, f_e in data['polariton_regions']:
            ax.axvspan(f_s, f_e, alpha=0.12, color='orange',
                       label='Polariton' if f_s == data['polariton_regions'][0][0] else None)
        for f_s, f_e in data['cavity_regions']:
            ax.axvspan(f_s, f_e, alpha=0.12, color='green',
                       label='Cavity' if f_s == data['cavity_regions'][0][0] else None)

        ax.plot(freq, trans_obs, 'ko', markersize=2.5, alpha=0.6, label='Obs')
        ax.plot(freq, trans_fit, 'r-', lw=2, label='WLS Fit')
        if has_ci:
            ax.fill_between(freq, ci_lo, ci_hi, color='red', alpha=0.2, label='95% CI')

        fwhm_str = (f"FWHM obs={fwhm_obs*1000:.1f} pred={fwhm_pred*1000:.1f} GHz"
                    if fwhm_obs and fwhm_pred else "")
        region_str = f"Pol={rmse_pol:.4f} Cav={rmse_cav:.4f}"
        peak_str = f" Δf={peak_err_ghz:.1f}GHz" if not np.isnan(peak_err_ghz) else ""
        ax.set_title(f"{label}  RMSE={rmse_all:.4f} ({region_str}){peak_str}\n{fwhm_str}",
                     fontsize=8, fontweight='bold')
        ax.set_xlabel('Frequency (THz)', fontsize=9)
        ax.set_ylabel('Transmittance', fontsize=9)
        ax.legend(fontsize=6, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim([freq.min(), freq.max()])
        ax.set_ylim([0, 1.05])

    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')
    plt.tight_layout()
    if save_dir:
        path = save_dir / f'fit_spectra_{model_form}.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ {path.name} saved")
    plt.close()


def plot_parameter_summary(params, param_std, model_form, save_dir=None):
    """パラメータ値と不確かさをプロット"""
    print(f"\nパラメータサマリープロット作成 ({model_form}-form)...")

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Parameter Estimates with 1σ Uncertainty ({model_form}-form) — v8.1-WLS',
                 fontsize=14)
    axes = axes.flatten()

    param_display = [
        (0, 'g-factor', ''),
        (1, 'a (coupling)', ''),
        (2, 'B₄', 'mK'),
        (3, 'B₆', 'mK'),
        (4, 'ε_bg', ''),
    ]
    for i, (pidx, label, unit) in enumerate(param_display):
        ax = axes[i]
        val = params[pidx]
        err = param_std[pidx] if np.isfinite(param_std[pidx]) else 0
        if unit == 'mK':
            val *= 1000
            err *= 1000
            xlabel = f'{label} ({unit})'
        else:
            xlabel = label

        ax.barh([label], [val], xerr=[err], color='steelblue', edgecolor='black',
                capsize=5, height=0.5)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(f'{label} = {val:.4g} ± {err:.2g}', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')

    for i in range(7):
        ax = axes[5 + i]
        val = params[5 + i]
        err = param_std[5 + i] if np.isfinite(param_std[5 + i]) else 0
        ax.barh([f'γ_{i+1}'], [val], xerr=[err], color='steelblue', edgecolor='black',
                capsize=5, height=0.5)
        ax.set_xlabel(f'γ_{i+1} (THz)', fontsize=9)
        ax.set_title(f'γ_{i+1} = {val:.4f} ± {err:.4f} THz ({val*1000:.1f} ± {err*1000:.1f} GHz)',
                     fontsize=9, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    if save_dir:
        path = save_dir / f'parameter_summary_{model_form}.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ {path.name} saved")
    plt.close()


def plot_correlation_matrix(correlation_matrix, model_form, save_dir=None):
    """パラメータ相関行列のヒートマップ"""
    print(f"\n相関行列プロット作成 ({model_form}-form)...")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(len(PARAM_NAMES)))
    ax.set_xticklabels(PARAM_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(PARAM_NAMES)))
    ax.set_yticklabels(PARAM_NAMES, fontsize=8)
    for i in range(len(PARAM_NAMES)):
        for j in range(len(PARAM_NAMES)):
            val = correlation_matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color='black' if abs(val) < 0.5 else 'white')
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title(f'Parameter Correlation Matrix ({model_form}-form) — v8.1-WLS', fontsize=12)
    plt.tight_layout()
    if save_dir:
        path = save_dir / f'correlation_matrix_{model_form}.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ {path.name} saved")
    plt.close()


def plot_energy_levels(params, datasets, model_form, save_dir=None,
                       cov_matrix=None, n_mc=N_MONTE_CARLO):
    """エネルギー準位図 (最適値 + 信頼帯)"""
    print(f"\nエネルギー準位プロット作成 ({model_form}-form)...")
    g, B4, B6 = params[0], params[2], params[3]

    B_fields = np.linspace(0, 10, 60)
    n_states = int(2 * S_VALUE + 1)

    # 最適値でのエネルギー準位
    median_evals = np.zeros((len(B_fields), n_states))
    for j, B_val in enumerate(B_fields):
        H_ham = get_hamiltonian(B_val, g, B4, B6)
        evals = np.sort(np.linalg.eigh(H_ham)[0])
        median_evals[j] = evals - evals[0]

    # モンテカルロ伝播で信頼帯
    has_ci = False
    if cov_matrix is not None and np.all(np.isfinite(cov_matrix)):
        try:
            mc_params = np.random.multivariate_normal(params, cov_matrix, size=n_mc)
            mc_params = np.clip(mc_params, PARAM_BOUNDS_LOWER, PARAM_BOUNDS_UPPER)
            all_evals = np.zeros((n_mc, len(B_fields), n_states))
            for i in range(n_mc):
                g_mc, B4_mc, B6_mc = mc_params[i, 0], mc_params[i, 2], mc_params[i, 3]
                for j, B_val in enumerate(B_fields):
                    H_ham = get_hamiltonian(B_val, g_mc, B4_mc, B6_mc)
                    evals = np.sort(np.linalg.eigh(H_ham)[0])
                    all_evals[i, j] = evals - evals[0]
            ci_lo = np.percentile(all_evals, 2.5, axis=0)
            ci_hi = np.percentile(all_evals, 97.5, axis=0)
            has_ci = True
        except (np.linalg.LinAlgError, ValueError):
            pass

    _fig, ax = plt.subplots(figsize=(10, 7))
    colors = [plt.colormaps['tab10'](i / n_states) for i in range(n_states)]
    for k in range(n_states):
        ax.scatter(B_fields, median_evals[:, k], c=[colors[k]], s=12, zorder=3,
                   label=f'|{k}⟩', edgecolors='none')
        if has_ci:
            ax.fill_between(B_fields, ci_lo[:, k], ci_hi[:, k],
                            color=colors[k], alpha=0.15)

    dataset_Bs = sorted(set(d['B'] for d in datasets))
    for Bval in dataset_Bs:
        ax.axvline(Bval, color='gray', linestyle=':', alpha=0.4, lw=0.8)

    ci_label = ' — 95% CI' if has_ci else ''
    ax.set_xlabel('Magnetic Field B (T)', fontsize=12)
    ax.set_ylabel('Energy E − E₀ (K)', fontsize=12)
    ax.set_title(f'Energy Level Diagram ({model_form}-form){ci_label}\n'
                 f'S={S_VALUE}, WLS optimized', fontsize=12)
    ax.legend(fontsize=7, ncol=4, loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_dir:
        path = save_dir / f'energy_levels_{model_form}.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ {path.name} saved")
    plt.close()


def plot_susceptibility(params, datasets, model_form, save_dir=None):
    """磁気感受率プロット (Re/Im)"""
    print(f"\n磁気感受率プロット作成 ({model_form}-form)...")
    g, a, B4, B6 = params[0], params[1], params[2], params[3]
    gamma_array = params[5:12]

    n_datasets = len(datasets)
    ncols = 2
    nrows = (n_datasets + 1) // 2

    fig_re, axes_re = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    fig_re.suptitle(f'Magnetic Susceptibility Re(χ₊) ({model_form}-form) — WLS',
                    fontsize=12, y=0.995)
    axes_re = axes_re.flatten()

    fig_im, axes_im = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    fig_im.suptitle(f'Magnetic Susceptibility Im(χ₊) ({model_form}-form) — WLS',
                    fontsize=12, y=0.995)
    axes_im = axes_im.flatten()

    for idx, data in enumerate(datasets):
        freq  = data['freq']
        B_ext = data['B']
        T_val = data['T']
        label = data['label']

        H_ham   = get_hamiltonian(B_ext, g, B4, B6)
        chi_raw = calculate_susceptibility(freq, H_ham, T_val, gamma_array)
        G0      = a * mu0 * N_SPIN * (g * muB) ** 2 / (2 * hbar) / THZ_TO_RAD_S
        chi     = G0 * chi_raw

        ax_re = axes_re[idx]
        ax_re.plot(freq, np.real(chi), 'b-', lw=1.5, label='WLS Fit')
        ax_re.set_title(f'{label} (B={B_ext}T, T={T_val}K)', fontsize=9, fontweight='bold')
        ax_re.set_xlabel('Frequency (THz)', fontsize=8)
        ax_re.set_ylabel('Re(χ₊)', fontsize=8)
        ax_re.legend(fontsize=6, loc='best')
        ax_re.grid(alpha=0.3)
        ax_re.axhline(0, color='gray', lw=0.5)

        ax_im = axes_im[idx]
        ax_im.plot(freq, np.imag(chi), 'r-', lw=1.5, label='WLS Fit')
        ax_im.set_title(f'{label} (B={B_ext}T, T={T_val}K)', fontsize=9, fontweight='bold')
        ax_im.set_xlabel('Frequency (THz)', fontsize=8)
        ax_im.set_ylabel('Im(χ₊)', fontsize=8)
        ax_im.legend(fontsize=6, loc='best')
        ax_im.grid(alpha=0.3)
        ax_im.axhline(0, color='gray', lw=0.5)

    for idx in range(n_datasets, len(axes_re)):
        axes_re[idx].axis('off')
        axes_im[idx].axis('off')

    fig_re.tight_layout()
    fig_im.tight_layout()
    if save_dir:
        path_re = save_dir / f'susceptibility_real_{model_form}.png'
        fig_re.savefig(path_re, dpi=300, bbox_inches='tight')
        print(f"  ✓ {path_re.name} saved")
        path_im = save_dir / f'susceptibility_imag_{model_form}.png'
        fig_im.savefig(path_im, dpi=300, bbox_inches='tight')
        print(f"  ✓ {path_im.name} saved")
    plt.close(fig_re)
    plt.close(fig_im)


def plot_residual_analysis(params, datasets, model_form, save_dir=None):
    """残差分析プロット (各データセットの残差分布 + Q-Qプロット)"""
    print(f"\n残差分析プロット作成 ({model_form}-form)...")
    g, a, B4, B6, eps_bg = params[0], params[1], params[2], params[3], params[4]
    gamma_array = params[5:12]

    n_datasets = len(datasets)
    ncols = 2
    nrows = (n_datasets + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    fig.suptitle(f'Residual Analysis ({model_form}-form) — v8.1-WLS', fontsize=12, y=0.995)
    axes = axes.flatten()

    for idx, data in enumerate(datasets):
        ax   = axes[idx]
        freq = data['freq']
        trans_obs = data['trans']
        label = data['label']

        trans_fit = calculate_transmission_for_params(
            freq, data['B'], data['T'], g, a, B4, B6, eps_bg, gamma_array, model_form)
        residuals = trans_obs - trans_fit

        # 領域別色分け
        pol_mask = freq < POLARITON_UPPER
        cav_mask = freq >= CAVITY_LOWER
        mid_mask = ~pol_mask & ~cav_mask

        ax.plot(freq[pol_mask], residuals[pol_mask], 'o', color='orange',
                markersize=2, alpha=0.6, label='Polariton')
        ax.plot(freq[mid_mask], residuals[mid_mask], 'o', color='gray',
                markersize=2, alpha=0.4, label='Gap')
        ax.plot(freq[cav_mask], residuals[cav_mask], 'o', color='green',
                markersize=2, alpha=0.6, label='Cavity')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_title(f'{label} — std={np.std(residuals):.4f}', fontsize=9, fontweight='bold')
        ax.set_xlabel('Frequency (THz)', fontsize=8)
        ax.set_ylabel('Residual', fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)

    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')
    plt.tight_layout()
    if save_dir:
        path = save_dir / f'residual_analysis_{model_form}.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ {path.name} saved")
    plt.close()


# ============================================================================
# モデル比較 (AIC / BIC)
# ============================================================================
def compute_model_comparison(result_H, result_B, n_datasets):
    """AIC / BIC によるモデル比較"""
    print(f"\n{'='*80}\nモデル比較 (AIC / BIC)\n{'='*80}")

    n_params = len(result_H.x)
    results = {}

    for form, result in [('H', result_H), ('B', result_B)]:
        n_residuals = len(result.fun)
        rss = 2.0 * result.cost  # sum of squared residuals
        # AIC = n * ln(RSS/n) + 2k
        aic = n_residuals * np.log(rss / n_residuals) + 2 * n_params
        # BIC = n * ln(RSS/n) + k * ln(n)
        bic = n_residuals * np.log(rss / n_residuals) + n_params * np.log(n_residuals)
        chi2_reduced = rss / max(n_residuals - n_params, 1)

        results[form] = {
            'n_residuals': n_residuals,
            'rss': rss,
            'aic': aic,
            'bic': bic,
            'chi2_reduced': chi2_reduced,
        }
        print(f"  {form}-form: RSS={rss:.4f}, χ²_ν={chi2_reduced:.4f}, AIC={aic:.2f}, BIC={bic:.2f}")

    delta_aic = results['H']['aic'] - results['B']['aic']
    delta_bic = results['H']['bic'] - results['B']['bic']
    winner_aic = 'H-form' if delta_aic < 0 else ('B-form' if delta_aic > 0 else '引き分け')
    winner_bic = 'H-form' if delta_bic < 0 else ('B-form' if delta_bic > 0 else '引き分け')

    print(f"\n  ΔAIC(H-B) = {delta_aic:.2f} → {winner_aic}")
    print(f"  ΔBIC(H-B) = {delta_bic:.2f} → {winner_bic}")

    results['delta_aic'] = delta_aic
    results['delta_bic'] = delta_bic
    results['winner_aic'] = winner_aic
    results['winner_bic'] = winner_bic

    return results


# ============================================================================
# メイン処理
# ============================================================================
def main():
    start_time = time.time()

    print(f"\n{'='*80}")
    print("WLS Analysis v8.1 — Mixed Residual (Polariton: spectrum / Cavity: FWHM + peak position)")
    print(f"bayesian_v8_mixed_likelihood.py v8.1 設計方針準拠、推定手法: 重み付き最小二乗法")
    print(f"{'='*80}")

    # v6 参照値読み込み
    v6_params_H = load_v6_optimized_params('H')
    v6_params_B = load_v6_optimized_params('B')
    if v6_params_H is None or v6_params_B is None:
        print("❌ v6 最適化結果の読み込みに失敗しました")
        return

    # 結果ディレクトリ
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = pathlib.Path(__file__).parent / f"wls_v8_results_{timestamp}"
    results_dir.mkdir(exist_ok=True)
    print(f"\n📁 結果保存先: {results_dir}")

    # データ読み込み
    datasets = load_all_datasets(TARGET_DATA)
    if not datasets:
        print("❌ データがありません")
        return

    # ==================== H-form ====================
    print(f"\n{'='*80}\nH 形式 WLS フィッティング\n{'='*80}")
    initial_H = v6_to_param_vector(v6_params_H)
    initial_H = np.clip(initial_H, PARAM_BOUNDS_LOWER, PARAM_BOUNDS_UPPER)
    result_H = run_wls_fitting(datasets, 'H', initial_H)
    cov_H, std_H, corr_H = estimate_parameter_uncertainty(result_H)

    print("\n  H-form 最適パラメータ:")
    for name, val, err in zip(PARAM_NAMES, result_H.x, std_H):
        unit = ' mK' if name in ('B4', 'B6') else (' THz' if 'gamma' in name else '')
        scale = 1000 if name in ('B4', 'B6') else 1
        print(f"    {name:10s} = {val*scale:12.6f} ± {err*scale:.6f}{unit}")

    # ==================== B-form ====================
    print(f"\n{'='*80}\nB 形式 WLS フィッティング\n{'='*80}")
    initial_B = v6_to_param_vector(v6_params_B)
    initial_B = np.clip(initial_B, PARAM_BOUNDS_LOWER, PARAM_BOUNDS_UPPER)
    result_B = run_wls_fitting(datasets, 'B', initial_B)
    cov_B, std_B, corr_B = estimate_parameter_uncertainty(result_B)

    print("\n  B-form 最適パラメータ:")
    for name, val, err in zip(PARAM_NAMES, result_B.x, std_B):
        unit = ' mK' if name in ('B4', 'B6') else (' THz' if 'gamma' in name else '')
        scale = 1000 if name in ('B4', 'B6') else 1
        print(f"    {name:10s} = {val*scale:12.6f} ± {err*scale:.6f}{unit}")

    # ==================== モデル比較 ====================
    comparison = compute_model_comparison(result_H, result_B, len(datasets))

    # ==================== 可視化 ====================
    print(f"\n{'='*80}\n📊 プロット生成\n{'='*80}")
    plot_fit_spectra(result_H.x, datasets, 'H', results_dir, cov_H)
    plot_parameter_summary(result_H.x, std_H, 'H', results_dir)
    plot_correlation_matrix(corr_H, 'H', results_dir)
    plot_energy_levels(result_H.x, datasets, 'H', results_dir, cov_H)
    plot_susceptibility(result_H.x, datasets, 'H', results_dir)
    plot_residual_analysis(result_H.x, datasets, 'H', results_dir)

    plot_fit_spectra(result_B.x, datasets, 'B', results_dir, cov_B)
    plot_parameter_summary(result_B.x, std_B, 'B', results_dir)
    plot_correlation_matrix(corr_B, 'B', results_dir)
    plot_energy_levels(result_B.x, datasets, 'B', results_dir, cov_B)
    plot_susceptibility(result_B.x, datasets, 'B', results_dir)
    plot_residual_analysis(result_B.x, datasets, 'B', results_dir)
    print("✅ 全プロット生成完了")

    # ==================== 結果保存 ====================
    print(f"\n{'='*80}\n結果保存\n{'='*80}")

    for form, result, std, corr in [('H', result_H, std_H, corr_H),
                                     ('B', result_B, std_B, corr_B)]:
        params_out = dict(zip(PARAM_NAMES, result.x))
        params_out['gamma'] = list(result.x[5:12])
        params_std_out = {f'{k}_std': v for k, v in zip(PARAM_NAMES, std)}

        df_params = pd.DataFrame([{**params_out, **params_std_out}])
        df_params.to_csv(results_dir / f'parameters_{form}.csv', index=False)
        print(f"  ✓ parameters_{form}.csv")

        # 相関行列
        df_corr = pd.DataFrame(corr, columns=PARAM_NAMES, index=PARAM_NAMES)
        df_corr.to_csv(results_dir / f'correlation_{form}.csv')
        print(f"  ✓ correlation_{form}.csv")

        # 全データセットの残差サマリー
        summary_rows = []
        g, a, B4, B6, eps_bg = result.x[0], result.x[1], result.x[2], result.x[3], result.x[4]
        gamma_array = result.x[5:12]
        for data in datasets:
            freq = data['freq']
            trans_obs = data['trans']
            trans_fit = calculate_transmission_for_params(
                freq, data['B'], data['T'], g, a, B4, B6, eps_bg, gamma_array, form)

            pol_mask = freq < POLARITON_UPPER
            cav_mask = freq >= CAVITY_LOWER
            rmse_all = np.sqrt(np.mean((trans_obs - trans_fit) ** 2))
            rmse_pol = np.sqrt(np.mean((trans_obs[pol_mask] - trans_fit[pol_mask]) ** 2)) if np.any(pol_mask) else np.nan
            rmse_cav = np.sqrt(np.mean((trans_obs[cav_mask] - trans_fit[cav_mask]) ** 2)) if np.any(cav_mask) else np.nan

            fwhm_pred, peak_freq_pred = compute_cavity_peak_info(freq, trans_fit)
            fwhm_obs = data['fwhm_obs']
            peak_freq_obs = data.get('peak_freq_obs')
            fwhm_err = abs(fwhm_pred - fwhm_obs) * 1000 if (fwhm_pred and fwhm_obs) else np.nan
            peak_err = abs(peak_freq_pred - peak_freq_obs) * 1000 if (peak_freq_pred and peak_freq_obs) else np.nan

            summary_rows.append({
                'label': data['label'],
                'B': data['B'], 'T': data['T'],
                'RMSE_all': rmse_all, 'RMSE_pol': rmse_pol, 'RMSE_cav': rmse_cav,
                'FWHM_obs_GHz': fwhm_obs * 1000 if fwhm_obs else np.nan,
                'FWHM_pred_GHz': fwhm_pred * 1000 if fwhm_pred else np.nan,
                'FWHM_err_GHz': fwhm_err,
                'peak_freq_obs_THz': peak_freq_obs if peak_freq_obs else np.nan,
                'peak_freq_pred_THz': peak_freq_pred if peak_freq_pred else np.nan,
                'peak_err_GHz': peak_err,
            })

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(results_dir / f'summary_{form}.csv', index=False)
        print(f"  ✓ summary_{form}.csv")

    # 総合評価JSON
    eval_results = {
        'H_form': {
            'success': result_H.success,
            'cost': float(result_H.cost),
            'n_residuals': len(result_H.fun),
            'nfev': result_H.nfev,
            'params': dict(zip(PARAM_NAMES, [float(x) for x in result_H.x])),
            'param_std': dict(zip(PARAM_NAMES, [float(x) for x in std_H])),
        },
        'B_form': {
            'success': result_B.success,
            'cost': float(result_B.cost),
            'n_residuals': len(result_B.fun),
            'nfev': result_B.nfev,
            'params': dict(zip(PARAM_NAMES, [float(x) for x in result_B.x])),
            'param_std': dict(zip(PARAM_NAMES, [float(x) for x in std_B])),
        },
        'comparison': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                       for k, v in comparison.items() if k not in ('H', 'B')},
        'comparison_H': {k: float(v) for k, v in comparison.get('H', {}).items()},
        'comparison_B': {k: float(v) for k, v in comparison.get('B', {}).items()},
        'timestamp': timestamp,
        'method': 'WLS (scipy.optimize.least_squares, trf)',
        'sigma_fwhm': SIGMA_FWHM,
        'sigma_peak_freq': SIGMA_PEAK_FREQ,
        'use_background_residual': USE_BACKGROUND_RESIDUAL,
    }
    with open(results_dir / 'model_evaluation.json', 'w') as fh:
        json.dump(eval_results, fh, indent=2, default=str)
    print("  ✓ model_evaluation.json")

    total_time = time.time() - start_time
    print(f"\n{'='*80}\n🎉 全処理完了\n{'='*80}")
    print(f"  実行時間  : {total_time:.1f} 秒 ({total_time/60:.1f} 分)")
    print(f"  結果保存先: {results_dir}")
    print(f"  手法      : 重み付き最小二乗法 (Mixed Residual)")
    print(f"  推奨モデル: {comparison.get('winner_aic', 'N/A')} (AIC)")
    print(f"  推奨モデル: {comparison.get('winner_bic', 'N/A')} (BIC)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
