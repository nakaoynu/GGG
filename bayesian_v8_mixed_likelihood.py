"""
Bayesian Hierarchical Analysis with Mixed Likelihood (v8.1)
領域別尤度: ポラリトン→透過率スペクトル形状, 共振器→FWHM + ピーク位置

【v8.1 改善 — レビュー Issue #1-#4 修正】
★ キャビティピーク位置尤度を追加 (Issue #1: Critical fix)
  - ピーク周波数の StudentT 尤度項を共振器領域に追加
  - FWHM のみでは捉えられなかったピーク位置の制約を実現
★ FWHM フォールバック修正 (Issue #2): ペナルティ値を使用
★ ベイズ因子計算の堅牢化 (Issue #3): flatten + nanmean
★ 領域別定量指標の追加 (Issue #4): Pol/Cav RMSE, Δf_peak

【v8.0 基盤機能 — ggg_research_strategy.pptx 解析方針実装】
★ 領域別複合尤度:
  - ポラリトン領域: p(D|Θ) ∝ T(ω, Θ, weight_pol) — スペクトル形状で推定
  - 高次共振器領域: p(D|Θ) ∝ FWHM(ω, Θ) + f_peak(ω, Θ) — FWHM+ピーク位置
  - 背景領域: weight 0.01 で包含
★ 複合尤度は pm.Potential で実装（log 尤度の加算）

【継承: test_fin_2a.py v7.1 Non-centered Hierarchical SMC】
- Non-centered 階層 γ モデル (log空間ハイパーパラメータ)
- SMC サンプリング (ESS>400目標)
- 物理的制約ベース事前分布
- H形式 / B形式の両モデル構築
- WAIC / ベイズファクターによるモデル比較

【事前分布設定 (v8.1 修士論文に基づく改訂)】
┌─────────┬──────────────┬────────────────────────────────────────┐
│ g       │TruncNormal   │理論値g≈2.0 (Gd³⁺), σ=0.05             │
│ a       │HalfNormal    │低値優先、σ=3.0、上限12                  │
│ B₄      │Normal        │負値許容、μ=0, σ=25mK, [-75,+75]mK     │
│ B₆      │Normal        │ゼロ中心対称、σ=5mK, [-25,+25]mK       │
│ ε_bg   │TruncNormal   │v6平均値中心、σ=0.3                    │
│log_γ_mu│Normal        │log空間で定義、μ=log(0.074)            │
│log_γ_sd│HalfNormal    │log空間標準偏差、σ=0.3                 │
│γ_raw_i │Normal(0,1)   │Non-centered: 標準正規分布              │
│ γ_i    │Deterministic │exp(log_μ + log_σ * z_i)               │
└─────────┴──────────────┴────────────────────────────────────────┘
"""

# ========== 設定 ==========
SAMPLER_TYPE = 'SMC'
USE_HIERARCHICAL_GAMMA = True
LIKELIHOOD_TYPE = 'mixed'       # 新: polariton=spectrum, cavity=FWHM
NU_STUDENTT = 4
RANDOM_SEED = 42

SMC_DRAWS   = 10000
SMC_CHAINS  = 16
SMC_PARALLEL = True

# 階層 γ ハイパーパラメータ (v7.1 準拠)
GAMMA_HYPERPRIOR_MU    = 0.074
GAMMA_HYPERPRIOR_SIGMA = 0.160
GAMMA_STD_PRIOR        = 0.092

# 【新設定】FWHM 尤度のノイズレベル [THz]
# ポラリトン尤度 (~100点) とスケール整合するよう調整
SIGMA_FWHM = 0.005     # 初期値 5 GHz — チューニング対象
SIGMA_PEAK_FREQ = 0.005  # THz (5 GHz) — キャビティピーク位置の尤度σ

# 背景領域尤度を使用するか
USE_BACKGROUND_LIKELIHOOD = True

# デバッグモード: True にすると 2 データセット・500 サンプルで高速テスト
DEBUG_MODE = False

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

import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from scipy.signal import find_peaks, peak_widths

import logging
logging.getLogger('pytensor').setLevel(logging.ERROR)

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

SCALING_FACTORS = {
    'g':    38.0,
    'a':    10.2,
    'B4':   1672.0,
    'B6':   25000.0,
    'eps':  17.0,
    'gamma': 100.0
}

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


# ============================================================================
# 物理関数群 (test_fin_2a.py v7.1 と同一)
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
    eps_mu    = np.where(eps_mu.real > 0, eps_mu, 0.1 + 1j * eps_mu.imag)
    n_complex = np.sqrt(eps_mu + 0j)
    impe      = np.sqrt(mu_r_safe / eps_bg + 0j)
    lambda_0  = np.where(omega > 1e-12, (2 * np.pi * c) / omega, np.inf)
    delta     = 2 * np.pi * n_complex * d / lambda_0
    delta     = np.clip(delta.real, -700, 700) + 1j * np.clip(delta.imag, -700, 700)
    numerator = 4 * impe
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
# 【新規】FWHM 計算ユーティリティ
# ============================================================================
POLARITON_UPPER = 0.361505   # THz — ポラリトン領域上限
CAVITY_LOWER    = 0.45       # THz — 共振器領域下限


def compute_fwhm_from_spectrum(freq, trans, cavity_lower=CAVITY_LOWER):
    """
    透過スペクトルの共振器ピーク (f >= cavity_lower) から FWHM を計算する。

    Returns
    -------
    fwhm_list : list[float]
        各共振器ピークの FWHM [THz]。ピークが見つからない場合は空リスト。
    peak_freq_list : list[float]
        各共振器ピークの中心周波数 [THz]。
    """
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


def compute_fwhm_representative(freq, trans, cavity_lower=CAVITY_LOWER):
    """
    共振器領域ピークのうち最も振幅の大きいものの FWHM を返す。
    ピークが存在しない場合は None を返す。
    """
    mask = freq >= cavity_lower
    if not np.any(mask):
        return None

    trans_cav = trans[mask]

    peaks, props = find_peaks(trans_cav, prominence=0.03, width=2)
    if len(peaks) == 0:
        return None

    prominences = props['prominences']
    best_idx    = np.argmax(prominences)
    peak_idx    = peaks[best_idx]

    widths_samples, _, _, _ = peak_widths(trans_cav, [peak_idx], rel_height=0.5)
    df = freq[1] - freq[0] if len(freq) > 1 else 1.0
    return float(widths_samples[0] * df)


def compute_cavity_peak_info(freq, trans, cavity_lower=CAVITY_LOWER):
    """
    共振器領域の最も顕著なピークの FWHM とピーク周波数を返す。
    Returns: (fwhm, peak_freq) — ピーク未検出時は (None, None)
    """
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


# ============================================================================
# ピーク検出・重み配列（test_fin_2a.py 準拠）
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

            # 【新規】観測 FWHM + ピーク位置の計算
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
                'fwhm_obs':         fwhm_obs,   # None if no cavity peak found
                'peak_freq_obs':    peak_freq_obs,  # None if no cavity peak found
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
# v6 結果読み込み（test_fin_2a.py 準拠）
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


# ============================================================================
# 【新規】MixedOutputModelOp
# ポラリトン: 透過率ベクトル, 共振器: FWHM スカラーを出力
# ============================================================================
class MixedOutputModelOp(Op):
    """
    v8.0 領域別尤度用 Op。
    perform() が trans_concat (全データセット連結) と
    fwhm_pred_vec (FWHM が存在するデータセット分) を出力する。
    """

    def __init__(self, datasets, model_form='H'):
        self.datasets    = datasets
        self.model_form  = model_form
        # FWHM が存在するデータセットのインデックスと観測値
        self.fwhm_indices = [i for i, d in enumerate(datasets) if d['fwhm_obs'] is not None]
        self.fwhm_obs_vec = np.array([datasets[i]['fwhm_obs'] for i in self.fwhm_indices])
        self.peak_freq_obs_vec = np.array([datasets[i]['peak_freq_obs'] for i in self.fwhm_indices])

    def make_node(self, a_scale_scaled, gamma_vec_scaled, g_factor_scaled,
                  B4_scaled, B6_scaled, eps_bg_scaled):
        a_scale_scaled  = pt.as_tensor_variable(a_scale_scaled)
        gamma_vec_scaled = pt.as_tensor_variable(gamma_vec_scaled)
        g_factor_scaled = pt.as_tensor_variable(g_factor_scaled)
        B4_scaled       = pt.as_tensor_variable(B4_scaled)
        B6_scaled       = pt.as_tensor_variable(B6_scaled)
        eps_bg_scaled   = pt.as_tensor_variable(eps_bg_scaled)

        out_trans     = pt.dvector()
        out_fwhm      = pt.dvector()
        out_peak_freq = pt.dvector()

        return Apply(self,
                     [a_scale_scaled, gamma_vec_scaled, g_factor_scaled,
                      B4_scaled, B6_scaled, eps_bg_scaled],
                     [out_trans, out_fwhm, out_peak_freq])

    def perform(self, _node, inputs, output_storage):
        a_scale_scaled, gamma_vec_scaled, g_factor_scaled, B4_scaled, B6_scaled, eps_bg_scaled = inputs

        g_factor = float(g_factor_scaled) / SCALING_FACTORS['g']
        a_scale  = float(a_scale_scaled)  / SCALING_FACTORS['a']
        B4       = float(B4_scaled)       / SCALING_FACTORS['B4']
        B6       = float(B6_scaled)       / SCALING_FACTORS['B6']
        eps_bg   = float(eps_bg_scaled)   / SCALING_FACTORS['eps']

        gamma_array_scaled = np.atleast_1d(gamma_vec_scaled).astype(np.float64)
        if len(gamma_array_scaled) != 7:
            gamma_array_scaled = np.full(7, gamma_array_scaled[0])
        gamma_array = gamma_array_scaled / SCALING_FACTORS['gamma']

        all_trans_pred = []
        fwhm_pred_list = []
        peak_freq_pred_list = []

        for idx, data in enumerate(self.datasets):
            freq = data['freq']
            B    = data['B']
            T    = data['T']

            H_ham   = get_hamiltonian(B, g_factor, B4, B6)
            chi_raw = calculate_susceptibility(freq, H_ham, T, gamma_array)
            G0      = a_scale * mu0 * N_SPIN * (g_factor * muB) ** 2 / (2 * hbar) / THZ_TO_RAD_S
            chi     = G0 * chi_raw

            if self.model_form == 'H':
                mu_r = 1.0 + chi
            else:
                denom = 1.0 - chi
                mu_r  = 1.0 / denom

            trans_pred = calculate_transmission(freq, mu_r, d_fixed, eps_bg)
            all_trans_pred.append(trans_pred)

            # FWHM + ピーク位置の計算（共振器ピークが存在するデータセットのみ）
            if idx in self.fwhm_indices:
                fwhm_val, peak_freq_val = compute_cavity_peak_info(freq, trans_pred)
                if fwhm_val is None:
                    # ペナルティ: 観測値の2倍を使用し大きな残差を発生させる
                    obs_idx = self.fwhm_indices.index(idx)
                    fwhm_val = self.fwhm_obs_vec[obs_idx] * 2.0
                    peak_freq_val = self.peak_freq_obs_vec[obs_idx] + 0.1
                fwhm_pred_list.append(fwhm_val)
                peak_freq_pred_list.append(peak_freq_val)

        output_storage[0][0] = np.concatenate(all_trans_pred)
        output_storage[1][0] = np.array(fwhm_pred_list) if fwhm_pred_list else np.array([0.0])
        output_storage[2][0] = np.array(peak_freq_pred_list) if peak_freq_pred_list else np.array([0.5])


# ============================================================================
# モデル評価（test_fin_2a.py 準拠）
# ============================================================================
def compute_model_evaluation(trace, model_name='Model'):
    print(f"\n{'='*80}\nモデル評価: {model_name}\n{'='*80}")
    result: dict = {'model_name': model_name}
    try:
        summary    = az.summary(trace)
        mean_rhat  = summary['r_hat'].mean()    if 'r_hat'     in summary.columns else np.nan
        mean_ess   = summary['ess_bulk'].mean() if 'ess_bulk' in summary.columns else np.nan
        print(f"  平均 R-hat : {mean_rhat:.4f}" if not np.isnan(mean_rhat) else "  R-hat: N/A")
        print(f"  平均 ESS   : {mean_ess:.1f}"  if not np.isnan(mean_ess)  else "  ESS  : N/A")
        result['mean_rhat'] = float(mean_rhat) if not np.isnan(mean_rhat) else None
        result['mean_ess']  = float(mean_ess) if not np.isnan(mean_ess) else None
    except Exception as e:
        print(f"  ⚠️ サマリー計算エラー: {e}")
    try:
        posterior    = trace.posterior
        n_chains     = posterior.dims.get('chain', 1)
        n_draws      = posterior.dims.get('draw', 0)
        result['n_chains'] = n_chains
        result['n_draws'] = n_draws
        result['total_samples'] = n_chains * n_draws
    except Exception as e:
        print(f"  ⚠️ 事後分布統計エラー: {e}")
    return result


def compare_models(eval_H, eval_B):
    print(f"\n{'='*80}\nモデル比較\n{'='*80}")
    ess_H = eval_H.get('mean_ess', 0) or 0
    ess_B = eval_B.get('mean_ess', 0) or 0
    print(f"  H-form ESS: {ess_H:.1f}  /  B-form ESS: {ess_B:.1f}")
    if ess_H > ess_B * 1.1:
        winner = "H-form"
    elif ess_B > ess_H * 1.1:
        winner = "B-form"
    else:
        winner = "引き分け"
    print(f"  🏆 推奨モデル: {winner}")
    return {'ess_H': ess_H, 'ess_B': ess_B, 'winner': winner, 'method': 'ESS comparison'}


def compute_bayes_factor_smc(trace_H, trace_B):
    print(f"\n{'='*80}\nベイズファクター計算\n{'='*80}")
    result = {}
    try:
        has_H = hasattr(trace_H, 'sample_stats') and 'log_marginal_likelihood' in trace_H.sample_stats
        has_B = hasattr(trace_B, 'sample_stats') and 'log_marginal_likelihood' in trace_B.sample_stats
        if has_H and has_B:
            lml_H_vals = np.asarray(trace_H.sample_stats['log_marginal_likelihood'].values).flatten()
            lml_B_vals = np.asarray(trace_B.sample_stats['log_marginal_likelihood'].values).flatten()
            lml_H = float(np.nanmean(lml_H_vals))
            lml_B = float(np.nanmean(lml_B_vals))
            lml_H_std = float(np.nanstd(lml_H_vals))
            lml_B_std = float(np.nanstd(lml_B_vals))
            log_BF = lml_H - lml_B
            log_BF_se = np.sqrt(lml_H_std ** 2 + lml_B_std ** 2)
            log10_BF  = log_BF / np.log(10)
            abs_log_BF = abs(log_BF)
            strength = ("ほぼ証拠なし" if abs_log_BF < 1.15 else
                        "弱い証拠"     if abs_log_BF < 2.3  else
                        "中程度の証拠"  if abs_log_BF < 4.6  else
                        "強い証拠")
            winner = "H-form" if log_BF > 0 else ("B-form" if log_BF < 0 else "引き分け")
            print(f"  H-form log(ML): {lml_H:.2f} ± {lml_H_std:.2f}")
            print(f"  B-form log(ML): {lml_B:.2f} ± {lml_B_std:.2f}")
            print(f"  log(BF_{{H/B}}): {log_BF:.2f} ± {log_BF_se:.2f}  → {strength}")
            print(f"  🏆 推奨モデル: {winner}")
            result = {'log_BF': log_BF, 'log10_BF': log10_BF,
                      'interpretation': strength, 'winner': winner,
                      'method': 'SMC marginal likelihood'}
        else:
            print("  ⚠️ SMC 周辺尤度が保存されていません")
            result = {'winner': 'N/A', 'method': 'unavailable'}
    except Exception as e:
        print(f"❌ ベイズファクター計算エラー: {e}")
        result = {'winner': 'N/A', 'error': str(e)}
    return result


# ============================================================================
# プロット関数（test_fin_2a.py から継承）
# ============================================================================
def plot_posterior_predictive_spectra(trace, datasets, model_form='H', save_dir=None, n_samples=300):
    print(f"\n{'='*80}\n事後予測スペクトルプロット ({model_form}-form)\n{'='*80}")
    posterior    = trace.posterior
    n_chains_val = posterior.dims['chain']
    n_draws_val  = posterior.dims['draw']
    total_samples = n_chains_val * n_draws_val
    if total_samples > n_samples:
        sample_indices = np.random.choice(total_samples, size=n_samples, replace=False)
    else:
        sample_indices = np.arange(total_samples)
        n_samples = total_samples

    n_datasets = len(datasets)
    ncols = 2
    nrows = (n_datasets + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    fig.suptitle(f'Posterior Predictive Spectra ({model_form}-form) — v8.0 Mixed Likelihood',
                 fontsize=12, y=0.995)
    axes = axes.flatten()

    for idx, data in enumerate(datasets):
        ax        = axes[idx]
        freq      = data['freq']
        trans_obs = data['trans']
        B, T      = data['B'], data['T']
        label     = data['label']
        fwhm_obs  = data['fwhm_obs']

        trans_samples = np.zeros((n_samples, len(freq)))
        for i, sample_idx in enumerate(sample_indices):
            ci = sample_idx // n_draws_val
            di = sample_idx % n_draws_val
            g   = float(posterior['g_factor_scaled'].values[ci, di]) / SCALING_FACTORS['g']
            a   = float(posterior['a_scale_scaled'].values[ci, di])  / SCALING_FACTORS['a']
            B4  = float(posterior['B4_scaled'].values[ci, di])       / SCALING_FACTORS['B4']
            B6  = float(posterior['B6_scaled'].values[ci, di])       / SCALING_FACTORS['B6']
            eps = float(posterior['eps_bg_scaled'].values[ci, di])   / SCALING_FACTORS['eps']
            gamma_array = np.array([
                float(posterior[f'gamma_{j+1}_scaled'].values[ci, di]) / SCALING_FACTORS['gamma']
                for j in range(7)
            ])
            trans_samples[i] = calculate_transmission_for_params(freq, B, T, g, a, B4, B6, eps, gamma_array, model_form)

        trans_median = np.median(trans_samples, axis=0)
        trans_hdi    = az.hdi(trans_samples, hdi_prob=0.94)

        fwhm_pred, peak_freq_pred = compute_cavity_peak_info(freq, trans_median)
        peak_freq_obs = data.get('peak_freq_obs')

        # 領域別 RMSE
        pol_mask = freq < POLARITON_UPPER
        cav_mask = freq >= CAVITY_LOWER
        rmse_pol = np.sqrt(np.mean((trans_obs[pol_mask] - trans_median[pol_mask]) ** 2)) if np.any(pol_mask) else np.nan
        rmse_cav = np.sqrt(np.mean((trans_obs[cav_mask] - trans_median[cav_mask]) ** 2)) if np.any(cav_mask) else np.nan

        # ピーク位置誤差
        peak_err_ghz = abs(peak_freq_pred - peak_freq_obs) * 1000 if (peak_freq_pred and peak_freq_obs) else np.nan

        # 領域ハイライト
        for f_s, f_e in data['polariton_regions']:
            ax.axvspan(f_s, f_e, alpha=0.12, color='orange', label='Polariton' if f_s == data['polariton_regions'][0][0] else None)
        for f_s, f_e in data['cavity_regions']:
            ax.axvspan(f_s, f_e, alpha=0.12, color='green',  label='Cavity'    if f_s == data['cavity_regions'][0][0]    else None)

        ax.plot(freq, trans_obs,    'ko',  markersize=2.5, alpha=0.6, label='Obs')
        ax.plot(freq, trans_median, 'r-',  lw=2,           label='Median')
        ax.fill_between(freq, trans_hdi[:, 0], trans_hdi[:, 1], color='red', alpha=0.2, label='94% HDI')

        rmse = np.sqrt(np.mean((trans_obs - trans_median) ** 2))
        fwhm_str = (f"FWHM obs={fwhm_obs*1000:.1f} pred={fwhm_pred*1000:.1f} GHz"
                    if fwhm_obs and fwhm_pred else "")
        region_str = f"Pol={rmse_pol:.4f} Cav={rmse_cav:.4f}"
        peak_str = f" Δf={peak_err_ghz:.1f}GHz" if not np.isnan(peak_err_ghz) else ""
        ax.set_title(f"{label}  RMSE={rmse:.4f} ({region_str}){peak_str}\n{fwhm_str}",
                     fontsize=8, fontweight='bold')
        ax.set_xlabel('Frequency (THz)', fontsize=9)
        ax.set_ylabel('Transmittance',   fontsize=9)
        ax.legend(fontsize=6, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim([freq.min(), freq.max()])
        ax.set_ylim([0, 1.05])

    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')
    plt.tight_layout()
    if save_dir:
        path = save_dir / f'posterior_predictive_spectra_{model_form}.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ {path.name} saved")
    plt.close()


def plot_posterior_distributions(trace, model_form='H', save_dir=None):
    print(f"\n事後分布プロット作成 ({model_form}-form)...")
    posterior = trace.posterior
    param_info = [
        ('g_factor_scaled', SCALING_FACTORS['g'],   'g-factor',  ''),
        ('a_scale_scaled',  SCALING_FACTORS['a'],   'a (coupling)', ''),
        ('B4_scaled',       SCALING_FACTORS['B4'],  'B₄',         'mK'),
        ('B6_scaled',       SCALING_FACTORS['B6'],  'B₆',         'mK'),
        ('eps_bg_scaled',   SCALING_FACTORS['eps'], 'ε_bg',      ''),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Posterior Distributions ({model_form}-form) — v8.0', fontsize=14)
    axes = axes.flatten()
    for i, (var, scale, label, unit) in enumerate(param_info):
        ax = axes[i]
        samples = posterior[var].values.flatten() / scale
        if unit == 'mK':
            samples *= 1000
            xlabel = f'{label} ({unit})'
        else:
            xlabel = label
        ax.hist(samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black', lw=0.5)
        mean_val   = np.mean(samples)
        median_val = np.median(samples)
        hdi        = az.hdi(samples, hdi_prob=0.94)
        ax.axvline(mean_val,   color='red',    linestyle='--', lw=1.5, label=f'Mean: {mean_val:.3g}')
        ax.axvline(median_val, color='orange', linestyle='-.', lw=1.5, label=f'Med: {median_val:.3g}')
        ax.axvspan(hdi[0], hdi[1], alpha=0.2, color='green', label='94% HDI')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    for i in range(7):
        ax = axes[5 + i]
        var_name = f'gamma_{i+1}_scaled'
        samples  = posterior[var_name].values.flatten() / SCALING_FACTORS['gamma']
        ax.hist(samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black', lw=0.5)
        mean_val   = np.mean(samples)
        median_val = np.median(samples)
        hdi        = az.hdi(samples, hdi_prob=0.94)
        ax.axvline(mean_val,   color='red',    linestyle='--', lw=1.5, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='orange', linestyle='-.', lw=1.5, label=f'Med: {median_val:.2f}')
        ax.axvspan(hdi[0], hdi[1], alpha=0.2, color='green', label='94% HDI')
        ax.set_xlabel(f'γ_{i+1} (THz)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'γ_{i+1}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_dir:
        path = save_dir / f'posterior_distributions_{model_form}.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ {path.name} saved")
    plt.close()


# ============================================================================
# 【新規】エネルギー準位図 (点プロット + 94% HDI)
# ============================================================================
def plot_energy_levels(trace, datasets, model_form='H', save_dir=None, n_samples=300):
    """
    事後分布からサンプリングしたパラメータでエネルギー固有値を計算し、
    磁場 B に対するエネルギー準位図を点でプロットする。94% HDI バンド付き。
    """
    print(f"\nエネルギー準位プロット作成 ({model_form}-form)...")
    posterior = trace.posterior
    n_chains_val = posterior.dims['chain']
    n_draws_val  = posterior.dims['draw']
    total_samples = n_chains_val * n_draws_val
    if total_samples > n_samples:
        sample_indices = np.random.choice(total_samples, size=n_samples, replace=False)
    else:
        sample_indices = np.arange(total_samples)
        n_samples = total_samples

    B_fields = np.linspace(0, 10, 60)
    n_states = int(2 * S_VALUE + 1)

    # shape (n_samples, n_B, n_states)
    all_eigenvals = np.zeros((n_samples, len(B_fields), n_states))

    for i, sample_idx in enumerate(sample_indices):
        ci = sample_idx // n_draws_val
        di = sample_idx % n_draws_val
        g  = float(posterior['g_factor_scaled'].values[ci, di]) / SCALING_FACTORS['g']
        B4 = float(posterior['B4_scaled'].values[ci, di])       / SCALING_FACTORS['B4']
        B6 = float(posterior['B6_scaled'].values[ci, di])       / SCALING_FACTORS['B6']
        for j, B_val in enumerate(B_fields):
            H_ham = get_hamiltonian(B_val, g, B4, B6)
            evals = np.sort(np.linalg.eigh(H_ham)[0])
            all_eigenvals[i, j] = evals

    # 基底状態を基準にシフト (各サンプルの B=0 での最低エネルギーを 0 に)
    ground_shift = all_eigenvals[:, :, 0:1]  # shape (n_samples, n_B, 1)
    all_eigenvals_shifted = all_eigenvals - ground_shift

    median_evals = np.median(all_eigenvals_shifted, axis=0)
    hdi_lo = np.zeros((len(B_fields), n_states))
    hdi_hi = np.zeros((len(B_fields), n_states))
    for j in range(len(B_fields)):
        for k in range(n_states):
            hdi = az.hdi(all_eigenvals_shifted[:, j, k], hdi_prob=0.94)
            hdi_lo[j, k] = hdi[0]
            hdi_hi[j, k] = hdi[1]

    _fig, ax = plt.subplots(figsize=(10, 7))
    colors = [plt.colormaps['tab10'](i / n_states) for i in range(n_states)]
    for k in range(n_states):
        ax.scatter(B_fields, median_evals[:, k], c=[colors[k]], s=12, zorder=3,
                   label=f'|{k}⟩', edgecolors='none')
        ax.fill_between(B_fields, hdi_lo[:, k], hdi_hi[:, k],
                        color=colors[k], alpha=0.15)

    # データセットの磁場条件を縦線で表示
    dataset_Bs = sorted(set(d['B'] for d in datasets))
    for Bval in dataset_Bs:
        ax.axvline(Bval, color='gray', linestyle=':', alpha=0.4, lw=0.8)

    ax.set_xlabel('Magnetic Field B (T)', fontsize=12)
    ax.set_ylabel('Energy E − E₀ (K)', fontsize=12)
    ax.set_title(f'Energy Level Diagram ({model_form}-form) — 94% HDI\n'
                 f'S={S_VALUE}, {n_samples} posterior samples', fontsize=12)
    ax.legend(fontsize=7, ncol=4, loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_dir:
        path = save_dir / f'energy_levels_{model_form}.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ {path.name} saved")
    plt.close()


# ============================================================================
# 【新規】磁気感受率プロット (実部・虚部別, 94% HDI)
# ============================================================================
def plot_susceptibility(trace, datasets, model_form='H', save_dir=None, n_samples=200):
    """
    事後分布からサンプリングしたパラメータで磁気感受率 χ+(ω) を計算し、
    Re(χ+) と Im(χ+) を別々の図にプロットする。94% HDI バンド付き。
    """
    print(f"\n磁気感受率プロット作成 ({model_form}-form)...")
    posterior = trace.posterior
    n_chains_val = posterior.dims['chain']
    n_draws_val  = posterior.dims['draw']
    total_samples = n_chains_val * n_draws_val
    if total_samples > n_samples:
        sample_indices = np.random.choice(total_samples, size=n_samples, replace=False)
    else:
        sample_indices = np.arange(total_samples)
        n_samples = total_samples

    n_datasets = len(datasets)
    ncols = 2
    nrows = (n_datasets + 1) // 2

    fig_re, axes_re = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    fig_re.suptitle(f'Magnetic Susceptibility Re(χ₊) ({model_form}-form) — 94% HDI',
                    fontsize=12, y=0.995)
    axes_re = axes_re.flatten()

    fig_im, axes_im = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    fig_im.suptitle(f'Magnetic Susceptibility Im(χ₊) ({model_form}-form) — 94% HDI',
                    fontsize=12, y=0.995)
    axes_im = axes_im.flatten()

    for idx, data in enumerate(datasets):
        freq  = data['freq']
        B_ext = data['B']
        T_val = data['T']
        label = data['label']

        chi_re_samples = np.zeros((n_samples, len(freq)))
        chi_im_samples = np.zeros((n_samples, len(freq)))

        for i, sample_idx in enumerate(sample_indices):
            ci = sample_idx // n_draws_val
            di = sample_idx % n_draws_val
            g   = float(posterior['g_factor_scaled'].values[ci, di]) / SCALING_FACTORS['g']
            a   = float(posterior['a_scale_scaled'].values[ci, di])  / SCALING_FACTORS['a']
            B4  = float(posterior['B4_scaled'].values[ci, di])       / SCALING_FACTORS['B4']
            B6  = float(posterior['B6_scaled'].values[ci, di])       / SCALING_FACTORS['B6']
            gamma_array = np.array([
                float(posterior[f'gamma_{j+1}_scaled'].values[ci, di]) / SCALING_FACTORS['gamma']
                for j in range(7)
            ])

            H_ham   = get_hamiltonian(B_ext, g, B4, B6)
            chi_raw = calculate_susceptibility(freq, H_ham, T_val, gamma_array)
            G0      = a * mu0 * N_SPIN * (g * muB) ** 2 / (2 * hbar) / THZ_TO_RAD_S
            chi     = G0 * chi_raw

            chi_re_samples[i] = np.real(chi)
            chi_im_samples[i] = np.imag(chi)

        # Re(χ+)
        ax_re = axes_re[idx]
        re_median = np.median(chi_re_samples, axis=0)
        re_hdi    = az.hdi(chi_re_samples, hdi_prob=0.94)
        ax_re.plot(freq, re_median, 'b-', lw=1.5, label='Median')
        ax_re.fill_between(freq, re_hdi[:, 0], re_hdi[:, 1],
                           color='blue', alpha=0.2, label='94% HDI')
        ax_re.set_title(f'{label} (B={B_ext}T, T={T_val}K)', fontsize=9, fontweight='bold')
        ax_re.set_xlabel('Frequency (THz)', fontsize=8)
        ax_re.set_ylabel('Re(χ₊)', fontsize=8)
        ax_re.legend(fontsize=6, loc='best')
        ax_re.grid(alpha=0.3)
        ax_re.axhline(0, color='gray', lw=0.5)

        # Im(χ+)
        ax_im = axes_im[idx]
        im_median = np.median(chi_im_samples, axis=0)
        im_hdi    = az.hdi(chi_im_samples, hdi_prob=0.94)
        ax_im.plot(freq, im_median, 'r-', lw=1.5, label='Median')
        ax_im.fill_between(freq, im_hdi[:, 0], im_hdi[:, 1],
                           color='red', alpha=0.2, label='94% HDI')
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


# ============================================================================
# ベイズモデル構築ヘルパー（H / B 共通）
# ============================================================================
def build_pymc_model(datasets, model_form, v6_params_H, v6_params_B):
    """
    v8.0 複合尤度モデルを構築して返す。

    尤度構造:
    - ポラリトン領域: StudentT(ν=4) で trans_pred vs trans_obs
    - 共振器領域   : StudentT(ν=4) で fwhm_pred  vs fwhm_obs
    - 背景領域     : StudentT(ν=4) で trans_pred vs trans_obs (weight=0.01)
    """
    _v6_params = v6_params_H if model_form == 'H' else v6_params_B
    eps_v6_avg = (v6_params_H['eps'] + v6_params_B['eps']) / 2

    # Op の生成
    model_op = MixedOutputModelOp(datasets, model_form)

    # 観測データ
    trans_obs_list  = [d['trans']  for d in datasets]
    weight_list     = [d['weight'] for d in datasets]

    # 領域マスク（ポラリトン / 共振器 / 背景）
    polariton_masks = []
    bg_masks        = []
    for d in datasets:
        weight = d['weight']
        polariton_masks.append(weight == 2.0)
        bg_masks.append(weight == 0.01)

    # 観測値の連結
    trans_obs_concat = np.concatenate(trans_obs_list)
    weight_concat    = np.concatenate(weight_list)
    polariton_mask_concat = np.concatenate(polariton_masks)
    bg_mask_concat        = np.concatenate(bg_masks)

    # 有効 σ (weight が大きいほど σ が小さい = 精度が高い)
    sigma_eff = 0.01 / np.sqrt(weight_concat)

    # FWHM 観測値 + ピーク位置観測値
    fwhm_obs_vec = model_op.fwhm_obs_vec  # shape (n_fwhm,)
    peak_freq_obs_vec = model_op.peak_freq_obs_vec  # shape (n_fwhm,)
    sigma_fwhm   = np.full_like(fwhm_obs_vec, SIGMA_FWHM) if len(fwhm_obs_vec) > 0 else np.array([SIGMA_FWHM])
    sigma_peak_freq = np.full_like(peak_freq_obs_vec, SIGMA_PEAK_FREQ) if len(peak_freq_obs_vec) > 0 else np.array([SIGMA_PEAK_FREQ])

    model = pm.Model()
    with model:
        # ------------------------------------------
        # 1. g 因子: TruncNormal (理論値 Gd³⁺ ≈ 2.0)
        # ------------------------------------------
        g_factor_scaled = pm.TruncatedNormal(
            'g_factor_scaled',
            mu=2.0 * SCALING_FACTORS['g'],
            sigma=0.05 * SCALING_FACTORS['g'],
            lower=1.5 * SCALING_FACTORS['g'],
            upper=2.8 * SCALING_FACTORS['g'])

        # ------------------------------------------
        # 2. a: HalfNormal + clip
        # ------------------------------------------
        a_raw_name = f'a_raw_{model_form}'
        a_raw = pm.HalfNormal(a_raw_name, sigma=3.0)
        a_scale_scaled = pm.Deterministic('a_scale_scaled',
            pt.clip(a_raw, 0.1, 12.0) * SCALING_FACTORS['a'])

        # ------------------------------------------
        # 3. B₄: Normal (負値許容, 修士論文 Table 3.7 改訂)
        # ------------------------------------------
        B4_raw_name = f'B4_raw_{model_form}'
        B4_raw = pm.Normal(B4_raw_name, mu=0, sigma=0.025)
        B4_scaled = pm.Deterministic('B4_scaled',
            pt.clip(B4_raw, -0.075, 0.075) * SCALING_FACTORS['B4'])

        # ------------------------------------------
        # 4. B₆: Normal + clip
        # ------------------------------------------
        B6_raw_name = f'B6_raw_{model_form}'
        B6_raw = pm.Normal(B6_raw_name, mu=0, sigma=0.005)
        B6_scaled = pm.Deterministic('B6_scaled',
            pt.clip(B6_raw, -0.025, 0.025) * SCALING_FACTORS['B6'])

        # ------------------------------------------
        # 5. ε_bg: TruncNormal
        # ------------------------------------------
        eps_bg_scaled = pm.TruncatedNormal(
            'eps_bg_scaled',
            mu=eps_v6_avg * SCALING_FACTORS['eps'],
            sigma=0.3 * SCALING_FACTORS['eps'],
            lower=13.0 * SCALING_FACTORS['eps'],
            upper=16.0 * SCALING_FACTORS['eps'])

        # ------------------------------------------
        # 6. γ: Non-centered 階層モデル (v7.1 継承)
        # ------------------------------------------
        log_gamma_mu = pm.Normal('log_gamma_mu',
            mu=np.log(GAMMA_HYPERPRIOR_MU), sigma=0.3)
        log_gamma_sd = pm.HalfNormal('log_gamma_sd', sigma=0.3)

        gamma_raw = pm.Normal('gamma_raw', mu=0, sigma=1, shape=7)
        gamma_vec_unscaled = pm.Deterministic('gamma_vec',
            pt.exp(log_gamma_mu + log_gamma_sd * gamma_raw))
        gamma_vec_scaled = pm.Deterministic('gamma_vec_scaled',
            pt.clip(gamma_vec_unscaled, 0.005, 0.5) * SCALING_FACTORS['gamma'])
        for i in range(7):
            pm.Deterministic(f'gamma_{i+1}_scaled', gamma_vec_scaled[i])
        pm.Deterministic('gamma_mean_scaled',
            pt.exp(log_gamma_mu) * SCALING_FACTORS['gamma'])
        pm.Deterministic('gamma_std_scaled',
            log_gamma_sd * SCALING_FACTORS['gamma'])

        # ------------------------------------------
        # 7. 【新規】MixedOutputModelOp で trans_pred + fwhm_pred + peak_freq_pred を取得
        # ------------------------------------------
        trans_pred_concat, fwhm_pred_vec, peak_freq_pred_vec = model_op(
            a_scale_scaled, gamma_vec_scaled, g_factor_scaled,
            B4_scaled, B6_scaled, eps_bg_scaled)

        # ------------------------------------------
        # 8. 【新規】複合尤度: pm.Potential で log 尤度を加算
        # ------------------------------------------
        # 8-a. ポラリトン領域 (スペクトル形状)
        # PyMC v5 では pm.logp(dist, value) を使用する
        if np.any(polariton_mask_concat):
            obs_pol  = trans_obs_concat[polariton_mask_concat]
            pred_pol = trans_pred_concat[polariton_mask_concat]
            sig_pol  = sigma_eff[polariton_mask_concat]
            dist_pol = pm.StudentT.dist(nu=NU_STUDENTT, mu=pred_pol, sigma=sig_pol)
            ll_pol   = pm.logp(dist_pol, obs_pol)
            pm.Potential('ll_polariton', ll_pol.sum())

        # 8-b. 共振器領域 (FWHM + ピーク位置)
        if len(fwhm_obs_vec) > 0:
            dist_fwhm = pm.StudentT.dist(nu=NU_STUDENTT, mu=fwhm_pred_vec, sigma=sigma_fwhm)
            ll_fwhm   = pm.logp(dist_fwhm, fwhm_obs_vec)
            pm.Potential('ll_cavity_fwhm', ll_fwhm.sum())

            # 8-b2. 共振器ピーク位置マッチング (Issue #1 修正)
            dist_peak = pm.StudentT.dist(nu=NU_STUDENTT, mu=peak_freq_pred_vec, sigma=sigma_peak_freq)
            ll_peak   = pm.logp(dist_peak, peak_freq_obs_vec)
            pm.Potential('ll_cavity_peak_freq', ll_peak.sum())

        # 8-c. 背景領域 (スペクトル、weight=0.01)
        if USE_BACKGROUND_LIKELIHOOD and np.any(bg_mask_concat):
            obs_bg  = trans_obs_concat[bg_mask_concat]
            pred_bg = trans_pred_concat[bg_mask_concat]
            sig_bg  = sigma_eff[bg_mask_concat]
            dist_bg = pm.StudentT.dist(nu=NU_STUDENTT, mu=pred_bg, sigma=sig_bg)
            ll_bg   = pm.logp(dist_bg, obs_bg)
            pm.Potential('ll_background', ll_bg.sum())

    return model


# ============================================================================
# メイン処理
# ============================================================================
def main():
    global TARGET_DATA, SMC_DRAWS, SMC_CHAINS

    start_time = time.time()

    if DEBUG_MODE:
        print("\n" + "🔧" * 40)
        print("デバッグモード: ON (2データセット, 500サンプル)")
        print("🔧" * 40 + "\n")
        TARGET_DATA = TARGET_DATA[:2]
        SMC_DRAWS   = 500
        SMC_CHAINS  = 2

    print(f"\n{'='*80}")
    print("Bayesian Analysis v8.1 — Mixed Likelihood (Polariton: spectrum / Cavity: FWHM + peak position)")
    print(f"ggg_research_strategy.pptx 解析方針実装 + レビュー Issue #1-#4 修正")
    print(f"{'='*80}")

    # v6 参照値読み込み
    v6_params_H = load_v6_optimized_params('H')
    v6_params_B = load_v6_optimized_params('B')
    if v6_params_H is None or v6_params_B is None:
        print("❌ v6 最適化結果の読み込みに失敗しました")
        return

    # 結果ディレクトリ
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = pathlib.Path(__file__).parent / f"bayesian_v8_results_{timestamp}"
    results_dir.mkdir(exist_ok=True)
    print(f"\n📁 結果保存先: {results_dir}")

    # データ読み込み
    datasets = load_all_datasets(TARGET_DATA)
    if not datasets:
        print("❌ データがありません")
        return

    print(f"\n{'='*80}\nH 形式モデル構築・サンプリング\n{'='*80}")
    with build_pymc_model(datasets, 'H', v6_params_H, v6_params_B):
        print(f"  Draws={SMC_DRAWS}, Chains={SMC_CHAINS}")
        trace_H = pm.sample_smc(
            draws=SMC_DRAWS, chains=SMC_CHAINS,
            cores=SMC_CHAINS if SMC_PARALLEL else 1,
            return_inferencedata=True, progressbar=True,
            random_seed=RANDOM_SEED)
    print("✅ H 形式サンプリング完了")

    print(f"\n{'='*80}\nB 形式モデル構築・サンプリング\n{'='*80}")
    with build_pymc_model(datasets, 'B', v6_params_H, v6_params_B):
        print(f"  Draws={SMC_DRAWS}, Chains={SMC_CHAINS}")
        trace_B = pm.sample_smc(
            draws=SMC_DRAWS, chains=SMC_CHAINS,
            cores=SMC_CHAINS if SMC_PARALLEL else 1,
            return_inferencedata=True, progressbar=True,
            random_seed=RANDOM_SEED)
    print("✅ B 形式サンプリング完了")

    # モデル評価
    eval_H = compute_model_evaluation(trace_H, 'H-form')
    eval_B = compute_model_evaluation(trace_B, 'B-form')
    comparison_result = compare_models(eval_H, eval_B)
    bf_result = compute_bayes_factor_smc(trace_H, trace_B)

    # 可視化
    print(f"\n{'='*80}\n📊 プロット生成\n{'='*80}")
    plot_posterior_distributions(trace_H, 'H', results_dir)
    plot_posterior_predictive_spectra(trace_H, datasets, 'H', results_dir)
    plot_posterior_distributions(trace_B, 'B', results_dir)
    plot_posterior_predictive_spectra(trace_B, datasets, 'B', results_dir)
    plot_energy_levels(trace_H, datasets, 'H', results_dir)
    plot_susceptibility(trace_H, datasets, 'H', results_dir)
    plot_energy_levels(trace_B, datasets, 'B', results_dir)
    plot_susceptibility(trace_B, datasets, 'B', results_dir)
    print("✅ 全プロット生成完了")

    # 結果保存
    print(f"\n{'='*80}\n結果保存\n{'='*80}")
    for form, trace in [('H', trace_H), ('B', trace_B)]:
        try:
            trace.to_netcdf(str(results_dir / f'trace_{form}.nc'))
            print(f"  ✓ trace_{form}.nc")
        except Exception:
            import pickle
            with open(results_dir / f'trace_{form}.pkl', 'wb') as fh:
                pickle.dump(trace, fh)
            print(f"  ✓ trace_{form}.pkl (pickle)")

        summary = az.summary(trace)
        summary.to_csv(results_dir / f'summary_{form}.csv')
        print(f"  ✓ summary_{form}.csv")

        posterior = trace.posterior
        params_out = {
            'g':   float(posterior['g_factor_scaled'].mean()) / SCALING_FACTORS['g'],
            'a':   float(posterior['a_scale_scaled'].mean())  / SCALING_FACTORS['a'],
            'B4':  float(posterior['B4_scaled'].mean())       / SCALING_FACTORS['B4'],
            'B6':  float(posterior['B6_scaled'].mean())       / SCALING_FACTORS['B6'],
            'eps': float(posterior['eps_bg_scaled'].mean())   / SCALING_FACTORS['eps'],
            'gamma': [float(posterior[f'gamma_{i+1}_scaled'].mean()) / SCALING_FACTORS['gamma'] for i in range(7)],
            'gamma_mean': float(posterior['gamma_mean_scaled'].mean()) / SCALING_FACTORS['gamma'],
        }
        pd.DataFrame([{k: v if not isinstance(v, list) else str(v) for k, v in params_out.items()}]).to_csv(
            results_dir / f'parameters_{form}.csv', index=False)
        print(f"  ✓ parameters_{form}.csv")

    eval_results = {
        'H_form': eval_H, 'B_form': eval_B,
        'comparison_ess': comparison_result,
        'comparison_bayes_factor': bf_result,
        'timestamp': timestamp,
        'sampler': SAMPLER_TYPE,
        'likelihood': LIKELIHOOD_TYPE,
        'sigma_fwhm': SIGMA_FWHM,
        'use_background_likelihood': USE_BACKGROUND_LIKELIHOOD,
    }
    with open(results_dir / 'model_evaluation.json', 'w') as fh:
        json.dump(eval_results, fh, indent=2, default=str)
    print("  ✓ model_evaluation.json")

    total_time = time.time() - start_time
    print(f"\n{'='*80}\n🎉 全処理完了\n{'='*80}")
    print(f"  実行時間  : {total_time:.1f} 秒 ({total_time/60:.1f} 分)")
    print(f"  結果保存先: {results_dir}")
    print(f"  尤度      : {LIKELIHOOD_TYPE} (ポラリトン=スペクトル形状 / 共振器=FWHM)")
    print(f"  推奨モデル: {comparison_result.get('winner', 'N/A')} (ESS比較)")
    if bf_result:
        print(f"  推奨モデル: {bf_result.get('winner', 'N/A')} (ベイズファクター)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
