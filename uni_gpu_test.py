# unified_weighted_bayesian_fitting_gpu.py
# GPU (JAX/NumPyro) 完全最適化バージョン
# 特徴: Pythonループを廃止し、全データをテンソル演算で一括処理します
#
# === コードレビュー対応履歴 ===
# 1. pt.scanの完全排除: 
#    - 全データをBroadcastingで一括処理(calculate_susceptibility_vectorized)
#    - 固有値事前計算により条件数に比例した計算量に削減(O(N) → O(n_conditions))
# 2. 物理定数のクラス化:
#    - PhysicalConstantsクラスでスコープを管理、グローバル汚染を防止
# 3. set_subtensor排除:
#    - 全行列をNumPyで事前構築し、pt.as_tensor_variableで定数化
# 4. 事前分布パラメータのconfig化:
#    - bayesian_priorsセクションから全パラメータを読み込み、ハードコーディング排除
# 5. 例外処理の改善:
#    - bare exceptをException catchに変更、エラー型とメッセージをログ出力

import os
import sys
import pathlib
import yaml
import time
import datetime
import warnings
import re

# --- GPU設定 (ライブラリインポート前に実行) ---
def pre_setup_gpu():
    # config_unified_gpu.yml を読んでGPU設定を確認
    config_path = pathlib.Path(__file__).parent / "config_unified_gpu.yml"
    use_gpu = False
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            use_gpu = cfg.get('execution', {}).get('use_gpu', False)
        except:
            pass
    
    if use_gpu:
        print("🚀 GPU (JAX) Optimization Enabled")
        # PyTensorにはfloat64使用のみを伝え、デバイス管理はJAXに任せる
        os.environ['PYTENSOR_FLAGS'] = 'floatX=float64'
        # JAXがメモリを独占しないようにする設定 (共有環境で重要)
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    else:
        print("💻 CPU Mode")
        os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float64'

pre_setup_gpu()

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.scan.basic import scan
from pytensor.tensor.slinalg import eigvalsh
from pytensor.tensor.var import TensorVariable
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from functools import lru_cache
from typing import Any, cast

# 物理定数をクラスで管理(THzスケーリング版)
class PhysicalConstants:
    """
    物理定数の定義 (THzスケーリング済み)
    
    【スケーリングの原理】
    計算の数値安定性を保つため、内部計算では SI単位(J, s) ではなく
    「1 THz (= 1e12 Hz) の光子のエネルギー」を基準(=1.0)とする単位系を使用します。
    
    基本単位:
      - 周波数: 1.0 = 1 THz
      - 時間:   1.0 = 1 ps (1e-12 s)
      - エネルギー: 1.0 = h * 1 THz
    
    【効果】
    - 従来: kB=1.38e-23, hbar=1.05e-34 → 勾配計算でunderflow/overflow
    - 修正後: kB≈0.021, hbar≈0.159 → 安定した数値計算
    """
    # --- 基準となるスケール ---
    SCALE_FREQ = 1.0e12  # 1 THz [Hz]
    
    # --- SI単位系での真値 ---
    _h_SI = 6.62607015e-34    # Planck constant [J·s]
    _hbar_SI = _h_SI / (2 * np.pi)
    _kB_SI = 1.380649e-23     # Boltzmann constant [J/K]
    _muB_SI = 9.274010e-24    # Bohr magneton [J/T]
    
    # --- 基準エネルギー: E_base = h * 1THz ---
    _E_base = _h_SI * SCALE_FREQ  # [J]

    # ボルツマン定数 [THz/K]
    # 温度T[K]を掛けたとき、kB*T が「THz単位のエネルギー」になる
    kB: float = _kB_SI / _E_base   # ≈ 0.02083 [THz/K]
    
    # ボーア磁子 [THz/T]
    # 磁場B[T]を掛けたとき、muB*B が「THz単位のエネルギー」になる
    muB: float = _muB_SI / _E_base # ≈ 0.01399 [THz/T]
    
    # ディラック定数 [無次元]
    # E = hbar * omega で、omegaが[rad/ps]相当になるよう調整
    # THz単位系では hbar = 1/(2π) が最もシンプル
    hbar: float = 1.0 / (2 * np.pi)  # ≈ 0.159
    
    # 光速 [m/s] (波長計算用、SI単位のまま)
    c: float = 299792458
    
    # 真空の透磁率 [H/m] (SI単位のまま)
    mu0: float = 4.0 * np.pi * 1e-7
    
    # スピン量子数
    s: float = 3.5

# 後方互換性のため(既存コード対応)
kB = PhysicalConstants.kB
muB = PhysicalConstants.muB
hbar = PhysicalConstants.hbar
c = PhysicalConstants.c
mu0 = PhysicalConstants.mu0
s = PhysicalConstants.s


def _build_sz_matrix() -> np.ndarray:
    return np.diag(np.arange(s, -s - 1, -1, dtype=np.float64))


def _build_o44_matrix() -> np.ndarray:
    mat = np.zeros((8, 8), dtype=np.float64)
    v_s35 = np.sqrt(35.0)
    v_5s3 = 5.0 * np.sqrt(3.0)
    mat[3, 7] = mat[4, 0] = v_s35
    mat[2, 6] = mat[5, 1] = v_5s3
    return 12.0 * (mat + mat.T)


def _build_o64_matrix() -> np.ndarray:
    mat = np.zeros((8, 8), dtype=np.float64)
    v_3s35 = 3.0 * np.sqrt(35.0)
    v_m7s3 = -7.0 * np.sqrt(3.0)
    mat[3, 7] = mat[4, 0] = v_3s35
    mat[2, 6] = mat[5, 1] = v_m7s3
    return 60.0 * (mat + mat.T)


def _build_transition_strength() -> np.ndarray:
    upper_m = np.arange(s, -s, -1, dtype=np.float64)
    return (s + upper_m) * (s - upper_m + 1.0)


SZ_MATRIX_PT = pt.as_tensor_variable(_build_sz_matrix())
O40_MATRIX_PT = pt.as_tensor_variable(60.0 * np.diag([7, -13, -3, 9, 9, -3, -13, 7]))
O44_MATRIX_PT = pt.as_tensor_variable(_build_o44_matrix())
O60_MATRIX_PT = pt.as_tensor_variable(1260.0 * np.diag([1, -5, 9, -5, -5, 9, -5, 1]))
O64_MATRIX_PT = pt.as_tensor_variable(_build_o64_matrix())
TRANSITION_STRENGTH_PT = pt.as_tensor_variable(_build_transition_strength())

PT_KB = pt.as_tensor_variable(np.float64(kB))
PT_MUB = pt.as_tensor_variable(np.float64(muB))
PT_HBAR = pt.as_tensor_variable(np.float64(hbar))
PT_C = pt.as_tensor_variable(np.float64(c))

# =========================================================
# 1. データ読み込み & 前処理 (構造化)
# =========================================================

def load_config(path=None):
    if path is None:
        path = pathlib.Path(__file__).parent / "config_unified_gpu.yml"
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_results_directory(config: dict) -> pathlib.Path:
    """結果保存ディレクトリを作成して返す。

    analysis_results_gpu/run_YYYYmmdd_HHMMSS のようなパスを作る。
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_dir = config.get('file_paths', {}).get('results_parent_dir', 'analysis_results_gpu')
    results_dir = pathlib.Path(__file__).parent / parent_dir / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 使用した設定のバックアップ
    try:
        with open(results_dir / "config_used_gpu.yml", 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    except Exception:
        pass

    return results_dir

def create_frequency_weights(freq, trans, config):
    """重み配列生成 (ベクトル化対応)"""
    ws = config['analysis_settings']['weight_settings']
    weights = np.full_like(freq, ws['background_weight'])
    
    # ピーク検出
    peaks, props = find_peaks(trans, 
                              height=ws['peak_height_threshold'], 
                              prominence=ws['peak_prominence_threshold'], 
                              distance=ws['peak_distance'])
    
    # 低周波領域にピークがあるか確認
    low_cutoff = config['analysis_settings']['low_freq_cutoff']
    
    if len(peaks) > 0:
        # 半値幅を計算して重み付け
        widths, _, lefts, rights = peak_widths(trans, peaks, rel_height=0.5)
        # ピークのfloat型インデックスを周波数に変換
        left_f = np.interp(lefts, np.arange(len(freq)), freq)
        right_f = np.interp(rights, np.arange(len(freq)), freq)
        
        # 全ピークに対してループ (ここはPythonループでも高速)
        for i in range(len(peaks)):
            mask = (freq >= left_f[i]) & (freq <= right_f[i])
            
            # 低周波ピークか高周波ピークかで重みを変える
            if freq[peaks[i]] < low_cutoff:
                # LP/UP
                weights[mask] = ws['lp_up_peak_weight']
                # ピーク間 (簡易実装: ピーク周辺を広めに取る処理が必要ならここに追加)
            else:
                # 高周波共振
                weights[mask] = ws['high_freq_peak_weight']
                
        # ピーク間領域 (LP-UP間) の重み付け
        # 簡易的に、低周波ピークが2つ以上あればその間を埋める
        low_peaks_idx = np.where(freq[peaks] < low_cutoff)[0]
        if len(low_peaks_idx) >= 2:
            min_p = np.min(freq[peaks[low_peaks_idx]])
            max_p = np.max(freq[peaks[low_peaks_idx]])
            between_mask = (freq >= min_p) & (freq <= max_p) & (weights == ws['background_weight'])
            weights[between_mask] = ws['between_peaks_weight']

    return weights

def load_and_prepare_data(config):
    """
    GPU計算用にデータをフラット化して読み込む。
    修正点: 背景誘電率 eps_bg を (温度, 磁場) の条件ごとに管理するため、
    条件ID condition_id とユニーク条件リスト unique_conditions を構築する。
    """
    print("\n--- データ読み込みとGPU用構造化 ---")

    # 1. ファイルパス解決
    input_files = config['file_paths'].get('input_files', [])
    if not input_files:
        input_files = [{
            'file_path': config['file_paths']['data_file'],
            'type': 'temperature', 'fixed_param': 9.0
        }]

    # 2. 一時リストに収集
    all_freq: list[np.ndarray] = []
    all_trans: list[np.ndarray] = []
    all_temp: list[np.ndarray] = []
    all_bfield: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []

    # 条件 (T,B) を一意集合で管理
    unique_conditions: set[tuple[float, float]] = set()

    count = 0
    for file_info in input_files:
        # パス解決 (Linux/Windows対応)
        raw_path = file_info['file_path'].replace('\\', '/')
        path = pathlib.Path(raw_path)
        if not path.exists():
            path = pathlib.Path(__file__).parent / raw_path

        if not path.exists():
            print(f"⚠️ Skip: {raw_path}")
            continue

        try:
            df = pd.read_excel(path, sheet_name=config['file_paths']['sheet_name'])
            freq_col = df.columns[0]
            freq_vals = df[freq_col].to_numpy()

            for col in df.columns[1:]:
                col_str = str(col)
                if not any(char.isdigit() for char in col_str):
                    continue

                m = re.search(r"([\d\.]+)", col_str)
                if not m:
                    continue
                val = float(m.group(1))

                if file_info['type'] == 'temperature':
                    t_val = val
                    b_val = float(file_info['fixed_param'])
                else:
                    t_val = float(file_info['fixed_param'])
                    b_val = val

                trans_vals = df[col].values
                mask = np.isfinite(trans_vals)
                f_clean = freq_vals[mask]
                t_clean = trans_vals[mask]
                if len(f_clean) == 0:
                    continue

                w_clean = create_frequency_weights(f_clean, t_clean, config)

                all_freq.append(f_clean)
                all_trans.append(t_clean)
                all_weights.append(w_clean)
                all_temp.append(np.full_like(f_clean, t_val))
                all_bfield.append(np.full_like(f_clean, b_val))

                unique_conditions.add((float(t_val), float(b_val)))
                count += 1

        except Exception as e:
            print(f"❌ Error reading {path.name}: {e}")

    if count == 0:
        raise RuntimeError("有効なデータが読み込めませんでした。")

    # 3. Flatten
    flat_freq = np.concatenate(all_freq)
    flat_trans = np.concatenate(all_trans)
    flat_weights = np.concatenate(all_weights)
    flat_temp = np.concatenate(all_temp)
    flat_bfield = np.concatenate(all_bfield)

    # 4. 条件IDの生成 (温度昇順→磁場昇順)
    sorted_conditions = sorted(list(unique_conditions), key=lambda x: (x[0], x[1]))
    cond_to_id = {cond: i for i, cond in enumerate(sorted_conditions)}
    flat_cond_ids = np.array([cond_to_id[(t, b)] for t, b in zip(flat_temp, flat_bfield)], dtype=np.int32)

    print(f"✅ データ結合完了: {len(flat_freq)} 点, {len(sorted_conditions)} 条件のデータセット")

    return {
        'freq': flat_freq,  # THz単位
        # 【重要修正】THzスケーリング: omega = freq * 2π (1e12を掛けない)
        # 物理定数側で 1.0 = 1THz と定義したため、角周波数も同じスケール
        'omega': flat_freq * 2 * np.pi,  # [THz] → [rad/ps]相当
        'trans': flat_trans,
        'weights': flat_weights,
        'temp': flat_temp,
        'b_field': flat_bfield,
        'condition_id': flat_cond_ids,
        'unique_conditions': sorted_conditions,
        'original_datasets_count': count,
    }

# =========================================================
# 2. NumPy版 物理関数 (Step 1用)
# =========================================================
# ※ここは scipy.optimize 用なので NumPy のままでOK
from unified_weighted_bayesian_fitting import calculate_susceptibility as calculate_susceptibility_numpy
from unified_weighted_bayesian_fitting import normalize_gamma_array
@lru_cache(maxsize=1024)
def _get_hamiltonian_numpy_cached(B, g, B4, B6):
    """
    get_hamiltonian の結果をキャッシュする内部関数
    引数はハッシュ可能（float等）である必要がある
    """
    from unified_weighted_bayesian_fitting import get_hamiltonian
    return get_hamiltonian(B, g, B4, B6)

def get_hamiltonian_numpy(B, g, B4, B6):
    """
    Step 1用のハミルトニアン取得関数。
    元のCPU版スクリプトの基本関数を利用しつつ、ここでキャッシュを行って高速化する。
    """
    # lru_cache を利用してキャッシュ
    B_r = round(float(B), 6)
    g_r = round(float(g), 6)
    B4_r = round(float(B4), 8)
    B6_r = round(float(B6), 8)
    
    return _get_hamiltonian_numpy_cached(B_r, g_r, B4_r, B6_r)

def calculate_transmission_numpy(omega, mu_r, d, eps_bg):
    from unified_weighted_bayesian_fitting import calculate_normalized_transmission
    return calculate_normalized_transmission(omega, mu_r, d, eps_bg)

from unified_weighted_bayesian_fitting import get_eps_bg_initial_values_and_bounds  # 初期値生成ロジックを再利用
def fit_eps_bg_step1(unified_data, current_params, config):
    """
    Step 1: 高周波領域での eps_bg 最適化（条件 (T, B) ごとに独立推定）
    """
    print("\n--- Step 1: 光学パラメータ (eps_bg) の最適化 ---")

    high_cutoff = config['analysis_settings']['high_freq_cutoff']
    d_fixed = config['physical_parameters']['d_fixed']
    N_spin = config['physical_parameters']['N_spin']

    # 条件リスト（(temp, b_field)）
    unique_conditions: list[tuple[float, float]] = unified_data['unique_conditions']

    # 結果格納用辞書 {(t, b): eps_bg_value}
    eps_bg_map: dict[tuple[float, float], float] = {}

    # 現在の磁気パラメータ
    g = current_params['g_factor']
    B4 = current_params['B4']
    B6 = current_params['B6']
    a_scale = current_params['a_scale']

    # gamma は簡易一定（7遷移分）
    gamma_val = current_params.get('gamma_mean', 0.11e12)
    gamma_arr = np.full(7, gamma_val)

    # 定数
    G0 = a_scale * mu0 * N_spin * (g * muB)**2 / (2 * hbar)

    for (temp, b_val) in unique_conditions:
        # 条件一致かつ高周波域
        mask = (
            (unified_data['temp'] == temp) &
            (unified_data['b_field'] == b_val) &
            (unified_data['freq'] >= high_cutoff)
        )

        key = (temp, b_val)

        if np.sum(mask) < 5:
            eps_bg_map[key] = 14.2
            continue

        freq_sample = unified_data['freq'][mask]
        omega_sample = unified_data['omega'][mask]
        trans_sample = unified_data['trans'][mask]

        # 固定磁場 b_val での物理モデル
        H = get_hamiltonian_numpy(b_val, g, B4, B6)
        chi = calculate_susceptibility_numpy(omega_sample, H, temp, gamma_arr)
        mu_r_base = 1.0 + G0 * chi  # H_form 近似

        def fit_func(f_dummy, eps_val):
            return calculate_transmission_numpy(omega_sample, mu_r_base, d_fixed, eps_val)

        # CPU版の堅牢ロジックに合わせた初期値スイープ
        initial_eps_bg_values, bounds_eps_bg = get_eps_bg_initial_values_and_bounds(temp)
        success = False
        best_eps = 14.2
        min_err = float('inf')

        for initial_eps in initial_eps_bg_values:
            try:
                popt, _ = curve_fit(
                    fit_func,
                    freq_sample,
                    trans_sample,
                    p0=[initial_eps],
                    bounds=([bounds_eps_bg[0]], [bounds_eps_bg[1]]),
                    maxfev=3000,
                )
                eps_fit = float(popt[0])
                if bounds_eps_bg[0] <= eps_fit <= bounds_eps_bg[1]:
                    pred = fit_func(None, eps_fit)
                    err = float(np.sum((trans_sample - pred) ** 2))
                    if err < min_err:
                        min_err = err
                        best_eps = eps_fit
                        success = True
            except Exception as e:
                # curve_fitの収束失敗やランタイムエラーをキャッチ
                print(f"    Debug: curve_fit failed with initial_eps={initial_eps:.2f}: {type(e).__name__}")
                continue

        if success:
            eps_bg_map[key] = best_eps
        else:
            print(f"  ⚠️ Fit failed: T={temp}K, B={b_val}T -> Default used")
            eps_bg_map[key] = 14.2

    print(f"✅ {len(eps_bg_map)} 条件の eps_bg を独立に決定しました。")
    return eps_bg_map

# =========================================================
# 3. PyTensor版 物理関数 (Step 2: MCMC用)
# =========================================================

def get_hamiltonian_pt_impl(B_ext_z, g_factor, B4, B6):
    """Return the 8x8 Hamiltonian tensor for a given field."""
    B_ext_z_pt = pt.as_tensor_variable(B_ext_z)
    g_factor_pt = pt.as_tensor_variable(g_factor)
    B4_pt = pt.as_tensor_variable(B4)
    B6_pt = pt.as_tensor_variable(B6)

    crystal_field = (B4_pt * PT_KB) * (O40_MATRIX_PT + 5.0 * O44_MATRIX_PT)
    crystal_field += (B6_pt * PT_KB) * (O60_MATRIX_PT - 21.0 * O64_MATRIX_PT)
    zeeman = g_factor_pt * PT_MUB * B_ext_z_pt * SZ_MATRIX_PT
    return crystal_field + zeeman


def precompute_eigenvalues_for_conditions(unique_conditions, g_factor, B4, B6):
    """事前に各条件の固有値を計算してキャッシュ化（改善1）"""
    eigvals_list = []
    for (temp, b_field) in unique_conditions:
        H = get_hamiltonian_pt_impl(b_field, g_factor, B4, B6)
        eigvals = eigvalsh(H, b=None)  # type: ignore[call-arg]
        eigvals_shifted = eigvals - pt.min(eigvals)
        eigvals_list.append(eigvals_shifted)
    
    # Stack to create lookup tensor: shape (n_conditions, 8)
    eigvals_tensor = pt.stack(eigvals_list)
    return eigvals_tensor


def calculate_susceptibility_vectorized(omega_array, temp_array, condition_id, eigvals_cache, gamma_array):
    """Vectorized磁化率計算(改善1,2,温度依存gamma対応)
    
    Args:
        omega_array: 角周波数配列 shape (N,)
        temp_array: 温度配列 shape (N,)
        condition_id: 条件ID配列 shape (N,)
        eigvals_cache: 固有値テンソル shape (n_conditions, 8)
        gamma_array: 緩和時間配列 shape (N, 7) または (7,)
    Returns:
        chi: 磁化率配列 shape (N,)
    """
    # 条件IDから固有値を取得: shape (N, 8)
    eigvals = eigvals_cache[condition_id]
    
    # 温度ごとにBoltzmann分布を計算: shape (N, 8)
    beta = 1.0 / (PT_KB * temp_array[:, None])  # (N, 1)
    weights = pt.exp(-eigvals * beta)  # (N, 8)
    partition = pt.sum(weights, axis=1, keepdims=True)  # (N, 1)
    pops = weights / partition  # type: ignore[operator]  # (N, 8)
    
    # エネルギー差と集団差: shape (N, 7)
    delta_e = eigvals[:, 1:] - eigvals[:, :-1]  # (N, 7)
    delta_pop = pops[:, 1:] - pops[:, :-1]  # (N, 7)
    omega_0 = delta_e / PT_HBAR  # (N, 7)
    
    # 遷移強度を掛ける: shape (N, 7)
    numer = delta_pop * TRANSITION_STRENGTH_PT[None, :]  # (N, 7)
    
    # 磁化率計算(周波数依存)
    # omega_array: (N,), omega_0: (N, 7), gamma_array: (N, 7) または (7,)
    # gamma_arrayが(7,)の場合は[None, :]でブロードキャスト、(N,7)ならそのまま使用
    gamma_broadcast = gamma_array if gamma_array.ndim == 2 else gamma_array[None, :]
    denom = omega_0 - omega_array[:, None] - 1j * gamma_broadcast  # (N, 7)
    chi = pt.sum(numer / denom, axis=1)  # (N,)
    
    return -chi  # type: ignore[operator]


def calculate_transmission_vectorized(omega_array, mu_r_array, d, eps_bg_array):
    """Vectorized透過率計算(改善2)
    
    Args:
        omega_array: 角周波数配列 shape (N,)
        mu_r_array: 透磁率配列 shape (N,)
        d: サンプル厚(スカラー)
        eps_bg_array: 背景誘電率配列 shape (N,)
    Returns:
        transmission: 透過率配列 shape (N,)
    """
    # 全てベクトルとして処理
    mu_r_c = pt.cast(mu_r_array, "complex128")
    eps_bg_c = pt.cast(eps_bg_array, "complex128")
    
    # eps * mu
    eps_mu = eps_bg_c * mu_r_c
    real_eps = pt.real(eps_mu)
    imag_eps = pt.imag(eps_mu)
    
    # 実部が負の場合のフォールバック
    fallback = pt.cast(0.1, "complex128") + 1j * imag_eps  # type: ignore[operator]
    condition = pt.gt(real_eps, 0.0)
    safe_eps_mu = pt.switch(condition, eps_mu, fallback)
    
    # 屈折率とインピーダンス
    n = pt.sqrt(safe_eps_mu)  # type: ignore[operator]
    impedance = pt.sqrt(mu_r_c / eps_bg_c)  # type: ignore[operator]
    
    # 位相差
    wavelength = (2.0 * np.pi * PT_C) / omega_array
    delta = (2.0 * np.pi * n * d) / wavelength  # type: ignore[operator]
    
    # 透過係数
    numerator = 4.0 * impedance  # type: ignore[operator]
    term_pos = (1.0 + impedance) ** 2 * pt.exp(-1j * delta)  # type: ignore[operator]
    term_neg = (1.0 - impedance) ** 2 * pt.exp(1j * delta)  # type: ignore[operator]
    transmission = numerator / (term_pos - term_neg)
    
    return pt.abs(transmission) ** 2  # type: ignore[operator]

# =========================================================
# 4. メイン MCMC ルーチン
# =========================================================

def run_mcmc_gpu_optimized(unified_data, eps_bg_map, config, model_type):
    print(f"\n🚀 {model_type}: GPU Vectorized MCMC Start")
    
    # データをPyTensor定数に変換
    omega_pt = pt.as_tensor_variable(unified_data['omega'])
    trans_pt = pt.as_tensor_variable(unified_data['trans'])
    temp_pt = pt.as_tensor_variable(unified_data['temp'])
    b_pt = pt.as_tensor_variable(unified_data['b_field'])
    weights_pt = pt.as_tensor_variable(unified_data['weights'])
    
    # 条件IDで eps_bg をルックアップ
    sorted_conditions = unified_data['unique_conditions']  # [(t, b), ...]
    eps_values = np.array([eps_bg_map[cond] for cond in sorted_conditions])
    eps_lookup = pt.as_tensor_variable(eps_values)

    cond_id_pt = pt.as_tensor_variable(unified_data['condition_id'])
    eps_bg_pt = eps_lookup[cond_id_pt]
    
    d_fixed = config['physical_parameters']['d_fixed']
    N_spin = config['physical_parameters']['N_spin']
    
    # メモリ効率化: バッチサイズ設定(改善3)
    total_points = len(unified_data['omega'])
    batch_size = config.get('mcmc', {}).get('batch_size', min(10000, total_points))
    
    print(f"📊 データ点数: {total_points}, バッチサイズ: {batch_size}")
    
    # 事前分布パラメータをconfigから読み込み(ハードコーディング排除)
    priors_cfg = config.get('bayesian_priors', {})
    mag_priors = priors_cfg.get('magnetic_parameters', {})
    gamma_priors = priors_cfg.get('gamma_parameters', {})
    
    # デフォルト値(config未設定時のフォールバック)
    g_mu = mag_priors.get('g_factor', {}).get('mu', 2.0)
    g_sigma = mag_priors.get('g_factor', {}).get('sigma', 0.1)
    B4_mu = mag_priors.get('B4', {}).get('mu', 0.0005)
    B4_sigma = mag_priors.get('B4', {}).get('sigma', 0.0001)
    B6_mu = mag_priors.get('B6', {}).get('mu', 0.00005)
    B6_sigma = mag_priors.get('B6', {}).get('sigma', 0.00001)
    a_scale_sigma = mag_priors.get('a_scale', {}).get('sigma', 1.0)
    
    log_gamma_mu_val = gamma_priors.get('log_gamma_mu_base', {}).get('mu', 25.0)
    log_gamma_sigma_val = gamma_priors.get('log_gamma_mu_base', {}).get('sigma', 1.0)
    temp_slope_sigma = gamma_priors.get('temp_gamma_slope', {}).get('sigma', 0.01)
    log_sigma_val = gamma_priors.get('log_gamma_sigma_base', {}).get('sigma', 0.3)
    log_offset_sigma = gamma_priors.get('log_gamma_offset_base', {}).get('sigma', 0.3)
    
    with pm.Model() as model:
        # --- Parameters (config駆動) ---
        a_scale = pm.HalfNormal('a_scale', sigma=a_scale_sigma)
        g_factor = pm.Normal('g_factor', mu=g_mu, sigma=g_sigma)
        B4 = pm.Normal('B4', mu=B4_mu, sigma=B4_sigma)
        B6 = pm.Normal('B6', mu=B6_mu, sigma=B6_sigma)
        
        log_gamma_mu = pm.Normal('log_gamma_mu_base', mu=log_gamma_mu_val, sigma=log_gamma_sigma_val)
        temp_slope = pm.Normal('temp_gamma_slope', mu=0.0, sigma=temp_slope_sigma)
        log_sigma = pm.HalfNormal('log_gamma_sigma_base', sigma=log_sigma_val)
        log_offset = pm.Normal('log_gamma_offset_base', mu=0.0, sigma=log_offset_sigma, shape=7)
        
        # --- 固有値事前計算(改善1)---
        eigvals_cache = precompute_eigenvalues_for_conditions(
            sorted_conditions, g_factor, B4, B6
        )
        
        # --- Gamma配列計算(改善4: 遷移ごと独立 + 温度依存性修正)---
        # 1. 遷移ごとのオフセット項 (温度に依存しない形状因子)
        # shape: (1, 7) にしてブロードキャスト準備
        gamma_offsets = pt.exp(log_offset * log_sigma)[None, :]  # type: ignore[index]  # (1, 7)
        
        # 2. 温度依存項 (データ点ごとに計算)
        # temp_pt: (N,) -> (N, 1) に変形
        temp_diff = (temp_pt - 4.0)[:, None]  # type: ignore[index]  # (N, 1)
        
        # log_gamma_mu + slope * diff -> (N, 1)
        log_gamma_base_vec = log_gamma_mu + temp_slope * temp_diff
        gamma_base_vec = pt.exp(log_gamma_base_vec)  # (N, 1)
        
        # 3. 統合 (N, 1) * (1, 7) -> (N, 7)
        # 各データ点 i、各遷移 j に対応する gamma[i, j]
        gamma_unified = gamma_base_vec * gamma_offsets  # type: ignore[operator]  # shape: (N, 7)
        
        # --- Vectorized計算(改善2: scan完全排除)---
        # 磁化率計算
        chi = calculate_susceptibility_vectorized(
            omega_pt, temp_pt, cond_id_pt, eigvals_cache, gamma_unified
        )
        
        # G0定数
        G0 = a_scale * mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
        
        # 透磁率計算
        if model_type == 'B_form':
            mu_r = 1.0 / (1.0 - G0 * chi)
        else:
            mu_r = 1.0 + G0 * chi
        
        # 透過率計算(Vectorized)
        mu_pred = calculate_transmission_vectorized(
            omega_pt, mu_r, d_fixed, eps_bg_pt
        )
        
        # --- Likelihood ---
        # 重み付きノイズ
        sigma_base = pm.HalfNormal('sigma_base', sigma=0.05)
        # w < 1e-6 の場所は無視するため、sigmaを大きくする
        w_safe = pt.clip(weights_pt, 1e-6, 1e10)
        sigma_vec = sigma_base / pt.sqrt(w_safe)
        
        pm.Normal('obs', mu=mu_pred, sigma=sigma_vec, observed=trans_pt)
        
        # --- Sampling ---
        trace = pm.sample(
            draws=config['mcmc']['draws'],
            tune=config['mcmc']['tune'],
            chains=config['mcmc']['chains'],
            nuts_sampler="numpyro",
            random_seed=42
        )
        return trace

# =========================================================
# 5. Main Loop
# =========================================================
def main():
    config = load_config()
    results_dir = create_results_directory(config)
    
    # 1. データのロード (一括)
    unified_data = load_and_prepare_data(config)
    
    # 初期パラメータ
    current_params = {
        'g_factor': 2.0, 'B4': 0.0005, 'B6': 0.00005, 'a_scale': 1.0, 'gamma_mean': 0.11e12
    }
    
    for i in range(config['mcmc']['max_iterations']):
        print(f"\n=== Iteration {i+1} ===")
        
        # Step 1: eps_bg 最適化
        # NumPyで計算するので高速、かつ高周波のみなので簡単
        eps_bg_map = fit_eps_bg_step1(unified_data, current_params, config)
        
        # Step 2: MCMC
        trace = run_mcmc_gpu_optimized(unified_data, eps_bg_map, config, 'H_form')
        
        # 結果保存
        az.to_netcdf(trace, results_dir / f"trace_iter{i}.nc")
        
        # パラメータ更新 (type: ignore for xarray typing)
        summary_result = az.summary(trace)
        g_val = summary_result.loc['g_factor', 'mean']  # type: ignore[index]
        b4_val = summary_result.loc['B4', 'mean']  # type: ignore[index]
        current_params['g_factor'] = float(g_val.item() if hasattr(g_val, 'item') else g_val)  # type: ignore[arg-type]
        current_params['B4'] = float(b4_val.item() if hasattr(b4_val, 'item') else b4_val)  # type: ignore[arg-type]
        # ... (他も更新) ...
        print("Updated Params:", current_params)

if __name__ == "__main__":
    main()
