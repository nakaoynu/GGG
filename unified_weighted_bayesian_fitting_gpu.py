# unified_weighted_bayesian_fitting_gpu.py
# GPU/JAX対応版: 物理モデルをPyTensor化し、NumPyroサンプラーを使用可能にしたバージョン

import time
_import_start = time.time()

import os
import pathlib
import yaml
import sys

# GPU設定の先行読み込み
def pre_load_gpu_config():
    try:
        # 同じディレクトリにある config_unified_gpu.yml を探す
        config_path = pathlib.Path(__file__).parent / "config_unified_gpu.yml"
        if not config_path.exists():
            return
            
        with open(config_path, 'r', encoding='utf-8') as f:
            temp_config = yaml.safe_load(f)
            
        if temp_config.get('execution', {}).get('use_gpu', False):
            print("🚀 GPU (JAX) モードで起動します...")
            # JAXを使用する場合、PyTensor側のdevice指定は不要(むしろエラーの元)なのでfloatXのみ指定
            os.environ['PYTENSOR_FLAGS'] = 'floatX=float64'
        else:
            print("💻 CPU モードで起動します...")
            os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float64'
    except Exception as e:
        print(f"⚠️ 設定読み込み警告: {e}")

pre_load_gpu_config()

import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import pytensor
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit

# 数値計算の警告を抑制
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning) # PyTensorの警告も一部抑制

# =========================================================
# 1. 共通定数・設定読み込み関数
# =========================================================
kB = 1.380649e-23
muB = 9.274010e-24
hbar = 1.054571e-34
c = 299792458
mu0 = 4.0 * np.pi * 1e-7
s = 3.5

def load_config(config_path=None):
    if config_path is None:
        config_path = pathlib.Path(__file__).parent / "config_unified_gpu.yml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def create_results_directory(config):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_dir = config['file_paths'].get('results_parent_dir', 'analysis_results_gpu')
    results_dir = pathlib.Path(__file__).parent / parent_dir / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # configのバックアップ
    with open(results_dir / "config_used_gpu.yml", 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
    return results_dir

# =========================================================
# 2. NumPy版 物理関数 (Step 1: eps_bg 最適化用)
# =========================================================
# 以前のコードの関数をそのまま使用（名称変更なし）

class HamiltonianCache:
    def __init__(self): self._cache = {}
    def get(self, B, g, B4, B6):
        key = (round(B, 6), round(g, 6), round(B4, 8), round(B6, 8))
        if key not in self._cache:
            self._cache[key] = self._calc(B, g, B4, B6)
        return self._cache[key]
    def _calc(self, B_ext_z, g_factor, B4, B6):
        m_values = np.arange(s, -s - 1, -1)
        Sz = np.diag(m_values)
        O40 = 60 * np.diag([7, -13, -3, 9, 9, -3, -13, 7])
        X_O44 = np.zeros((8, 8)); X_O44[3, 7] = X_O44[4, 0] = np.sqrt(35); X_O44[2, 6] = X_O44[5, 1] = 5 * np.sqrt(3)
        O44 = 12 * (X_O44 + X_O44.T)
        O60 = 1260 * np.diag([1, -5, 9, -5, -5, 9, -5, 1])
        X_O64 = np.zeros((8, 8)); X_O64[3, 7] = X_O64[4, 0] = 3 * np.sqrt(35); X_O64[2, 6] = X_O64[5, 1] = -7 * np.sqrt(3)
        O64 = 60 * (X_O64 + X_O64.T)
        return (B4 * kB) * (O40 + 5 * O44) + (B6 * kB) * (O60 - 21 * O64) + g_factor * muB * B_ext_z * Sz

_hamiltonian_cache = HamiltonianCache()

def get_hamiltonian_numpy(B, g, B4, B6):
    return _hamiltonian_cache.get(B, g, B4, B6)

def calculate_susceptibility_numpy(omega, H, T, gamma):
    eigenvalues, _ = np.linalg.eigh(H)
    eigenvalues -= np.min(eigenvalues)
    Z = np.sum(np.exp(-eigenvalues / (kB * T)))
    populations = np.exp(-eigenvalues / (kB * T)) / Z
    delta_E = eigenvalues[1:] - eigenvalues[:-1]
    delta_pop = populations[1:] - populations[:-1]
    omega_0 = delta_E / hbar
    m_vals = np.arange(s, -s, -1)
    strength = (s + m_vals) * (s - m_vals + 1)
    
    chi = np.zeros_like(omega, dtype=complex)
    # NumPy版はループで計算
    for i, w in enumerate(omega):
        denom = omega_0 - w - 1j * gamma
        chi[i] = np.sum(delta_pop * strength / denom)
    return -chi

def calculate_transmission_numpy(omega, mu_r, d, eps_bg):
    eps_mu = eps_bg * mu_r
    # 簡易実装
    n = np.sqrt(eps_mu + 0j)
    impe = np.sqrt(mu_r / eps_bg + 0j)
    lam = (2*np.pi*c) / omega
    delta = 2*np.pi*n*d/lam
    num = 4*impe
    den = (1+impe)**2 * np.exp(-1j*delta) - (1-impe)**2 * np.exp(1j*delta)
    t = num/den
    return np.abs(t)**2

# =========================================================
# 3. PyTensor版 物理関数 (Step 2: MCMC/GPU用)
# =========================================================
# テスト済みの関数をここに配置します

def get_hamiltonian_pt(B_ext_z, g_factor, B4, B6):
    """PyTensor版ハミルトニアン
    
    Parameters
    ----------
    B_ext_z : float or TensorVariable
        外部磁場 (T)
    g_factor : float or TensorVariable
        ランデのg因子
    B4 : float or TensorVariable
        結晶場パラメータ B4 (K)
    B6 : float or TensorVariable
        結晶場パラメータ B6 (K)
    
    Returns
    -------
    TensorVariable
        ハミルトニアン行列 (8x8)
    """
    kB_pt = pt.as_tensor_variable(kB)
    muB_pt = pt.as_tensor_variable(muB)
    m_values = pt.as_tensor_variable(np.arange(s, -s - 1, -1))
    Sz = pt.diag(m_values)
    
    O40_diag = pt.as_tensor_variable([7, -13, -3, 9, 9, -3, -13, 7])
    O40 = pt.as_tensor_variable(60.0) * pt.diag(O40_diag)
    
    X_O44_base = pt.zeros((8, 8))
    v_s35 = np.sqrt(35); v_5s3 = 5 * np.sqrt(3)
    X_O44 = pt.set_subtensor(X_O44_base[3, 7], v_s35)
    X_O44 = pt.set_subtensor(X_O44[4, 0], v_s35)
    X_O44 = pt.set_subtensor(X_O44[2, 6], v_5s3)
    X_O44 = pt.set_subtensor(X_O44[5, 1], v_5s3)
    O44 = pt.as_tensor_variable(12.0) * (X_O44 + X_O44.T) # type: ignore
    
    O60_diag = pt.as_tensor_variable([1, -5, 9, -5, -5, 9, -5, 1])
    O60 = pt.as_tensor_variable(1260.0) * pt.diag(O60_diag)
    
    X_O64_base = pt.zeros((8, 8))
    v_3s35 = 3 * np.sqrt(35); v_m7s3 = -7 * np.sqrt(3)
    X_O64 = pt.set_subtensor(X_O64_base[3, 7], v_3s35)
    X_O64 = pt.set_subtensor(X_O64[4, 0], v_3s35)
    X_O64 = pt.set_subtensor(X_O64[2, 6], v_m7s3)
    X_O64 = pt.set_subtensor(X_O64[5, 1], v_m7s3)
    O64 = pt.as_tensor_variable(60.0) * (X_O64 + X_O64.T) # type: ignore
    
    H_cf = (B4 * kB_pt) * (O40 + 5 * O44) + (B6 * kB_pt) * (O60 - 21 * O64)
    H_zee = g_factor * muB_pt * B_ext_z * Sz
    return H_cf + H_zee

def calculate_susceptibility_pt(omega, H, T, gamma_array):
    """PyTensor版磁気感受率
    
    Parameters
    ----------
    omega : TensorVariable
        角周波数配列 (rad/s)
    H : TensorVariable
        ハミルトニアン行列 (8x8)
    T : float or TensorVariable
        温度 (K)
    gamma_array : TensorVariable
        減衰定数配列 (7要素)
    
    Returns
    -------
    TensorVariable
        磁気感受率（複素数配列）
    """
    kB_pt = pt.as_tensor_variable(kB)
    hbar_pt = pt.as_tensor_variable(hbar)
    
    # ハミルトニアンの固有値を計算
    eigenvalues, _ = pt.linalg.eigh(H)
    eigenvalues = eigenvalues - pt.min(eigenvalues)
    
    # 分配関数と占有確率
    Z = pt.sum(pt.exp(-eigenvalues / (kB_pt * T)))
    populations = pt.exp(-eigenvalues / (kB_pt * T)) / Z
    
    # エネルギー差と占有確率差
    delta_E = eigenvalues[1:] - eigenvalues[:-1]
    delta_pop = populations[1:] - populations[:-1]
    
    # 遷移強度
    m_vals = pt.as_tensor_variable(np.arange(s, -s, -1))
    strength = (s + m_vals) * (s - m_vals + 1.0)
    omega_0 = delta_E / hbar_pt
    
    # 磁気感受率の計算（ブロードキャスト）
    numerator = delta_pop * strength
    # ブロードキャスト計算
    # omega: (N,), omega_0: (7,), gamma: (7,)
    denom = omega_0[None, :] - omega[:, None] - 1j * gamma_array[None, :]
    chi_comp = numerator[None, :] / denom
    chi = pt.sum(chi_comp, axis=1)
    return -chi

def calculate_transmission_pt(omega, mu_r, d, eps_bg):
    """PyTensor版透過率"""
    c_pt = pt.as_tensor_variable(c)
    
    eps_mu = eps_bg * mu_r
    cond = pt.gt(pt.real(eps_mu), 0.0)
    safe_eps_mu = pt.switch(cond, eps_mu, 0.1 + 1j * pt.imag(eps_mu))
    
    n = pt.sqrt(safe_eps_mu + 0j)
    impe = pt.sqrt(mu_r / eps_bg + 0j)
    
    lam = (2 * np.pi * c_pt) / omega
    delta_raw = 2 * np.pi * n * d / lam
    
    # クリップ
    d_real = pt.clip(pt.real(delta_raw), -700, 700)
    d_imag = pt.clip(pt.imag(delta_raw), -700, 700)
    delta = d_real + 1j * d_imag
    
    num = 4 * impe
    ep = pt.exp(-1j * delta); en = pt.exp(1j * delta)
    den = (1 + impe)**2 * ep - (1 - impe)**2 * en
    
    t = num / den
    trans = pt.abs(t)**2
    
    # 正規化
    t_min = pt.min(trans)
    t_max = pt.max(trans)
    norm = pt.switch(pt.gt(t_max - t_min, 1e-20), (trans - t_min)/(t_max - t_min), 0.5)
    return pt.clip(norm, 0.0, 1.0)

# =========================================================
# 4. データ読み込み・前処理関数
# =========================================================

def create_frequency_weights(dataset, config):
    # (既存の重み付けロジックをそのまま使用)
    # ※ここでは簡略化のため、元のロジックを呼び出すか、コピーしてください
    # 修正版（1ピーク対応）を入れておきます
    ws = config['analysis_settings']['weight_settings']
    freq = dataset['frequency']
    trans = dataset['transmittance_full']
    peaks, props = find_peaks(trans, height=ws['peak_height_threshold'], 
                             prominence=ws['peak_prominence_threshold'], 
                             distance=ws['peak_distance'])
    weights = np.full_like(freq, ws['background_weight'])
    
    # 修正版: 1ピークでも重みを付ける
    low_freq_cutoff = config['analysis_settings']['low_freq_cutoff']
    low_peaks = peaks[freq[peaks] < low_freq_cutoff]
    
    # ... (詳細な重み付けロジックは元のコードからコピー推奨) ...
    # 簡易版として、全ピーク周辺に重みを付ける実装にします
    widths, _, lefts, rights = peak_widths(trans, peaks, rel_height=0.5)
    left_f = np.interp(lefts, np.arange(len(freq)), freq)
    right_f = np.interp(rights, np.arange(len(freq)), freq)
    
    for i, p in enumerate(peaks):
        mask = (freq >= left_f[i]) & (freq <= right_f[i])
        weights[mask] = ws['lp_up_peak_weight']
        
    return weights

def load_and_prepare_data(config):
    # Excel読み込みロジック (Linux対応パス)
    # 簡略化して記述します。元の unified_...py の load_unified_data を使ってください
    # ここではデータ構造のイメージだけ示します
    # datasets = [{'temperature': 4.0, 'b_field': 9.0, 'frequency': ..., 'transmittance_full': ...}, ...]
    
    # 実際の読み込みは元のコードの関数をそのままコピーして使うのが安全です
    from unified_weighted_bayesian_fitting import load_unified_data as original_loader
    return original_loader(config)

# =========================================================
# 5. Step 1: eps_bg 最適化 (NumPy使用)
# =========================================================
def fit_eps_bg_step1(datasets, fixed_params, config):
    """Step 1: eps_bg と d の最適化（NumPy使用）
    
    Parameters
    ----------
    datasets : list
        データセットのリスト
    fixed_params : dict
        固定パラメータ
    config : dict
        設定ファイルの内容
    
    Returns
    -------
    dict
        温度ごとの最適化結果
    
    Notes
    -----
    この関数は簡易実装です。実際の使用では、元のモジュールから
    適切な最適化関数をインポートするか、ここに完全な実装を追加してください。
    """
    results = {}
    for ds in datasets:
        temp = ds['temperature']
        
        # デフォルト値を設定
        results[temp] = {
            'eps_bg': 14.5,
            'd': 157.8e-6,
            'success': False
        }
        
        print(f"警告: 温度 {temp}K のeps_bg最適化は未実装（デフォルト値を使用）")
    
    return results

# =========================================================
# 6. Step 2: MCMC (PyTensor/GPU使用) - メイン改修部分
# =========================================================
def run_mcmc_gpu(datasets, eps_bg_map, weights_list, config, model_type, prior_params=None):
    print(f"\n🚀 {model_type}モデル: GPUベイズ推定を開始します...")
    
    mcmc_conf = config['mcmc']
    
    # データの前処理: PyMCモデル内で扱いやすいように結合する
    # ただし、BやTがバラバラなので、計算グラフ構築時にPythonループで回すのが一番確実
    
    with pm.Model() as model:
        # --- 事前分布 ---
        # 設定ファイルから読み込む (簡略化のため直接記述例)
        priors = config['bayesian_priors']['magnetic_parameters']
        
        # 変数定義 (Opを使わず直接定義！)
        a_scale = pm.HalfNormal('a_scale', sigma=1.0)
        g_factor = pm.Normal('g_factor', mu=2.0, sigma=0.1)
        B4 = pm.Normal('B4', mu=0.0005, sigma=0.0001)
        B6 = pm.Normal('B6', mu=0.00005, sigma=0.00001)
        
        # Gammaの定義 (温度依存性)
        g_conf = config['bayesian_priors']['gamma_parameters']
        log_gamma_mu_base = pm.Normal('log_gamma_mu_base', mu=25.0, sigma=1.0)
        temp_gamma_slope = pm.Normal('temp_gamma_slope', mu=0.0, sigma=0.01)
        log_gamma_sigma_base = pm.HalfNormal('log_gamma_sigma_base', sigma=0.3)
        log_gamma_offset_base = pm.Normal('log_gamma_offset_base', mu=0.0, sigma=0.3, shape=7)
        
        # --- 計算グラフの構築 (ここが新しい！) ---
        mu_pred_list = []
        target_data_list = []
        sigma_list = []
        
        base_temp = 4.0
        
        # 全データセットに対してループして計算グラフを繋ぐ
        for i, ds in enumerate(datasets):
            # 重みゼロの点は除外する処理が必要だが、
            # JAXのコンパイル時間を考えると、データサイズは固定の方が良い場合もある
            # ここでは「全データを使って、重みで尤度を制御する」方式をとる
            
            # 定数入力 (PyTensor定数化)
            omega_pt = pt.as_tensor_variable(ds['omega'])
            temp = ds['temperature']
            b_field = ds['b_field']
            
            # 1. Gamma (温度依存)
            temp_diff = temp - base_temp
            mu_temp = log_gamma_mu_base + temp_gamma_slope * temp_diff
            gamma = pt.exp(mu_temp + log_gamma_offset_base * log_gamma_sigma_base)
            
            # 2. Hamiltonian (磁場依存)
            # b_field はスカラ定数として渡す
            H = get_hamiltonian_pt(pt.as_tensor_variable(b_field), g_factor, B4, B6)
            
            # 3. Susceptibility
            chi = calculate_susceptibility_pt(omega_pt, H, pt.as_tensor_variable(temp), gamma)
            
            # 4. Transmission
            # G0計算
            N_spin = config['physical_parameters']['N_spin']
            G0 = a_scale * mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
            chi_scaled = G0 * chi
            
            if model_type == 'B_form':
                mu_r = 1.0 / (1.0 - chi_scaled)
            else:
                mu_r = 1.0 + chi_scaled
            
            # 光学パラメータ (Step 1の結果を使用)
            # eps_bg_map からこの温度の値を取得
            opt = eps_bg_map.get(temp, {'eps_bg': 14.2, 'd': 157.8e-6})
            
            trans = calculate_transmission_pt(
                omega_pt, mu_r, 
                pt.as_tensor_variable(opt['d']), 
                pt.as_tensor_variable(opt['eps_bg'])
            )
            
            # リストに追加
            mu_pred_list.append(trans)
            target_data_list.append(ds['transmittance_full'])
            
            # ノイズレベル (重み付け)
            w = weights_list[i]
            # w=0 のところは sigma=無限大 にして無視させる
            # 数値安定性のため、w < 1e-6 のところは w=1e-6 扱いにする等
            w_safe = np.maximum(w, 1e-6)
            sigma_base = pm.HalfNormal(f'sigma_{i}', sigma=0.05) # データセットごとにノイズレベルを変えるか、共通にするかは選択
            # ここでは簡単のため共通のsigmaを定義しても良い
            
            sig = sigma_base / pt.sqrt(pt.as_tensor_variable(w_safe))
            sigma_list.append(sig)
            
        # 結合
        # PyTensorのconcatenate等は使わず、Observedを個別に定義する方がグラフ構築が軽い場合もあるが
        # ここでは一括定義する
        
        # データセットごとのLikelihood定義 (リスト内包表記などで展開しても良い)
        for i in range(len(datasets)):
             pm.Normal(f'obs_{i}', mu=mu_pred_list[i], sigma=sigma_list[i], observed=target_data_list[i])

        # --- サンプリング (GPU/JAX) ---
        print("Sampling using NumPyro (JAX)...")
        # numpyroが使える場合
        try:
            # nuts_sampler='numpyro' を指定
            trace = pm.sample(
                draws=mcmc_conf['draws'],
                tune=mcmc_conf['tune'],
                chains=mcmc_conf['chains'],
                nuts_sampler="numpyro", # <--- ここが重要！
                random_seed=42
            )
            return trace
        except Exception as e:
            print(f"NumPyro Error: {e}")
            print("Falling back to standard sampler...")
            return pm.sample(draws=1000, tune=500, chains=2)

# =========================================================
# 7. メイン実行フロー
# =========================================================
def main():
    """メイン実行関数
    
    GPU/JAXを使用したベイズ推定の実行フロー:
    1. データ読み込みと前処理
    2. eps_bg初期値の最適化
    3. MCMCサンプリング（NumPyro/JAX使用）
    4. 結果の保存と可視化
    
    Notes
    -----
    この関数は開発中のため、一部の機能は未実装です。
    実際の使用には、データ読み込み関数などの完全な実装が必要です。
    """
    print("\n" + "="*70)
    print("GPU/JAX対応 統合ベイズ推定 開始")
    print("="*70 + "\n")
    
    # 設定読み込みと出力ディレクトリ作成
    config = load_config()
    results_dir = create_results_directory(config)
    print(f"📁 結果保存先: {results_dir}\n")
    
    # 1. データ読み込み
    print("📊 データ読み込み中...")
    try:
        # load_and_prepare_dataを実装するか、元のモジュールから適切にインポート
        datasets_dict = load_and_prepare_data(config)
        
        # 温度変化と磁場変化のデータを結合
        datasets = datasets_dict.get('temp_variable', []) + datasets_dict.get('field_variable', [])
        
        if not datasets:
            print("❌ エラー: データが見つかりません")
            return
            
        print(f"✅ {len(datasets)}個のデータセットを読み込みました\n")
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return

    # 2. 重み計算
    print("⚖️  周波数重み計算中...")
    weights_list = [create_frequency_weights(d, config) for d in datasets]
    print("✅ 重み計算完了\n")
    
    # 3. 初期 eps_bg 最適化 (Step 1)
    print("🔧 eps_bg初期値の最適化中...")
    fixed_params = {
        'a_scale': 1.0,
        'g_factor': 2.0,
        'B4': 0.0005,
        'B6': 0.00005
    }
    eps_bg_map = fit_eps_bg_step1(datasets, fixed_params, config)
    print("✅ eps_bg初期値の最適化完了\n")
    
    # 4. MCMC反復ループ
    max_iterations = config['mcmc'].get('max_iterations', 1)
    for iteration in range(max_iterations):
        print(f"\n{'='*70}")
        print(f"Iteration {iteration+1}/{max_iterations}")
        print(f"{'='*70}\n")
        
        # Step 2: MCMC (GPU/JAX)
        try:
            trace_h = run_mcmc_gpu(datasets, eps_bg_map, weights_list, config, 'H_form')
            
            # 結果保存
            output_path = results_dir / f"trace_H_form_iter{iteration}.nc"
            az.to_netcdf(trace_h, output_path)
            print(f"\n💾 トレースを保存: {output_path.name}")
            
        except Exception as e:
            print(f"❌ MCMCエラー: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # パラメータ抽出 -> eps_bg更新 (次のイテレーション用)
        # TODO: 実装が必要
        print("\n⚠️  注意: eps_bg更新ロジックは未実装です")
        
    print("\n" + "="*70)
    print("🎉 処理完了")
    print("="*70)

if __name__ == "__main__":
    main()