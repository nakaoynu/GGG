import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTensorの設定（メモリ効率改善）
import os
os.environ['PYTENSOR_FLAGS'] = 'optimizer=fast_compile,floatX=float32'

import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
try:
    import japanize_matplotlib # 日本語表示のため
except ImportError:
    print("注意: japanize_matplotlib がインストールされていません。日本語表示に影響する可能性があります。")
    print("インストール方法: pip install japanize-matplotlib")

# --- 0. プロット設定 ---
plt.rcParams['figure.dpi'] = 100

# 画像保存用ディレクトリパス
import pathlib
IMAGE_DIR = pathlib.Path(__file__).parent / "pymc_image"
IMAGE_DIR.mkdir(exist_ok=True)  # ディレクトリが存在しない場合は作成

# --- 1. 物理定数とパラメータ定義 ---
kB = 1.380649e-23
muB = 9.274010e-24
hbar = 1.054571e-34
c = 299792458
mu0 = 4.0 * np.pi * 1e-7
s = 3.5
N_spin = 24 / 1.238 * 1e27  # GGGのスピン数密度

# 定数として扱うパラメータ
d = 157.8e-6 * 0.99  # 試料厚さ（定数）

# 初期値
eps_bg_init = 13.1404
g_factor_init = 2.02
B4_init = 0.0015
B6_init = -0.00003
gamma_init = np.full(7, 0.11e12)
a_init = 1.5  # スケーリング係数の初期値

# --- 2. 物理モデル関数 ---
def get_hamiltonian(B_ext_z, B4, B6, g_factor):
    m_values = np.arange(s, -s - 1, -1)
    Sz = np.diag(m_values)
    
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
    
    H_cf = (B4 * kB) * (O40 + 5 * O44) + (B6 * kB) * (O60 - 21 * O64)
    H_zee = g_factor * muB * B_ext_z * Sz
    return H_cf + H_zee

def calculate_susceptibility(omega_array, H, T, gamma_array):
    eigenvalues, _ = np.linalg.eigh(H)
    eigenvalues -= np.min(eigenvalues)
    Z = np.sum(np.exp(-eigenvalues / (kB * T)))
    populations = np.exp(-eigenvalues / (kB * T)) / Z
    
    delta_E = eigenvalues[1:] - eigenvalues[:-1]
    delta_pop = populations[1:] - populations[:-1]
    omega_0 = delta_E / hbar
    m_vals = np.arange(s, -s, -1)
    transition_strength = (s + m_vals) * (s - m_vals + 1)
    
    if np.isscalar(gamma_array):
        gamma_array = np.full(len(delta_E), gamma_array)
    else:
        gamma_array = np.asarray(gamma_array)
        if len(gamma_array) != len(delta_E):
            gamma_array = np.pad(gamma_array, (0, len(delta_E) - len(gamma_array)), 'edge')

    numerator = delta_pop * transition_strength
    denominator = (omega_0[:, np.newaxis]**2 - omega_array**2) - (1j * gamma_array[:, np.newaxis] * omega_array)
    
    safe_denominator = np.where(np.abs(denominator) < 1e-20, 1e-20, denominator)
    chi_array = np.sum(numerator[:, np.newaxis] / safe_denominator, axis=0)
    return -chi_array

def calculate_transmission_intensity(omega_array, mu_r_array, eps_bg, d):
    n_complex = np.sqrt(eps_bg * mu_r_array + 0j)
    impe = np.sqrt(mu_r_array / eps_bg + 0j)
    
    lambda_0 = np.full_like(omega_array, np.inf, dtype=float)
    nonzero_mask = omega_array > 1e-12
    lambda_0[nonzero_mask] = (2 * np.pi * c) / omega_array[nonzero_mask]
    
    delta = 2 * np.pi * n_complex * d / lambda_0
    exp_2j_delta = np.exp(2j * delta)
    
    numerator = 4 * impe
    denominator = (1 + impe)**2 * np.exp(-1j * delta) - (1 - impe)**2 * np.exp(1j * delta)
    
    safe_mask = np.abs(denominator) > 1e-12
    t = np.zeros_like(denominator, dtype=complex)
    t[safe_mask] = numerator[safe_mask] / denominator[safe_mask]
    
    return np.abs(t)**2

def calculate_normalized_transmission(omega_array, mu_r_array, eps_bg, d):
    transmission = calculate_transmission_intensity(omega_array, mu_r_array, eps_bg, d)
    min_trans = np.min(transmission)
    max_trans = np.max(transmission)
    if max_trans - min_trans == 0:
        return np.full_like(transmission, 0.5)
    return (transmission - min_trans) / (max_trans - min_trans)

# --- 3. PyMCと連携するためのOpクラス（マルチ磁場対応） ---
class MultiFieldPhysicsModelOp(Op):
    def __init__(self, model_type, freqs_all_fields, data_points_per_field):
        self.model_type = model_type
        self.freqs_all_fields = freqs_all_fields
        self.data_points_per_field = data_points_per_field
        self.n_fields = len(data_points_per_field)
        self.itypes = [pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dvector]  # a, eps_bg, B4, B6, g_factor, gamma
        self.otypes = [pt.dvector]

    def perform(self, node, inputs, output_storage):
        a, eps_bg, B4, B6, g_factor, gamma = inputs
        
        G0 = a * mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
        
        full_predicted_y = []
        start_index = 0
        
        for i, n_points in enumerate(self.data_points_per_field):
            end_index = start_index + n_points
            freqs_rad = self.freqs_all_fields[start_index:end_index] * 2 * np.pi
            
            b_val = magnetic_fields[i]
            
            if self.model_type == 'B_form':
                H_ext_z = b_val / mu0
            else: # H_form
                H_ext_z = b_val

            H = get_hamiltonian(H_ext_z, B4, B6, g_factor)
            chi = calculate_susceptibility(freqs_rad, H, T=35.0, gamma_array=gamma)
            mu_r = 1 + G0 * chi
            
            predicted_y = calculate_normalized_transmission(freqs_rad, mu_r, eps_bg, d)
            full_predicted_y.append(predicted_y)
            
            start_index = end_index
            
        output_storage[0][0] = np.concatenate(full_predicted_y)

def run_bayesian_analysis(model_type, freqs_all_fields, trans_all_fields, data_points_per_field):
    print(f"\n--- [{model_type}] マルチ磁場モデルのサンプリングを開始します ---")
    
    physics_model_op = MultiFieldPhysicsModelOp(model_type, freqs_all_fields, data_points_per_field)

    with pm.Model() as model:
        a = pm.TruncatedNormal('a', mu=a_init, sigma=1.0, lower=0)
        eps_bg = pm.TruncatedNormal('eps_bg', mu=eps_bg_init, sigma=0.1, lower=0)
        B4 = pm.Normal('B4', mu=B4_init, sigma=0.01)
        B6 = pm.Normal('B6', mu=B6_init, sigma=0.001)
        g_factor = pm.TruncatedNormal('g_factor', mu=g_factor_init, sigma=0.2, lower=0)
        gamma = pm.HalfNormal('gamma', sigma=1e11, shape=7)

        mu = physics_model_op(a, eps_bg, B4, B6, g_factor, gamma)
        
        sigma = pm.HalfCauchy('sigma', beta=0.1)
        nu = pm.Gamma('nu', alpha=2, beta=0.1)
        
        Y_obs = pm.StudentT('Y_obs', mu=mu, sigma=sigma, nu=nu, observed=trans_all_fields)

    with model:
        trace = pm.sample(4000, tune=2000, cores=4, chains=4, target_accept=0.9)
        posterior_predictive = pm.sample_posterior_predictive(trace)

    print(f"--- [{model_type}] マルチ磁場モデルのサンプリング完了 ---")
    print(az.summary(trace, var_names=['a', 'eps_bg', 'B4', 'B6', 'g_factor', 'gamma', 'sigma', 'nu']))
    
    return trace, posterior_predictive, model

# --- 4. 改良されたデータ読み込み関数 ---
def load_multi_field_data(file_path=None, sheet_name='Sheet2', b_field_columns=None, freq_limit=0.376, use_manual_data=False):
    """
    複数の磁場条件でのデータを読み込む（Pandasベース）
    
    Parameters:
    file_path: Excelファイルのパス（Noneの場合は手動データを使用）
    sheet_name: シート名
    b_field_columns: 磁場データの列名リスト
    freq_limit: 周波数の上限（フィルタリング用）
    use_manual_data: 手動データを強制使用するかどうか
    
    Returns:
    dict: 各磁場での周波数と透過率データ
    """
    
    if use_manual_data or file_path is None:
        print("手動データを使用します...")
        return load_manual_data_as_dataframe(freq_limit)
    
    try:
        # Excelファイルからデータを読み込み
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
        print(f"Excelファイルの読み込みに成功しました。読み込み件数: {len(df)}件")
        
        # 列名の確認
        print("利用可能な列名:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
        
        # 周波数列の確認
        freq_columns = ['Frequency (THz)', 'Frequency', 'freq', 'frequency']
        freq_column = None
        for col in freq_columns:
            if col in df.columns:
                freq_column = col
                break
        
        if freq_column is None:
            print("周波数列が見つかりません。手動データを使用します。")
            return load_manual_data_as_dataframe(freq_limit)
        
        # 磁場列の自動検出
        if b_field_columns is None:
            transmittance_cols = [col for col in df.columns if 'Transmittance' in col and 'T)' in col]
            if transmittance_cols:
                b_field_columns = transmittance_cols
                print(f"自動検出された透過率列: {b_field_columns}")
            else:
                print("透過率列が見つかりません。手動データを使用します。")
                return load_manual_data_as_dataframe(freq_limit)
        
        # データ処理
        return process_dataframe_to_multi_field(df, freq_column, b_field_columns, freq_limit)
        
    except Exception as e:
        print(f"Excelファイル読み込み中にエラー: {e}")
        print("手動データを使用します...")
        return load_manual_data_as_dataframe(freq_limit)

def load_manual_data_as_dataframe(freq_limit=0.376):
    """
    手動データをPandasのDataFrameとして作成し、処理する
    
    Parameters:
    freq_limit: 周波数の上限（フィルタリング用）
    
    Returns:
    dict: 各磁場での周波数と透過率データ
    """
    
    # 手動データの定義
    frequency_data = [
        1.51E-01, 1.56E-01, 1.61E-01, 1.66E-01, 1.71E-01, 1.76E-01, 1.81E-01, 1.86E-01, 
        1.91E-01, 1.95E-01, 2.00E-01, 2.05E-01, 2.10E-01, 2.15E-01, 2.20E-01, 2.25E-01, 
        2.30E-01, 2.34E-01, 2.39E-01, 2.44E-01, 2.49E-01, 2.54E-01, 2.59E-01, 2.64E-01, 
        2.69E-01, 2.74E-01, 2.78E-01, 2.83E-01, 2.88E-01, 2.93E-01, 2.98E-01, 3.03E-01, 
        3.08E-01, 3.13E-01, 3.18E-01, 3.22E-01, 3.27E-01, 3.32E-01, 3.37E-01, 3.42E-01, 
        3.47E-01, 3.52E-01, 3.57E-01, 3.62E-01, 3.66E-01, 3.71E-01, 3.76E-01
    ]
    
    transmittance_data = {
        'Transmittance (5.0T)': [
            2.14E-03, 3.02E-03, 3.84E-03, 4.53E-03, 5.12E-03, 5.69E-03, 6.36E-03, 7.26E-03,
            8.52E-03, 1.02E-02, 1.25E-02, 1.53E-02, 1.87E-02, 2.25E-02, 2.67E-02, 3.10E-02,
            3.53E-02, 3.96E-02, 4.38E-02, 4.79E-02, 5.19E-02, 5.57E-02, 5.92E-02, 6.22E-02,
            6.45E-02, 6.59E-02, 6.61E-02, 6.51E-02, 6.28E-02, 5.95E-02, 5.54E-02, 5.09E-02,
            4.62E-02, 4.17E-02, 3.74E-02, 3.36E-02, 3.03E-02, 2.75E-02, 2.53E-02, 2.35E-02,
            2.21E-02, 2.11E-02, 2.05E-02, 2.01E-02, 2.00E-02, 2.01E-02, 2.03E-02
        ],
        'Transmittance (7.7T)': [
            3.58E-02, 3.61E-02, 3.40E-02, 3.01E-02, 2.49E-02, 1.92E-02, 1.37E-02, 8.86E-03,
            5.05E-03, 2.40E-03, 8.76E-04, 3.04E-04, 4.03E-04, 8.75E-04, 1.48E-03, 2.14E-03,
            2.90E-03, 3.99E-03, 5.64E-03, 8.10E-03, 1.16E-02, 1.61E-02, 2.16E-02, 2.79E-02,
            3.47E-02, 4.14E-02, 4.74E-02, 5.23E-02, 5.55E-02, 5.70E-02, 5.66E-02, 5.46E-02,
            5.15E-02, 4.75E-02, 4.32E-02, 3.88E-02, 3.46E-02, 3.08E-02, 2.74E-02, 2.46E-02,
            2.24E-02, 2.07E-02, 1.95E-02, 1.87E-02, 1.81E-02, 1.78E-02, 1.77E-02
        ],
        'Transmittance (9.0T)': [
            2.93E-02, 3.81E-02, 4.65E-02, 5.34E-02, 5.81E-02, 6.02E-02, 5.95E-02, 5.64E-02,
            5.11E-02, 4.41E-02, 3.63E-02, 2.82E-02, 2.08E-02, 1.44E-02, 9.57E-03, 6.13E-03,
            3.86E-03, 2.39E-03, 1.38E-03, 6.44E-04, 1.78E-04, 1.39E-04, 7.99E-04, 2.46E-03,
            5.39E-03, 9.69E-03, 1.53E-02, 2.17E-02, 2.86E-02, 3.52E-02, 4.08E-02, 4.50E-02,
            4.76E-02, 4.83E-02, 4.74E-02, 4.51E-02, 4.17E-02, 3.75E-02, 3.31E-02, 2.86E-02,
            2.46E-02, 2.12E-02, 1.87E-02, 1.70E-02, 1.62E-02, 1.61E-02, 1.64E-02
        ]
    }
    
    # DataFrameを作成
    data_dict = {'Frequency (THz)': frequency_data}
    data_dict.update(transmittance_data)
    df = pd.DataFrame(data_dict)
    
    print(f"手動データでDataFrameを作成しました。形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 標準のデータ処理関数を使用
    freq_column = 'Frequency (THz)'
    b_field_columns = list(transmittance_data.keys())
    
    return process_dataframe_to_multi_field(df, freq_column, b_field_columns, freq_limit)

def process_dataframe_to_multi_field(df, freq_column, b_field_columns, freq_limit):
    """
    DataFrameをマルチ磁場データ形式に変換する共通関数
    
    Parameters:
    df: pandas DataFrame
    freq_column: 周波数列名
    b_field_columns: 透過率列名のリスト
    freq_limit: 周波数の上限
    
    Returns:
    dict: 各磁場での周波数と透過率データ
    """
    
    # データの基本チェック
    if freq_column not in df.columns:
        raise ValueError(f"周波数列 '{freq_column}' が見つかりません")
    
    # 周波数データを取得
    frequency_full = df[freq_column].dropna().astype(float)
    
    # フィルタリング
    filter_mask = frequency_full <= freq_limit
    df_filtered = df[filter_mask].copy()
    frequency_filtered = frequency_full[filter_mask]
    
    print(f"フィルタリング後のデータ件数: {len(df_filtered)}件")
    print(f"周波数範囲: {frequency_filtered.min():.3f} - {frequency_filtered.max():.3f} THz")
    
    # 各磁場でのデータを処理
    multi_field_data = {}
    
    for col in b_field_columns:
        if col not in df.columns:
            print(f"警告: {col} 列が見つかりません。スキップします。")
            continue
            
        # 磁場値を列名から抽出
        import re
        b_match = re.search(r'\(([0-9.]+)T\)', col)
        if b_match:
            b_value = float(b_match.group(1))
        else:
            print(f"警告: {col} から磁場値を抽出できません。")
            continue
        
        # 透過率データを取得（NaNを除去）
        transmittance_full = df[col].dropna().astype(float)
        transmittance_filtered = df_filtered[col].dropna().astype(float)
        
        # データサイズの整合性チェック
        min_length = min(len(frequency_filtered), len(transmittance_filtered))
        frequency_filtered_adj = frequency_filtered.iloc[:min_length]
        transmittance_filtered_adj = transmittance_filtered.iloc[:min_length]
        
        min_length_full = min(len(frequency_full), len(transmittance_full))
        frequency_full_adj = frequency_full.iloc[:min_length_full]
        transmittance_full_adj = transmittance_full.iloc[:min_length_full]
        
        # 正規化（フィルタリングしたデータの範囲で）
        min_exp = transmittance_filtered_adj.min()
        max_exp = transmittance_filtered_adj.max()
        
        if max_exp - min_exp == 0:
            print(f"警告: 磁場 {b_value} T のデータに変動がありません。")
            continue
            
        transmittance_normalized = (transmittance_filtered_adj - min_exp) / (max_exp - min_exp)
        transmittance_normalized_full = (transmittance_full_adj - min_exp) / (max_exp - min_exp)
        
        multi_field_data[b_value] = {
            'frequency_filtered': frequency_filtered_adj.values,
            'frequency_full': frequency_full_adj.values,
            'transmittance_normalized': transmittance_normalized.values,
            'transmittance_normalized_full': transmittance_normalized_full.values,
            'omega_filtered': frequency_filtered_adj.values * 1e12 * 2 * np.pi,
            'omega_full': frequency_full_adj.values * 1e12 * 2 * np.pi
        }
        
        print(f"磁場 {b_value} T のデータを処理しました。")
        print(f"  フィルタリング後データ点数: {len(transmittance_normalized)}")
        print(f"  透過率範囲: {transmittance_filtered_adj.min():.6f} - {transmittance_filtered_adj.max():.6f}")
    
    return multi_field_data

# --- 5. パラメータ分析関数 ---
def analyze_results(model_type, trace):
    print(f"\n=== {model_type} マルチ磁場ベイズ最適化によるパラメータ分析 ===")
    summary = az.summary(trace, var_names=['a', 'eps_bg', 'B4', 'B6', 'g_factor', 'gamma', 'sigma', 'nu'])
    
    a_mean = summary.loc['a', 'mean']
    g_factor_mean = summary.loc['g_factor', 'mean']
    B4_mean = summary.loc['B4', 'mean']
    B6_mean = summary.loc['B6', 'mean']
    eps_bg_mean = summary.loc['eps_bg', 'mean']
    gamma_means = summary[summary.index.str.startswith('gamma[')]['mean']
    sigma_mean = summary.loc['sigma', 'mean']
    nu_mean = summary.loc['nu', 'mean']

    G0 = a_mean * mu0 * N_spin * (g_factor_mean * muB)**2 / (2 * hbar)

    print(f"スケーリング係数 a: {a_mean:.3f}")
    print(f"g因子: {g_factor_mean:.3f} (理論値: ~2.0)")
    print(f"B4: {B4_mean:.6f}, B6: {B6_mean:.6f}")
    print(f"G0: {G0:.3e}")
    print(f"gamma(平均): {gamma_means.mean():.3e} ± {gamma_means.std():.3e}")
    for i, g_mean in enumerate(gamma_means):
        print(f"  gamma[{i}]: {g_mean:.3e}")
    print(f"eps_bg: {eps_bg_mean:.3f}")
    print(f"Student-t nu: {nu_mean:.3f}, sigma: {sigma_mean:.6f}")

# --- 6. 診断プロット関数 ---
def create_diagnostic_plots(traces, posterior_predictives, model_comparison):
    print("\n=== 診断プロット作成中 ===")
    try:
        var_names_trace = ['g_factor', 'eps_bg', 'a', 'gamma']
        num_vars = len(var_names_trace)
        fig1, axes1 = plt.subplots(num_vars, 2, figsize=(12, 3 * num_vars))
        az.plot_trace(traces['H_form'], var_names=var_names_trace, axes=axes1)
        fig1.suptitle('H_form Trace Plot', fontsize=16)
        fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(IMAGE_DIR / 'multi_field_trace_H_form.png')
        plt.close(fig1)

        fig2, axes2 = plt.subplots(num_vars, 2, figsize=(12, 3 * num_vars))
        az.plot_trace(traces['B_form'], var_names=var_names_trace, axes=axes2)
        fig2.suptitle('B_form Trace Plot', fontsize=16)
        fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(IMAGE_DIR / 'multi_field_trace_B_form.png')
        plt.close(fig2)

        # フォレストプロット
        if len(traces) > 1:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            # トレース内の実際の変数名を使用
            try:
                # トレース内の利用可能な変数をチェック
                first_trace = list(traces.values())[0]
                available_vars = list(first_trace.posterior.data_vars.keys())
                plot_vars = [var for var in ['a', 'g_factor', 'eps_bg', 'd', 'B4', 'B6'] if var in available_vars]
                
                if plot_vars:
                    az.plot_forest(traces, var_names=plot_vars, ax=ax3)
                    plt.title('パラメータ比較 (フォレストプロット)', fontsize=14)
                else:
                    # 変数名指定なしでプロット
                    az.plot_forest(traces, ax=ax3)
                    plt.title('パラメータ比較 (フォレストプロット - 全変数)', fontsize=14)
                    
            except Exception as e:
                print(f"フォレストプロット作成中にエラー: {e}")
                # シンプルなプロットにフォールバック
                ax3.text(0.5, 0.5, f'フォレストプロット作成エラー:\n{str(e)}', 
                        ha='center', va='center', transform=ax3.transAxes)
                plt.title('パラメータ比較 (エラー)', fontsize=14)
                
            plt.tight_layout()
            plt.savefig(IMAGE_DIR / 'multi_field_forest_plot.png', dpi=300)
            plt.close(fig3)
        
        # エネルギープロット
        if 'H_form' in traces:
            fig4 = az.plot_energy(traces['H_form'])
            plt.suptitle('H_form モデル エネルギープロット', fontsize=14)
            plt.tight_layout()
            plt.savefig(IMAGE_DIR / 'multi_field_energy_H_form.png', dpi=300)
            plt.close(fig4)
        
        print("診断プロットが正常に作成されました。")
        return True
        
    except Exception as e:
        print(f"診断プロット作成中にエラー: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def plot_multi_field_results(multi_field_data, traces, model_types, colors):
    """マルチ磁場の結果をプロットする関数"""
    
    # ① フィッティング領域のプロット
    fig1, axes1 = plt.subplots(1, len(multi_field_data), figsize=(5*len(multi_field_data), 6))
    if len(multi_field_data) == 1:
        axes1 = [axes1]
    
    sorted_b_values = sorted(multi_field_data.keys())
    
    for i, b_val in enumerate(sorted_b_values):
        data = multi_field_data[b_val]
        
        # 実験データをプロット
        axes1[i].scatter(data['frequency_filtered'], data['transmittance_normalized'], 
                        alpha=0.8, s=30, color='black', label='実験データ')
        
        # 各モデルのベストフィット曲線
        for mt in model_types:
            if mt in traces:
                trace = traces[mt]
                
                # ベストフィット曲線の計算
                a_mean = trace.posterior['a'].mean().item()
                g_factor_mean = trace.posterior['g_factor'].mean().item()
                B4_mean = trace.posterior['B4'].mean().item()
                B6_mean = trace.posterior['B6'].mean().item()
                gamma_mean = trace.posterior['gamma'].mean(dim=['chain', 'draw']).values
                
                H_best = get_hamiltonian(B_ext_z=b_val, B4=B4_mean, 
                                       B6=B6_mean, g_factor=g_factor_mean)
                G0_best = a_mean * mu0 * N_spin * (g_factor_mean * muB)**2 / (2 * hbar)
                chi_best_raw = calculate_susceptibility(data['omega_filtered'], H_best, T=35.0, 
                                                      gamma_array=gamma_mean)
                chi_best = G0_best * chi_best_raw
                
                if mt == 'H_form':
                    mu_r_best = 1 + chi_best
                else: 
                    mu_r_best = 1 / (1-chi_best)
                
                best_fit_prediction = calculate_normalized_transmission(data['omega_filtered'], 
                                                                     mu_r_best, eps_bg, d)
                
                axes1[i].plot(data['frequency_filtered'], best_fit_prediction, 
                            color=colors[mt], lw=3, label=f'ベストフィット ({mt})')
        
        axes1[i].set_xlabel('周波数 (THz)')
        axes1[i].set_ylabel('正規化透過率')
        axes1[i].legend()
        axes1[i].grid(True, alpha=0.3)
        axes1[i].set_title(f'磁場 {b_val} T', fontsize=14)
        axes1[i].set_ylim(-0.1, 1.1)
    
    fig1.suptitle('マルチ磁場ベイズ最適化結果: フィッティング領域', fontsize=16)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / 'multi_field_fitting_region.png', dpi=300)
    plt.show()
    plt.close(fig1)

    # ② 全領域の予測プロット
    fig2, axes2 = plt.subplots(1, len(multi_field_data), figsize=(5*len(multi_field_data), 6))
    if len(multi_field_data) == 1:
        axes2 = [axes2]
    
    for i, b_val in enumerate(sorted_b_values):
        data = multi_field_data[b_val]
        
        # 実験データをプロット
        axes2[i].scatter(data['frequency_full'], data['transmittance_normalized_full'], 
                        alpha=0.6, s=20, color='gray', label='実験データ（全領域）')
        
        # 各モデルのベストフィット曲線（全領域）
        freq_plot_full = np.linspace(np.min(data['frequency_full']), 
                                    np.max(data['frequency_full']), 1000)
        omega_plot_full = freq_plot_full * 1e12 * 2 * np.pi
        
        for mt in model_types:
            if mt in traces:
                trace = traces[mt]
                
                a_mean = trace.posterior['a'].mean().item()
                g_factor_mean = trace.posterior['g_factor'].mean().item()
                B4_mean = trace.posterior['B4'].mean().item()
                B6_mean = trace.posterior['B6'].mean().item()
                gamma_mean = trace.posterior['gamma'].mean(dim=['chain', 'draw']).values
                
                H_best = get_hamiltonian(B_ext_z=b_val, B4=B4_mean, 
                                       B6=B6_mean, g_factor=g_factor_mean)
                G0_best = a_mean * mu0 * N_spin * (g_factor_mean * muB)**2 / (2 * hbar)
                chi_best_raw_full = calculate_susceptibility(omega_plot_full, H_best, T=35.0, 
                                                           gamma_array=gamma_mean)
                chi_best_full = G0_best * chi_best_raw_full
                
                if mt == 'H_form':
                    mu_r_best_full = 1 + chi_best_full
                else: 
                    mu_r_best_full = 1 / (1-chi_best_full)
                
                best_fit_prediction_full = calculate_normalized_transmission(omega_plot_full, 
                                                                           mu_r_best_full, 
                                                                           eps_bg, d)
                
                axes2[i].plot(freq_plot_full, best_fit_prediction_full, 
                            color=colors[mt], lw=2, label=f'ベストフィット 全領域 ({mt})')
        
        # フィッティング領域境界を表示
        axes2[i].axvline(x=0.376, color='red', linestyle=':', alpha=0.7, label='フィッティング領域上限')
        
        axes2[i].set_xlabel('周波数 (THz)')
        axes2[i].set_ylabel('正規化透過率')
        axes2[i].legend()
        axes2[i].grid(True, alpha=0.3)
        axes2[i].set_title(f'磁場 {b_val} T (全領域予測)', fontsize=14)
        axes2[i].set_ylim(-0.1, 2.0)
    
    fig2.suptitle('マルチ磁場ベイズ最適化結果: 全領域予測', fontsize=16)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / 'multi_field_full_range_prediction.png', dpi=300)
    plt.show()
    plt.close(fig2)

def plot_bayesian_credible_intervals(multi_field_data, traces, model_types, colors, n_samples=500):
    """ベイズ推定による95%信用区間をプロットする関数"""
    
    sorted_b_values = sorted(multi_field_data.keys())
    
    for mt in model_types:
        if mt not in traces:
            continue
            
        trace = traces[mt]
        
        # サンプル数を制限して計算時間を短縮（全体で共通使用）
        total_samples = len(trace.posterior.chain) * len(trace.posterior.draw)
        sample_indices = np.random.choice(total_samples, 
                                        size=min(n_samples, total_samples), 
                                        replace=False)
        
        # フィッティング領域の信用区間プロット
        fig1, axes1 = plt.subplots(1, len(multi_field_data), figsize=(5*len(multi_field_data), 6))
        if len(multi_field_data) == 1:
            axes1 = [axes1]
        
        for i, b_val in enumerate(sorted_b_values):
            data = multi_field_data[b_val]
            
            # 実験データをプロット
            axes1[i].scatter(data['frequency_filtered'], data['transmittance_normalized'], 
                            alpha=0.8, s=30, color='black', label='実験データ', zorder=5)
            
            # ベイズサンプルから予測の分布を計算
            predictions = []
            
            for idx in sample_indices:
                chain_idx = idx // len(trace.posterior.draw)
                draw_idx = idx % len(trace.posterior.draw)
                
                # サンプルからパラメータを取得
                a_sample = float(trace.posterior['a'].isel(chain=chain_idx, draw=draw_idx))
                g_factor_sample = float(trace.posterior['g_factor'].isel(chain=chain_idx, draw=draw_idx))
                gamma_sample = trace.posterior['gamma'].isel(chain=chain_idx, draw=draw_idx).values
                eps_bg_sample = float(trace.posterior['eps_bg'].isel(chain=chain_idx, draw=draw_idx))
                B4_sample = float(trace.posterior['B4'].isel(chain=chain_idx, draw=draw_idx))
                B6_sample = float(trace.posterior['B6'].isel(chain=chain_idx, draw=draw_idx))
                
                # このサンプルでの予測を計算
                H_sample = get_hamiltonian(B_ext_z=b_val, B4=B4_sample, B6=B6_sample, g_factor=g_factor_sample)
                G0_sample = a_sample * mu0 * N_spin * (g_factor_sample * muB)**2 / (2 * hbar)
                chi_sample_raw = calculate_susceptibility(data['omega_filtered'], H_sample, T=35.0, 
                                                        gamma_array=gamma_sample)
                chi_sample = G0_sample * chi_sample_raw
                
                if mt == 'H_form':
                    mu_r_sample = 1 + chi_sample
                else: 
                    mu_r_sample = 1 / (1-chi_sample)
                
                prediction_sample = calculate_normalized_transmission(data['omega_filtered'], mu_r_sample, 
                                                                    eps_bg_sample, d)
                predictions.append(prediction_sample)
            
            predictions = np.array(predictions)
            
            # 95%信用区間を計算
            mean_prediction = np.mean(predictions, axis=0)
            ci_lower = np.percentile(predictions, 2.5, axis=0)
            ci_upper = np.percentile(predictions, 97.5, axis=0)
            
            # 信用区間をプロット
            axes1[i].fill_between(data['frequency_filtered'], ci_lower, ci_upper, 
                                 alpha=0.3, color=colors[mt], label=f'95%信用区間 ({mt})')
            
            # 平均予測をプロット
            axes1[i].plot(data['frequency_filtered'], mean_prediction, 
                         color=colors[mt], lw=2, label=f'平均予測 ({mt})')
            
            axes1[i].set_xlabel('周波数 (THz)')
            axes1[i].set_ylabel('正規化透過率')
            axes1[i].legend()
            axes1[i].grid(True, alpha=0.3)
            axes1[i].set_title(f'磁場 {b_val} T - 95%信用区間', fontsize=14)
            axes1[i].set_ylim(-0.1, 1.1)
        
        fig1.suptitle(f'{mt}モデル: ベイズ推定95%信用区間 (フィッティング領域)', fontsize=16)
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / f'multi_field_credible_intervals_{mt}_fitting.png', dpi=300)
        plt.show()
        plt.close(fig1)
        
        # 全領域の信用区間プロット
        fig2, axes2 = plt.subplots(1, len(multi_field_data), figsize=(5*len(multi_field_data), 6))
        if len(multi_field_data) == 1:
            axes2 = [axes2]
        
        # 全領域用のサンプル数を削減（計算コストを考慮）
        sample_indices_reduced = sample_indices[:min(100, len(sample_indices))]
        
        for i, b_val in enumerate(sorted_b_values):
            data = multi_field_data[b_val]
            
            # 実験データをプロット（全領域）
            axes2[i].scatter(data['frequency_full'], data['transmittance_normalized_full'], 
                            alpha=0.6, s=20, color='gray', label='実験データ（全領域）', zorder=5)
            
            # 全領域での予測用周波数グリッド
            freq_plot_full = np.linspace(np.min(data['frequency_full']), 
                                        np.max(data['frequency_full']), 200)
            omega_plot_full = freq_plot_full * 1e12 * 2 * np.pi
            
            # ベイズサンプルから全領域予測の分布を計算
            predictions_full = []
            
            for idx in sample_indices_reduced:
                chain_idx = idx // len(trace.posterior.draw)
                draw_idx = idx % len(trace.posterior.draw)
                
                # サンプルからパラメータを取得
                a_sample = float(trace.posterior['a'].isel(chain=chain_idx, draw=draw_idx))
                g_factor_sample = float(trace.posterior['g_factor'].isel(chain=chain_idx, draw=draw_idx))
                gamma_sample = trace.posterior['gamma'].isel(chain=chain_idx, draw=draw_idx).values
                eps_bg_sample = float(trace.posterior['eps_bg'].isel(chain=chain_idx, draw=draw_idx))
                B4_sample = float(trace.posterior['B4'].isel(chain=chain_idx, draw=draw_idx))
                B6_sample = float(trace.posterior['B6'].isel(chain=chain_idx, draw=draw_idx))
                
                # このサンプルでの予測を計算
                H_sample = get_hamiltonian(B_ext_z=b_val, B4=B4_sample, B6=B6_sample, g_factor=g_factor_sample)
                G0_sample = a_sample * mu0 * N_spin * (g_factor_sample * muB)**2 / (2 * hbar)
                chi_sample_raw_full = calculate_susceptibility(omega_plot_full, H_sample, T=35.0, 
                                                           gamma_array=gamma_sample)
                chi_sample_full = G0_sample * chi_sample_raw_full
                
                if mt == 'H_form':
                    mu_r_sample_full = 1 + chi_sample_full
                else: 
                    mu_r_sample_full = 1 / (1-chi_sample_full)
                
                prediction_sample_full = calculate_normalized_transmission(omega_plot_full, mu_r_sample_full, 
                                                                         eps_bg_sample, d)
                predictions_full.append(prediction_sample_full)
            
            predictions_full = np.array(predictions_full)
            
            # 95%信用区間を計算
            mean_prediction_full = np.mean(predictions_full, axis=0)
            ci_lower_full = np.percentile(predictions_full, 2.5, axis=0)
            ci_upper_full = np.percentile(predictions_full, 97.5, axis=0)
            
            # 信用区間をプロット
            axes2[i].fill_between(freq_plot_full, ci_lower_full, ci_upper_full, 
                                 alpha=0.3, color=colors[mt], label=f'95%信用区間 ({mt})')
            
            # 平均予測をプロット
            axes2[i].plot(freq_plot_full, mean_prediction_full, 
                         color=colors[mt], lw=2, label=f'平均予測 ({mt})')
            
            # フィッティング領域境界を表示
            axes2[i].axvline(x=0.376, color='red', linestyle=':', alpha=0.7, label='フィッティング領域上限')
            
            axes2[i].set_xlabel('周波数 (THz)')
            axes2[i].set_ylabel('正規化透過率')
            axes2[i].legend()
            axes2[i].grid(True, alpha=0.3)
            axes2[i].set_title(f'磁場 {b_val} T - 95%信用区間（全領域）', fontsize=14)
            axes2[i].set_ylim(-0.1, 2.0)
        
        fig2.suptitle(f'{mt}モデル: ベイズ推定95%信用区間 (全領域予測)', fontsize=16)
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / f'multi_field_credible_intervals_{mt}_full.png', dpi=300)
        plt.show()
        plt.close(fig2)
    
    print("95%信用区間プロットが正常に作成されました。")

# --- 7. メイン実行ブロック ---
if __name__ == '__main__':
    try:
        import psutil
        import os
        
        def print_memory_usage(stage):
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"[{stage}] メモリ使用量: {memory_mb:.1f} MB")
    except ImportError:
        def print_memory_usage(stage):
            print(f"[{stage}] メモリ監視は利用できません (psutilが必要)")
    
    print_memory_usage("開始時")
    
    # --- データの読み込み ---
    print("=== マルチ磁場実験データを読み込みます ===")
    
    # ファイルパスの設定
    file_path = "C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\Circular_Polarization_B_Field.xlsx"
    
    # まずExcelファイルからの読み込みを試行
    multi_field_data = load_multi_field_data(
        file_path=file_path, 
        sheet_name='Sheet2', 
        freq_limit=0.376, 
        use_manual_data=False  # まずExcelファイルを試す
    )
    
    # Excelファイルでの読み込みが失敗した場合、手動データを使用
    if multi_field_data is None or len(multi_field_data) == 0:
        print("Excelファイルからの読み込みに失敗しました。手動データを使用します。")
        multi_field_data = load_multi_field_data(use_manual_data=True, freq_limit=0.376)
    
    if multi_field_data is None or len(multi_field_data) == 0:
        print("データの読み込みに失敗しました。プログラムを終了します。")
        exit()
    
    print(f"読み込まれた磁場条件: {sorted(multi_field_data.keys())} T")
    print_memory_usage("データ読み込み後")

    # --- 連結されたデータの準備 ---
    sorted_b_values = sorted(multi_field_data.keys())
    omega_arrays = [multi_field_data[b_val]['omega_filtered'] for b_val in sorted_b_values]
    concatenated_transmittance = np.concatenate([multi_field_data[b_val]['transmittance_normalized'] 
                                               for b_val in sorted_b_values])
    
    print(f"連結されたデータサイズ: {len(concatenated_transmittance)}")
    print(f"各磁場のデータ点数: {[len(multi_field_data[b_val]['transmittance_normalized']) for b_val in sorted_b_values]}")

    # --- モデル比較の準備 ---
    model_types = ['H_form', 'B_form']
    traces = {}
    ppcs = {}
    n_transitions = 7

    # --- 各モデルでサンプリングを実行 ---
    for mt in model_types:
        print(f"\n--- [{mt}] マルチ磁場モデルのサンプリングを開始します ---")
        physics_model = MultiFieldPhysicsModelOp(omega_arrays, T_val=35.0, B_values=sorted_b_values, 
                                                 model_type=mt, n_transitions=n_transitions)
        
        with pm.Model() as model:
            # 事前分布の設定            
            # gammaの事前分布をトランケートして0以上に制限
            log_gamma_sigma = pm.HalfNormal('log_gamma_sigma', sigma=1.0)
            log_gamma_array = pm.Normal('log_gamma', mu=np.log(gamma_init), sigma=log_gamma_sigma, shape=n_transitions)
            gamma_array = pm.Deterministic('gamma', pt.exp(log_gamma_array))

            # 物理パラメータの事前分布
            a = pm.TruncatedNormal('a', mu=a_init, sigma=1.0, lower=0.0, upper=5.0)
            eps_bg = pm.TruncatedNormal('eps_bg', mu=eps_bg_init, sigma=2.0, lower=11.0, upper=16.0)
            B4 = pm.Normal('B4', mu=B4_init, sigma=abs(B4_init)*0.5)
            B6 = pm.Normal('B6', mu=B6_init, sigma=abs(B6_init)*0.5)
            g_factor = pm.TruncatedNormal('g_factor', mu=g_factor_init, sigma=0.1, lower=1.8, upper=2.3)
            
            # Student-t分布による外れ値耐性
            nu = pm.Gamma('nu', alpha=3, beta=0.2)  # 自由度パラメータ
            sigma_obs = pm.HalfCauchy('sigma', beta=0.5)  # より保守的

            # 物理モデルの予測（全磁場データ）
            mu = physics_model(a, gamma_array, eps_bg, B4, B6, g_factor)

            # ロバストな尤度関数
            Y_obs = pm.StudentT('Y_obs', 
                           nu=nu,
                           mu=mu, 
                           sigma=sigma_obs, 
                           observed=concatenated_transmittance)            
            
            traces[mt] = pm.sample(
                2000,  # サンプル数を調整（メモリ使用量考慮）
                tune=2000,  # チューニング数を調整
                target_accept=0.95,  # 受容率を現実的な値に
                chains=4, 
                cores=4, 
                random_seed=42, 
                init='adapt_diag',
                idata_kwargs={"log_likelihood": True},
                nuts={"max_treedepth": 12},  # ツリー深度を調整
                compute_convergence_checks=False  # 収束チェックを無効化してメモリ節約
            )
            
            ppcs[mt] = pm.sample_posterior_predictive(traces[mt], random_seed=42)
        
        print(f"--- [{mt}] マルチ磁場モデルのサンプリング完了 ---")
        print(az.summary(traces[mt], var_names=['a', 'eps_bg', 'B4', 'B6', 'g_factor', 'sigma']))

    # --- 5. モデル比較の結果表示 ---
    print("\n--- マルチ磁場ベイズ的モデル比較 (LOO-CV) ---")
    idata_dict = {k: v for k, v in traces.items()} 
    compare_df = az.compare(idata_dict)
    print(compare_df)
    
    try:
        axes = az.plot_compare(compare_df, figsize=(8, 4))
        fig = axes.ravel()[0].figure if hasattr(axes, "ravel") else axes.figure
        fig.suptitle('マルチ磁場モデル比較', fontsize=16)
        fig.tight_layout()
        plt.savefig(IMAGE_DIR / 'multi_field_model_comparison.png', dpi=300)
        plt.show()
        plt.close()
    except Exception as e:
        print(f"モデル比較プロット中にエラー: {e}")

    # --- 6. ベストフィット曲線のプロット ---    
    colors = {'H_form': 'blue', 'B_form': 'red'}

    # パラメータの計算
    best_params = {}
    
    for mt in model_types:
        trace = traces[mt]; ppc = ppcs[mt]
        analyze_results(mt, trace)
        
        # ベストフィットパラメータの抽出
        a_mean = trace.posterior['a'].mean().item()
        gamma_mean = trace.posterior['gamma'].mean(dim=['chain', 'draw']).values
        eps_bg_mean = trace.posterior['eps_bg'].mean().item()
        B4_mean = trace.posterior['B4'].mean().item()
        B6_mean = trace.posterior['B6'].mean().item()
        g_factor_mean = trace.posterior['g_factor'].mean().item()
        
        best_params[mt] = {
            'a_mean': a_mean,
            'gamma_mean': gamma_mean,
            'eps_bg_mean': eps_bg_mean,
            'B4_mean': B4_mean,
            'B6_mean': B6_mean,
            'g_factor_mean': g_factor_mean
        }

    # マルチ磁場結果のプロット
    plot_multi_field_results(multi_field_data, traces, model_types, colors)

    # --- 7. フィッティング品質の評価 ---
    print("\n=== マルチ磁場フィッティング品質の評価 ===")
    
    for mt in model_types:
        trace = traces[mt]
        ppc = ppcs[mt]
        y_pred_mean = ppc.posterior_predictive['Y_obs'].mean(dim=['chain', 'draw']).values
        rmse_total = np.sqrt(np.mean((concatenated_transmittance - y_pred_mean)**2))
        
        print(f"\n{mt} モデル:")
        print(f"  全磁場データ RMSE: {rmse_total:.6f}")
        
        # 磁場別RMSE
        start_idx = 0
        for b_val in sorted_b_values:
            data_length = len(multi_field_data[b_val]['transmittance_normalized'])
            end_idx = start_idx + data_length
            
            field_data = concatenated_transmittance[start_idx:end_idx]
            field_pred = y_pred_mean[start_idx:end_idx]
            field_rmse = np.sqrt(np.mean((field_data - field_pred)**2))
            
            print(f"  磁場 {b_val} T RMSE: {field_rmse:.6f}")
            start_idx = end_idx

    # --- 8. 診断プロットと残差分析 ---
    try:
        print("\n=== 診断プロット作成中 ===")
        create_diagnostic_plots(traces, ppcs, None)
        print_memory_usage("診断・残差分析後")
    except Exception as e:
        print(f"診断・残差分析中にエラー: {e}")

    # --- 9. ベイズ推定95%信用区間プロット ---
    try:
        print("\n=== ベイズ推定95%信用区間プロット作成中 ===")
        plot_bayesian_credible_intervals(multi_field_data, traces, model_types, colors, n_samples=300)
        print_memory_usage("信用区間プロット後")
    except Exception as e:
        print(f"信用区間プロット作成中にエラー: {e}")

    print("マルチ磁場ベイズ推定の全ての処理が完了しました。")
    print(f"\n=== 結果ファイル (保存先: {IMAGE_DIR}) ===")
    print("- multi_field_fitting_region.png: フィッティング結果")
    print("- multi_field_full_range_prediction.png: 全領域予測")
    print("- multi_field_model_comparison.png: モデル比較")
    print("- multi_field_trace_H_form.png: H_formトレース")
    print("- multi_field_trace_B_form.png: B_formトレース")
    print("- multi_field_forest_plot.png: パラメータ比較")
    print("- multi_field_energy_H_form.png: エネルギープロット")
    print("- multi_field_credible_intervals_H_form_fitting.png: H_form 95%信用区間(フィッティング)")
    print("- multi_field_credible_intervals_H_form_full.png: H_form 95%信用区間(全領域)")
    print("- multi_field_credible_intervals_B_form_fitting.png: B_form 95%信用区間(フィッティング)")
    print("- multi_field_credible_intervals_B_form_full.png: B_form 95%信用区間(全領域)")
    
    print("\n=== 結果要約 ===")
    print(f"最良モデル: {compare_df.index[0]}")
    print("主要パラメータ:")
    best_model = compare_df.index[0]
    if best_model in best_params:
        params = best_params[best_model]
        print(f"  g因子: {params['g_factor_mean']:.3f}")
        print(f"  試料厚さ: {d*1e6:.1f} μm (定数)")
        print(f"  eps_bg: {params['eps_bg_mean']:.3f}")
        print(f"  スケーリング係数: {params['a_mean']:.3f}")
        print(f"  B4: {params['B4_mean']:.6f}, B6: {params['B6_mean']:.6f}")
        print(f"  gamma_base: {az.summary(traces[best_model]).loc['gamma_base', 'mean']:.3e}")

    print("\n信用区間プロットについて:")
    print("- 95%信用区間は、パラメータの不確実性を考慮した予測の範囲を示します")
    print("- 塗りつぶし領域は、95%の確率でデータが存在する範囲です")
    print("- 実線は、すべてのサンプルからの平均予測を表します")
    print("- フィッティング領域と全領域の両方で信用区間が計算されます")

    print("\n注意事項:")
    print("- divergencesが発生している →→ サンプリング品質に注意してください。")
    print("- R-hat > 1.01 のパラメータがある →→ 収束を確認してください。")
    print("- ESS < 100 のパラメータがある →→ より多くのサンプルが必要です。")
    print("- ESS < 100 のパラメータがある →→ より多くのサンプルが必要です。")
