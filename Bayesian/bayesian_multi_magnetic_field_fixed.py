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
import pathlib
try:
    import japanize_matplotlib # 日本語表示のため
except ImportError:
    print("注意: japanize_matplotlib がインストールされていません。日本語表示に影響する可能性があります。")
    print("インストール方法: pip install japanize-matplotlib")

# --- 0. プロット設定 ---
plt.rcParams['figure.dpi'] = 100
IMAGE_DIR = pathlib.Path(__file__).parent / "pymc_B_images"
IMAGE_DIR.mkdir(exist_ok=True)
print(f"画像は {IMAGE_DIR} に保存されます。")

# --- 1. 物理定数とパラメータ定義 ---
kB = 1.380649e-23; muB = 9.274010e-24; hbar = 1.054571e-34; c = 299792458; mu0 = 4.0 * np.pi * 1e-7
s = 3.5
N_spin = 24/1.238 * 1e27 #GGGのスピン数密度

"""
手動計算で分かっている最良の値を初期値とする
"""
d = 157.8e-6 * 0.99
eps_bg = 13.1404
B4 = 0.8 / 240 * 0.606; B6 = 0.04 / 5040 * -1.513

B4_init = B4 
B6_init = B6
gamma_init = 0.11e12
a_init = 1.5
g_factor_init = 2.02 


# --- 2. 汎用化された物理モデル関数 ---

def get_hamiltonian(B_ext_z, g_factor, B4_val=None, B6_val=None):
    # デフォルト値としてグローバル定数を使用
    if B4_val is None:
        B4_val = B4
    if B6_val is None:
        B6_val = B6

    m_values = np.arange(s, -s - 1, -1)
    Sz = np.diag(m_values)
    O04 = 60*np.diag([7,-13,-3,9,9,-3,-13,7]) 
    X_O44 = np.zeros((8,8)); X_O44[3,7],X_O44[4,0]=np.sqrt(35),np.sqrt(35); X_O44[2,6],X_O44[5,1]=5*np.sqrt(3),5*np.sqrt(3); O44=12*(X_O44+X_O44.T)
    O06 = 1260*np.diag([1,-5,9,-5,-5,9,-5,1])
    X_O46 = np.zeros((8,8)); X_O46[3,7],X_O46[4,0]=3*np.sqrt(35),3*np.sqrt(35); X_O46[2,6],X_O46[5,1]=-7*np.sqrt(3),-7*np.sqrt(3); O46=60*(X_O46+X_O46.T)
    H_cf = (B4_val * kB) * (O04 + 5 * O44) + (B6_val * kB) * (O06 - 21 * O46)
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
    
    # gamma_arrayがdelta_Eと同じ次元を持つように調整
    if len(gamma_array) != len(delta_E):
        if len(gamma_array) > len(delta_E):
            gamma_array = gamma_array[:len(delta_E)]
        else:
            gamma_array = np.pad(gamma_array, (0, len(delta_E) - len(gamma_array)), 'edge')

    numerator = delta_pop * transition_strength
    denominator = (omega_0[:, np.newaxis] - omega_array) - (1j * gamma_array[:, np.newaxis])
    
    # ゼロ除算チェックを追加
    safe_denominator = np.where(np.abs(denominator) < 1e-15, 1e-15, denominator)
    chi_array = np.sum(numerator[:, np.newaxis] / safe_denominator, axis=0)
    return -chi_array

def calculate_transmission_intensity(omega_array, mu_r_array):
    """強度|t|^2を計算するヘルパー関数（数値安定性を改善）"""    
    n_complex = np.sqrt(eps_bg * mu_r_array + 0j)
    impe = np.sqrt(mu_r_array / eps_bg + 0j)
    
    lambda_0 = np.full_like(omega_array, np.inf, dtype=float)
    nonzero_mask = omega_array > 1e-12  # より安全な閾値
    lambda_0[nonzero_mask] = (2 * np.pi * c) / omega_array[nonzero_mask]
    
    delta = 2 * np.pi * n_complex * d / lambda_0
    
    exp_2j_delta = np.exp(2j * delta)
    exp_j_delta = np.exp(1j * delta)
    
    numerator = 4 * impe * exp_j_delta
    denominator = (1 + impe)**2 - (1 - impe)**2 * exp_2j_delta
    
    safe_mask = np.abs(denominator) > 1e-12
    t = np.zeros_like(denominator, dtype=complex)
    t[safe_mask] = numerator[safe_mask] / denominator[safe_mask]
    t[~safe_mask] = np.inf
    
    result = np.abs(t)**2
    return result

def calculate_normalized_transmission(omega_array, mu_r_array):
    """正規化された透過率を計算するヘルパー関数"""
    transmission = calculate_transmission_intensity(omega_array, mu_r_array)
    min_trans = np.min(transmission)
    max_trans = np.max(transmission)
    normalized_transmission = (transmission - min_trans) / (max_trans - min_trans)
    return normalized_transmission

# --- 3. PyMCと連携するためのOpクラス（マルチ磁場対応） ---
class MultiFieldPhysicsModelOp(Op):
    """複数の磁場条件での物理モデルを同時に計算するOp"""
    itypes = [pt.dscalar, pt.dvector, pt.dscalar, pt.dscalar, pt.dscalar] # a, gamma_array, g_factor, B4, B6
    otypes = [pt.dvector] # 出力は全磁場の連結されたT(ω)

    def __init__(self, omega_arrays, T_val, B_values, model_type, n_transitions):
        """
        omega_arrays: 各磁場条件での周波数配列のリスト
        T_val: 温度
        B_values: 磁場値のリスト
        model_type: 'H_form' または 'B_form'
        n_transitions: 遷移数
        """
        self.omega_arrays = omega_arrays
        self.T = T_val
        self.B_values = B_values
        self.model_type = model_type
        self.n_transitions = n_transitions
        self.total_length = sum(len(omega_array) for omega_array in omega_arrays)
        
        # 物理定数を設定
        self.eps_bg = eps_bg
        self.d = d

    def perform(self, node, inputs, output_storage):
        a, gamma_array, g_factor, B4_val, B6_val = inputs

        G0 = a * mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
        
        # 各磁場での透過率を計算
        all_transmissions = []
        
        for i, (omega_array, B_val) in enumerate(zip(self.omega_arrays, self.B_values)):
            H_B = get_hamiltonian(B_val, g_factor, B4_val, B6_val)
            
            # 磁場依存性を考慮したgamma調整
            # 高磁場では線幅が広がる傾向を考慮
            B_ref = 5.0  # 基準磁場 [T]
            gamma_field_factor = 1.0 + 0.15 * (B_val - B_ref) / B_ref  # 磁場依存補正
            gamma_adjusted = gamma_array * gamma_field_factor
            
            chi_B_raw = calculate_susceptibility(omega_array, H_B, self.T, gamma_adjusted)
            
            # スケーリング係数は既にG0に含まれている
            chi_B = G0 * chi_B_raw
            
            # モデルタイプに応じてmu_rを計算
            if self.model_type == 'H_form':
                # H = μ₀(H + M) の関係から μᵣ = 1 + χ
                mu_r_B = 1 + chi_B
            elif self.model_type == 'B_form':
                # B = μ₀μᵣH の関係から μᵣ = 1/(1-χ)
                epsilon = 1e-12  # 数値安定性のため
                denominator = 1 - chi_B
                safe_mask = np.abs(denominator) > epsilon
                mu_r_B = np.ones_like(chi_B, dtype=complex)
                mu_r_B[safe_mask] = 1.0 / denominator[safe_mask]
                mu_r_B[~safe_mask] = 1e6  # 発散を避ける
            else:
                raise ValueError("Unknown model_type")
                
            # 絶対透過率 T(B) を計算
            T_B = calculate_normalized_transmission(omega_array, mu_r_B)
            all_transmissions.append(T_B)
        
        # 全ての透過率データを連結
        concatenated_transmission = np.concatenate(all_transmissions)
        output_storage[0][0] = concatenated_transmission

    def make_node(self, *inputs):
        """計算グラフのノードを作成するメソッド"""
        inputs = [pt.as_tensor_variable(inp) for inp in inputs]
        outputs = [pt.vector(dtype='float64', shape=(self.total_length,))]
        return Apply(self, inputs, outputs)

# --- 4. データ読み込み関数 ---
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
def analyze_physics_parameters(trace, model_name):
    """最適化されたパラメータの物理的意味を検証"""
    print(f"\n=== {model_name} マルチ磁場ベイズ最適化によるパラメータ分析 ===")
    a_mean = trace.posterior['a'].mean().item()
    print(f"スケーリング係数 a: {a_mean:.3f}")

    g_mean = trace.posterior['g_factor'].mean().item()
    print(f"g因子: {g_mean:.3f} (理論値: ~2.0)")
    
    # B4とB6パラメータの分析
    B4_mean = trace.posterior['B4'].mean().item()
    B6_mean = trace.posterior['B6'].mean().item()
    print(f"結晶場パラメータ B4: {B4_mean:.6f} (初期値: {B4_init:.6f})")
    print(f"結晶場パラメータ B6: {B6_mean:.6f} (初期値: {B6_init:.6f})")

    G0_mean = a_mean * mu0 * N_spin * (g_mean * muB)**2 / (2 * hbar)
    print(f"G0: {G0_mean:.3e}")

    # gamma配列の適切な処理
    gamma_posterior = trace.posterior['gamma']  # (chain, draw, transition)
    
    # 各遷移ごとの統計を計算
    gamma_means = gamma_posterior.mean(dim=['chain', 'draw']).values
    gamma_stds = gamma_posterior.std(dim=['chain', 'draw']).values
    
    print(f"gamma配列統計:")
    print(f"  全体平均: {np.mean(gamma_means):.3e}")
    print(f"  全体標準偏差: {np.mean(gamma_stds):.3e}")
    
    for i, (mean_val, std_val) in enumerate(zip(gamma_means, gamma_stds)):
        print(f"  gamma[{i}]: {mean_val:.3e} ± {std_val:.3e}")
    
    nu_mean = trace.posterior['nu'].mean().item()
    sigma_mean = trace.posterior['sigma'].mean().item()
    print(f"Student-t nu: {nu_mean:.3f}, sigma: {sigma_mean:.6f}")

# --- 6. 診断プロット関数 ---
def create_diagnostic_plots(traces):
    """診断プロットを作成する関数"""
    try:
        # トレースプロット
        fig1, axes1 = plt.subplots(4, 2, figsize=(12, 16))
        
        if 'H_form' in traces:
            az.plot_trace(traces['H_form'], var_names=['g_factor', 'a', 'B4', 'B6'], axes=axes1)
            fig1.suptitle('H_form モデル トレースプロット', fontsize=14)
            plt.tight_layout()
            plt.savefig(IMAGE_DIR / 'multi_field_fixed_trace_H_form.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig1)
        
        if 'B_form' in traces:
            fig2, axes2 = plt.subplots(4, 2, figsize=(12, 16))
            az.plot_trace(traces['B_form'], var_names=['g_factor', 'a', 'B4', 'B6'], axes=axes2)
            fig2.suptitle('B_form モデル トレースプロット', fontsize=14)
            plt.tight_layout()
            plt.savefig(IMAGE_DIR / 'multi_field_fixed_trace_B_form.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig2)
        
        # フォレストプロット:失敗中
        if len(traces) > 1:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            # トレース内の実際の変数名を使用
            try:
                # トレース内の利用可能な変数をチェック
                first_trace = list(traces.values())[0]
                available_vars = list(first_trace.posterior.data_vars.keys())
                plot_vars = [var for var in ['g_factor', 'a', 'B4', 'B6'] if var in available_vars]

                if plot_vars:
                    idata_dict = {k: v for k, v in traces.items()}
                    az.plot_forest(idata_dict, var_names=plot_vars, ax=ax3)
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
            plt.savefig(IMAGE_DIR / 'multi_field_fixed_forest_plot.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig3)
        
        # エネルギープロット
        """
        エネルギープロットとは？: 
        このプロットは、サンプリングが確率分布の全体を効率的に探索できているかを評価するのに役立ちます。エネルギー遷移分布と周辺エネルギー分布という2つの分布を重ねて表示し、両者が大きく乖離している場合、サンプリングに問題がある可能性を示唆します。
        """
        fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        if 'H_form' in traces:
            az.plot_energy(traces['H_form'], ax=axes4[0])
            axes4[0].set_title('H_form モデル')
        else:
            axes4[0].axis('off')

        if 'B_form' in traces:
            az.plot_energy(traces['B_form'], ax=axes4[1])
            axes4[1].set_title('B_form モデル')
        else:
            axes4[1].axis('off')
            
        fig4.suptitle('エネルギープロット', fontsize=16)
        plt.savefig(IMAGE_DIR / 'multi_field_fixed_energy_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig4)
        
        print("診断プロットが正常に作成されました。")
        return True
        
    except Exception as e:
        print(f"診断プロット作成中にエラー: {e}")
        import traceback
        print(traceback.format_exc())
        return None

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
                B4_sample = float(trace.posterior['B4'].isel(chain=chain_idx, draw=draw_idx))
                B6_sample = float(trace.posterior['B6'].isel(chain=chain_idx, draw=draw_idx))
                
                # このサンプルでの予測を計算
                H_sample = get_hamiltonian(B_ext_z=b_val, g_factor=g_factor_sample,
                                         B4_val=B4_sample, B6_val=B6_sample)
                G0_sample = a_sample * mu0 * N_spin * (g_factor_sample * muB)**2 / (2 * hbar)
                chi_sample_raw = calculate_susceptibility(data['omega_filtered'], H_sample, T=1.5, 
                                                        gamma_array=gamma_sample)
                chi_sample = G0_sample * chi_sample_raw
                
                if mt == 'H_form':
                    mu_r_sample = 1 + chi_sample
                else: 
                    mu_r_sample = 1 / (1-chi_sample)
                
                prediction_sample = calculate_normalized_transmission(data['omega_filtered'], mu_r_sample)
                predictions.append(prediction_sample)
            
            predictions = np.array(predictions)
            
            # 95%信用区間を計算
            mean_prediction = np.mean(predictions, axis=0)
            ci_lower = np.percentile(predictions, 2.5, axis=0)
            ci_upper = np.percentile(predictions, 97.5, axis=0)
            
            # 信用区間をプロット
            axes1[i].fill_between(data['frequency_filtered'], ci_lower, ci_upper, 
                                 alpha=0.3, color=colors[mt], label=f'95%信用区間 ({mt})')
            
            # 平均予測をプロット(ベイズ推定)
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
        plt.savefig(IMAGE_DIR / f'multi_field_credible_intervals_{mt}_fitting.png', dpi=300, bbox_inches='tight')
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
                B4_sample = float(trace.posterior['B4'].isel(chain=chain_idx, draw=draw_idx))
                B6_sample = float(trace.posterior['B6'].isel(chain=chain_idx, draw=draw_idx))
                
                # このサンプルでの予測を計算
                H_sample = get_hamiltonian(B_ext_z=b_val, g_factor=g_factor_sample,
                                         B4_val=B4_sample, B6_val=B6_sample)
                G0_sample = a_sample * mu0 * N_spin * (g_factor_sample * muB)**2 / (2 * hbar)
                chi_sample_raw_full = calculate_susceptibility(omega_plot_full, H_sample, T=1.5, 
                                                             gamma_array=gamma_sample)
                chi_sample_full = G0_sample * chi_sample_raw_full
                
                if mt == 'H_form':
                    mu_r_sample_full = 1 + chi_sample_full
                else: 
                    mu_r_sample_full = 1 / (1-chi_sample_full)
                
                prediction_sample_full = calculate_normalized_transmission(omega_plot_full, mu_r_sample_full)
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
        plt.savefig(IMAGE_DIR / f'multi_field_credible_intervals_{mt}_full.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig2)
    
    print("95%信用区間プロットが正常に作成されました。")

def plot_multi_field_results(multi_field_data, best_params, model_types, colors):
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
            if mt in best_params:
                params = best_params[mt]
                
                # ベストフィット曲線の計算
                H_best = get_hamiltonian(B_ext_z=b_val, g_factor=params['g_factor_mean'], 
                                       B4_val=params['B4_mean'], B6_val=params['B6_mean'])
                G0_best = params['a_mean'] * mu0 * N_spin * (params['g_factor_mean'] * muB)**2 / (2 * hbar)
                chi_best_raw = calculate_susceptibility(data['omega_filtered'], H_best, T=1.5, 
                                                      gamma_array=params['gamma_mean'])
                chi_best = G0_best * chi_best_raw
                
                if mt == 'H_form':
                    mu_r_best = 1 + chi_best
                else: 
                    mu_r_best = 1 / (1-chi_best)
                
                best_fit_prediction = calculate_normalized_transmission(data['omega_filtered'], 
                                                                     mu_r_best)
                
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
    plt.savefig(IMAGE_DIR / 'multi_field_fixed_fitting_region.png', dpi=300, bbox_inches='tight')
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
            if mt in best_params:
                params = best_params[mt]
                
                H_best = get_hamiltonian(B_ext_z=b_val, g_factor=params['g_factor_mean'],
                                       B4_val=params['B4_mean'], B6_val=params['B6_mean'])
                G0_best = params['a_mean'] * mu0 * N_spin * (params['g_factor_mean'] * muB)**2 / (2 * hbar)
                chi_best_raw_full = calculate_susceptibility(omega_plot_full, H_best, T=1.5, 
                                                           gamma_array=params['gamma_mean'])
                chi_best_full = G0_best * chi_best_raw_full
                
                if mt == 'H_form':
                    mu_r_best_full = 1 + chi_best_full
                else: 
                    mu_r_best_full = 1 / (1-chi_best_full)
                
                best_fit_prediction_full = calculate_normalized_transmission(omega_plot_full, 
                                                                           mu_r_best_full)
                
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
    plt.savefig(IMAGE_DIR / 'multi_field_fixed_full_range_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig2)

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
        physics_model = MultiFieldPhysicsModelOp(omega_arrays, T_val=1.5, B_values=sorted_b_values, 
                                                 model_type=mt, n_transitions=n_transitions)
        
        # モデルごとに独立したモデルコンテキストを作成
        with pm.Model() as model:
            # 階層的事前分布でgammaの変動を制御
            GAMMA_SCALE = pt.constant(1e11)
            gamma_mu = pm.Normal(f'gamma_mu_{mt}', mu=np.log(gamma_init / 1e11), sigma=0.3)
            gamma_sigma = pm.HalfNormal(f'gamma_sigma_{mt}', sigma=0.3)
            log_gamma_scaled = pm.Normal(f'log_gamma_scaled_{mt}',
                                         mu=gamma_mu,
                                         sigma=gamma_sigma,
                                         shape=n_transitions)
            gamma_array = pm.Deterministic(f'gamma_{mt}', pt.exp(log_gamma_scaled) * GAMMA_SCALE)
            
            # 両モデルで同一の事前分布（公平な比較のため）
            a = pm.TruncatedNormal(f'a_{mt}', mu=a_init, sigma=0.3, lower=0.5, upper=3.0)
            g_factor = pm.TruncatedNormal(f'g_factor_{mt}', mu=g_factor_init, sigma=0.05, lower=1.85, upper=2.20)
            B4 = pm.Normal(f'B4_{mt}', mu=B4_init, sigma=0.3*abs(B4_init))
            B6 = pm.Normal(f'B6_{mt}', mu=B6_init, sigma=0.3*abs(B6_init))
            
            # Student-t分布による外れ値耐性（モデル固有の名前）
            nu = pm.Gamma(f'nu_{mt}', alpha=3, beta=0.2)
            sigma_obs = pm.HalfCauchy(f'sigma_{mt}', beta=0.5)

            # 物理モデルの予測
            mu = physics_model(a, gamma_array, g_factor, B4, B6)

            # 尤度関数
            Y_obs = pm.StudentT(f'Y_obs_{mt}', 
                       nu=nu,
                       mu=mu, 
                       sigma=sigma_obs, 
                       observed=concatenated_transmittance)
            
            # パラメータ名をモデル後で修正するため、エイリアスを作成
            pm.Deterministic('a', a)
            pm.Deterministic('gamma', gamma_array)
            pm.Deterministic('g_factor', g_factor)
            pm.Deterministic('B4', B4)
            pm.Deterministic('B6', B6)
            pm.Deterministic('nu', nu)
            pm.Deterministic('sigma', sigma_obs)
            
            traces[mt] = pm.sample(
                2000,  
                tune=2000, 
                target_accept=0.9,  
                chains=4, 
                cores=4, 
                random_seed=42 + hash(mt) % 1000,  # モデルごとに異なるシード
                init='adapt_diag',
                idata_kwargs={"log_likelihood": True},
                nuts={
                    "max_treedepth": 15,  # より深い探索
                },
                compute_convergence_checks=True
            )
            
            ppcs[mt] = pm.sample_posterior_predictive(traces[mt], random_seed=42 + hash(mt) % 1000)
        
        print(f"--- [{mt}] マルチ磁場モデルのサンプリング完了 ---")
        print(az.summary(traces[mt], var_names=['a', 'gamma', 'g_factor', 'B4', 'B6', 'sigma']))

    # --- 5. モデル比較の結果表示 ---
    print("\n--- マルチ磁場ベイズ的モデル比較 (LOO-CV) ---")
    idata_dict = {k: v for k, v in traces.items()} 
    compare_df = az.compare(idata_dict)
    print(compare_df)
    
    # モデル比較の詳細分析
    print("\n=== モデル比較詳細分析 ===")
    try:
        # ELPD差分の直接取得を試みる
        if len(compare_df) >= 2:
            first_model_elpd = compare_df.iloc[0]['elpd_loo']
            second_model_elpd = compare_df.iloc[1]['elpd_loo'] 
            elpd_diff_calc = second_model_elpd - first_model_elpd
            print(f"計算されたELPD差分: {elpd_diff_calc:.6f}")
            
            # 差分の解釈
            if abs(elpd_diff_calc) < 1.0:
                print("⚠️  ELPD差分が1.0未満です。モデル間の予測性能に明確な差はありません。")
                print("   これは以下の原因が考えられます：")
                print("   1. 両モデルが実質的に同じ物理現象を記述している")
                print("   2. データ量が不十分でモデルの違いを捉えられない")
                print("   3. モデル間の物理的差異が小さい")
            elif abs(elpd_diff_calc) < 2.0:
                print("📊 ELPD差分が小さく、モデル間の性能差は軽微です。")
            else:
                print("✅ ELPD差分が2.0以上で、明確なモデル選択が可能です。")
            
        # 各モデルの基本統計
        print("\n各モデルの詳細:")
        for i, model_name in enumerate(compare_df.index):
            rank = compare_df.iloc[i]['rank']
            elpd = compare_df.iloc[i]['elpd_loo']
            se = compare_df.iloc[i]['se']
            print(f"  {model_name}モデル (rank={rank}): ELPD = {elpd:.3f} ± {se:.3f}")
            
    except Exception as e:
        print(f"詳細分析中にエラー: {e}")
        print("基本的なモデル比較結果のみ表示されます。")
    
    try:
        axes = az.plot_compare(compare_df, figsize=(8, 4))
        fig = axes.ravel()[0].figure if hasattr(axes, "ravel") else axes.figure
        fig.suptitle('マルチ磁場モデル比較', fontsize=16)
        fig.tight_layout()
        plt.savefig(IMAGE_DIR / 'multi_field_fixed_model_comparison.png', dpi=150)
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
        analyze_physics_parameters(trace, mt)
        
        # ベストフィットパラメータの抽出
        a_mean = trace.posterior['a'].mean().item()
        gamma_mean = trace.posterior['gamma'].mean(dim=['chain', 'draw']).values
        g_factor_mean = trace.posterior['g_factor'].mean().item()
        B4_mean = trace.posterior['B4'].mean().item()
        B6_mean = trace.posterior['B6'].mean().item()
        
        best_params[mt] = {
            'a_mean': a_mean,
            'gamma_mean': gamma_mean,
            'g_factor_mean': g_factor_mean,
            'B4_mean': B4_mean,
            'B6_mean': B6_mean
        }

    # マルチ磁場結果のプロット
    plot_multi_field_results(multi_field_data, best_params, model_types, colors)

    # --- 7. フィッティング品質の評価 ---
    print("\n=== マルチ磁場フィッティング品質の評価 ===")
    
    for mt in model_types:
        trace = traces[mt]
        ppc = ppcs[mt]
        
        # 正しい変数名を取得
        ppc_var_name = f'Y_obs_{mt}'
        if ppc_var_name in ppc.posterior_predictive:
            y_pred_mean = ppc.posterior_predictive[ppc_var_name].mean(dim=['chain', 'draw']).values
        else:
            # フォールバック: 利用可能な変数名を確認
            available_vars = list(ppc.posterior_predictive.data_vars.keys())
            print(f"利用可能な変数: {available_vars}")
            # Y_obsで始まる変数を探す
            y_obs_vars = [var for var in available_vars if var.startswith('Y_obs')]
            if y_obs_vars:
                y_pred_mean = ppc.posterior_predictive[y_obs_vars[0]].mean(dim=['chain', 'draw']).values
            else:
                print(f"警告: {mt} モデルの予測変数が見つかりません。")
                continue
                
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
        create_diagnostic_plots(traces)
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
    print("\n=== 結果ファイル ===")
    print("- multi_field_fixed_fitting_region.png: フィッティング結果")
    print("- multi_field_fixed_full_range_prediction.png: 全領域予測")
    print("- multi_field_fixed_model_comparison.png: モデル比較")
    print("- multi_field_fixed_trace_H_form.png: H_formトレース")
    print("- multi_field_fixed_trace_B_form.png: B_formトレース")
    print("- multi_field_fixed_forest_plot.png: パラメータ比較")
    print("- multi_field_fixed_energy_H_form.png: エネルギープロット")
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
        print(f"  スケーリング係数: {params['a_mean']:.3f}")
        print(f"  結晶場パラメータ B4: {params['B4_mean']:.6f}")
        print(f"  結晶場パラメータ B6: {params['B6_mean']:.6f}")

    print("\n信用区間プロットについて:")
    print("- 95%信用区間は、パラメータの不確実性を考慮した予測の範囲を示します")
    print("- 塗りつぶし領域は、95%の確率でデータが存在する範囲です")
    print("- 実線は、すべてのサンプルからの平均予測を表します")
    print("- フィッティング領域と全領域の両方で信用区間が計算されます")

    print("\n注意事項:")
    print("- divergencesが発生している →→ サンプリング品質に注意してください。")
    print("- R-hat > 1.01 のパラメータがある →→ 収束を確認してください。")
    print("- ESS < 100 のパラメータがある →→ より多くのサンプルが必要です。")
