import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# --- 1. 物理定数とパラメータ定義 ---
kB = 1.380649e-23; muB = 9.274010e-24; hbar = 1.054571e-34; c = 299792458; mu0 = 4.0 * np.pi * 1e-7
s = 3.5
N_spin = 24/1.238 * 1e27 #GGGのスピン数密度
"""
手動計算で分かっている最良の値を初期値とする
"""
d_init = 157.8e-6 * 0.99
eps_bg_init = 13.1404
#✅ g_factor_init, B4_init, B6_initは要更新
g_factor_init = 1.95 
B4_init = 0.8 / 240 * 0.606; B6_init = 0.04 / 5040 * -1.513
gamma_init = 0.11e12

# --- 2. 汎用化された物理モデル関数 ---

def get_hamiltonian(B_ext_z, B4, B6, g_factor):
    m_values = np.arange(s, -s - 1, -1)
    Sz = np.diag(m_values)
    O04 = 60*np.diag([7,-13,-3,9,9,-3,-13,7]) 
    X_O44 = np.zeros((8,8)); X_O44[3,7],X_O44[4,0]=np.sqrt(35),np.sqrt(35); X_O44[2,6],X_O44[5,1]=5*np.sqrt(3),5*np.sqrt(3); O44=12*(X_O44+X_O44.T)
    O06 = 1260*np.diag([1,-5,9,-5,-5,9,-5,1])
    X_O46 = np.zeros((8,8)); X_O46[3,7],X_O46[4,0]=3*np.sqrt(35),3*np.sqrt(35); X_O46[2,6],X_O46[5,1]=-7*np.sqrt(3),-7*np.sqrt(3); O46=60*(X_O46+X_O46.T)
    H_cf = (B4 * kB) * (O04 + 5 * O44) + (B6 * kB) * (O06 - 21 * O46)
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

def calculate_transmission_intensity(omega_array, mu_r_array, eps_bg, d):
    """強度|t|^2を計算するヘルパー関数（数値安定性を改善）"""
    eps_bg = np.clip(eps_bg, 1, 30)  # eps_bgの範囲を制限
    
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

def calculate_normalized_transmission(omega_array, mu_r_array, eps_bg, d):
    """正規化された透過率を計算するヘルパー関数"""
    transmission = calculate_transmission_intensity(omega_array, mu_r_array, eps_bg, d)
    min_trans = np.min(transmission)
    max_trans = np.max(transmission)
    normalized_transmission = (transmission - min_trans) / (max_trans - min_trans)
    return normalized_transmission

# --- 3. PyMCと連携するためのOpクラス ---
class PhysicsModelOp(Op):
    itypes = [pt.dvector, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar] # gamma_array, eps_bg, d, B4, B6, g_factor
    otypes = [pt.dvector] # 出力はT(ω)

    def __init__(self, omega_array, T_val, B_val, model_type, n_transitions):
        self.omega_array = omega_array
        self.T = T_val
        self.B = B_val
        self.model_type = model_type
        self.n_transitions = n_transitions

    def perform(self, node, inputs, output_storage):
        gamma_array, eps_bg, d, B4, B6, g_factor = inputs

        H_B = get_hamiltonian(self.B, B4, B6, g_factor)
        G0 = mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
        
        chi_B_raw = calculate_susceptibility(self.omega_array, H_B, self.T, gamma_array)
        chi_B = G0 * chi_B_raw
        
        # モデルタイプに応じてmu_rを計算
        if self.model_type == 'H_form':
            mu_r_B = 1 + chi_B
        elif self.model_type == 'B_form':
            mu_r_B = np.divide(1, 1 - chi_B, where=(1 - chi_B)!=0, out=np.full_like(chi_B, np.inf, dtype=complex))
        else:
            raise ValueError("Unknown model_type")
            
        # 絶対透過率 T(B) を計算
        T_B = calculate_normalized_transmission(self.omega_array, mu_r_B, eps_bg, d)
        output_storage[0][0] = T_B

    def make_node(self, *inputs):
        """
        計算グラフのノードを作成するメソッド
        """
        inputs = [pt.as_tensor_variable(inp) for inp in inputs]
        output_length = len(self.omega_array)
        outputs = [pt.vector(dtype='float64', shape=(output_length,))]
        return Apply(self, inputs, outputs)

# --- 4. パラメータ分析関数 ---
def analyze_physics_parameters(trace, model_name):
    """最適化されたパラメータの物理的意味を検証"""
    print(f"\n=== {model_name} ベイズ最適化によるパラメータ分析 ===")

    g_mean = trace.posterior['g_factor'].mean().item()
    print(f"g因子: {g_mean:.3f} (理論値: ~2.0)")
    
    d_mean = trace.posterior['d'].mean().item()
    print(f"試料厚さ: {d_mean*1e6:.1f} μm")
    
    B4_mean = trace.posterior['B4'].mean().item()
    B6_mean = trace.posterior['B6'].mean().item()
    print(f"B4: {B4_mean:.6f}, B6: {B6_mean:.6f}")

    # 'a'パラメータが存在しないため、G0の計算を直接行う
    G0_mean = mu0 * N_spin * (g_mean * muB)**2 / (2 * hbar)
    print(f"G0: {G0_mean:.3e}")

    # gamma配列の処理を修正
    gamma_array = trace.posterior['gamma'].values  # numpy配列として取得
    if gamma_array.ndim > 1:
        gamma_mean = np.mean(gamma_array)  # 全体の平均
        gamma_std = np.std(gamma_array)    # 標準偏差
        print(f"gamma(平均): {gamma_mean:.3e} ± {gamma_std:.3e}")
        
        # 各遷移の平均値も表示
        gamma_per_transition = np.mean(gamma_array, axis=(0, 1))  # chain, drawの軸で平均
        for i, gamma_val in enumerate(gamma_per_transition):
            print(f"  gamma[{i}]: {gamma_val:.3e}")
    else:
        print(f"gamma: {gamma_array:.3e}")
    
    eps_bg_mean = trace.posterior['eps_bg'].mean().item()
    print(f"eps_bg: {eps_bg_mean:.3f}")
    
    nu_mean = trace.posterior['nu'].mean().item()
    sigma_mean = trace.posterior['sigma'].mean().item()
    print(f"Student-t nu: {nu_mean:.3f}, sigma: {sigma_mean:.6f}")

# --- 5. 診断プロット関数 ---
def create_diagnostic_plots(traces):
    """診断プロットを作成する関数"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        if 'H_form' in traces:
            az.plot_trace(traces['H_form'], var_names=['sigma'], axes=axes[0, :2])
            axes[0, 2].axis('off')
        
        if len(traces) > 1:
            az.plot_forest(traces, var_names=['g_factor', 'eps_bg'], ax=axes[1, 0])
        
        if 'H_form' in traces:
            az.plot_pair(traces['H_form'], var_names=['g_factor'], ax=axes[1, 1:])
        
        plt.tight_layout()
        plt.savefig('diagnostic_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        return fig
    except Exception as e:
        print(f"診断プロット作成中にエラー: {e}")
        return None

def analyze_residuals(y_true, y_pred, freq):
    """残差分析を行う関数"""
    try:
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].scatter(freq, residuals, alpha=0.7)
        axes[0].set_xlabel('周波数 (THz)')
        axes[0].set_ylabel('残差')
        axes[0].axhline(y=0, color='red', linestyle='--')
        axes[0].set_title('残差 vs 周波数')
        
        axes[1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('残差')
        axes[1].set_ylabel('頻度')
        axes[1].set_title('残差の分布')
        
        try:
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[2])
            axes[2].set_title('Q-Q プロット')
        except ImportError:
            axes[2].text(0.5, 0.5, 'scipy が必要です', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Q-Q プロット (利用不可)')
        
        plt.tight_layout()
        plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        return fig
    except Exception as e:
        print(f"残差分析中にエラー: {e}")
        return None

# --- 6. メイン実行ブロック ---
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
    print("実験データを読み込みます...")
    file_path = "C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\Circular_Polarization_B_Field.xlsx"
    sheet_name = 'Sheet2'
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
        print(f"データの読み込みに成功しました。読み込み件数: {len(df)}件")
        
        # 全領域のデータを保存
        exp_freq_thz_full = df['Frequency (THz)'].to_numpy(dtype=float)
        exp_transmittance_full = df['Transmittance (7.7T)'].to_numpy(dtype=float)
        
        # フィルタリング
        df_filtered = df[df['Frequency (THz)'] <= 0.376].copy()
        print(f"フィルタリング後のデータ件数: {len(df_filtered)}件")
        exp_freq_thz = df_filtered['Frequency (THz)'].to_numpy(dtype=float)
        exp_transmittance = df_filtered['Transmittance (7.7T)'].to_numpy(dtype=float)
        
        # 正規化（フィルタリングしたデータの範囲で）
        min_exp, max_exp = np.min(exp_transmittance), np.max(exp_transmittance)
        exp_transmittance_normalized = (exp_transmittance - min_exp) / (max_exp - min_exp)
        
        # 全領域のデータを同じ正規化係数で正規化
        min_exp_full, max_exp_full = np.min(exp_transmittance_full), np.max(exp_transmittance_full)
        exp_transmittance_normalized_full = (exp_transmittance_full - min_exp_full) / (max_exp_full - min_exp_full)

    except Exception as e: 
        print(f"データ読み込み中にエラー: {e}")
        exit()

    exp_omega_rad_s = exp_freq_thz * 1e12 * 2 * np.pi
    print_memory_usage("データ読み込み後")

    # --- モデル比較の準備 ---
    model_types = ['H_form', 'B_form']
    traces = {}
    ppcs = {}
    n_transitions = 7

    # --- 各モデルでサンプリングを実行 ---
    for mt in model_types:
        print(f"\n--- [{mt}] 改善モデルのサンプリングを開始します ---")
        physics_model = PhysicsModelOp(exp_omega_rad_s, T_val=35.0, B_val=7.8, model_type=mt, n_transitions=n_transitions)
        
        with pm.Model() as model:
            # 事前分布の設定            
            # gammaの事前分布をトランケートして0以上に制限
            log_gamma_sigma = pm.HalfNormal('log_gamma_sigma', sigma=1.0)
            log_gamma_array = pm.Normal('log_gamma', mu=np.log(gamma_init), sigma=log_gamma_sigma, shape=n_transitions)
            gamma_array = pm.Deterministic('gamma', pt.exp(log_gamma_array))

            # gamma_sigma = pm.HalfNormal('gamma_sigma', sigma=5e10)
            # gamma_array = pm.TruncatedNormal('gamma', mu=gamma_init, sigma=gamma_sigma, lower=0, shape=n_transitions)
            
            # 物理パラメータの事前分布
            eps_bg = pm.TruncatedNormal('eps_bg', mu=eps_bg_init, sigma=2.0, lower=11.0, upper=16.0)
            d = pm.TruncatedNormal('d', mu=d_init, sigma=20e-6, lower=100e-6, upper=200e-6)
            B4 = pm.Normal('B4', mu=B4_init, sigma=abs(B4_init)*0.5)
            B6 = pm.Normal('B6', mu=B6_init, sigma=abs(B6_init)*0.5)
            g_factor = pm.TruncatedNormal('g_factor', mu=g_factor_init, sigma=0.1, lower=1.8, upper=2.3)
            
            # Student-t分布による外れ値耐性
            nu = pm.Gamma('nu', alpha=2, beta=0.1)  # 自由度パラメータ
            sigma_obs = pm.HalfCauchy('sigma', beta=0.5)  # より保守的

            # 物理モデルの予測
            mu = physics_model(gamma_array, eps_bg, d, B4, B6, g_factor)

            # ロバストな尤度関数
            Y_obs = pm.StudentT('Y_obs', 
                           nu=nu,
                           mu=mu, 
                           sigma=sigma_obs, 
                           observed=exp_transmittance_normalized)            
            traces[mt] = pm.sample(
                2000, 
                tune=2000, 
                target_accept=0.9,
                chains=4, 
                cores=4, 
                random_seed=42, 
                idata_kwargs={"log_likelihood": True}
            )
            
            ppcs[mt] = pm.sample_posterior_predictive(traces[mt], random_seed=42)
        
        print(f"--- [{mt}] モデルのサンプリング完了 ---")
        print(az.summary(traces[mt], var_names=['eps_bg', 'd', 'B4', 'B6', 'g_factor', 'sigma']))

    # --- 5. モデル比較の結果表示 ---
    print("\n--- ベイズ的モデル比較 (LOO-CV) ---")
    idata_dict = {k: v for k, v in traces.items()} 
    compare_df = az.compare(idata_dict)
    print(compare_df)
    
    try:
        axes = az.plot_compare(compare_df, figsize=(8, 4))
        fig = axes.ravel()[0].figure if hasattr(axes, "ravel") else axes.figure
        fig.suptitle('モデル比較', fontsize=16)
        fig.tight_layout()
        plt.savefig('model_comparison_test.png', dpi=300)
        plt.show()
        plt.close()
    except Exception as e:
        print(f"モデル比較プロット中にエラー: {e}")

    # --- 6. ベストフィット曲線のプロット（修正版：別々のファイル出力） ---    
    colors = {'H_form': 'blue', 'B_form': 'red'}

    freq_thz_plot_full = np.linspace(np.min(exp_freq_thz_full), np.max(exp_freq_thz_full), 1000)
    omega_rad_s_plot_full = freq_thz_plot_full * 1e12 * 2 * np.pi

    freq_thz_plot_fit = np.linspace(np.min(exp_freq_thz), np.max(exp_freq_thz), 500)
    omega_rad_s_plot_fit = freq_thz_plot_fit * 1e12 * 2 * np.pi

    # パラメータの計算（一度だけ実行）
    best_params = {}
    best_predictions_full = {}
    best_predictions_fit = {}
    
    for mt in model_types:
        trace = traces[mt]; ppc = ppcs[mt]
        analyze_physics_parameters(trace, mt)
        
        # ベストフィット曲線の計算
        gamma_mean = trace.posterior['gamma'].mean(dim=['chain', 'draw']).values
        eps_bg_mean = trace.posterior['eps_bg'].mean().item()
        d_mean = trace.posterior['d'].mean().item()
        B4_mean = trace.posterior['B4'].mean().item()
        B6_mean = trace.posterior['B6'].mean().item()
        g_factor_mean = trace.posterior['g_factor'].mean().item()
        
        best_params[mt] = {
            'gamma_mean': gamma_mean,
            'eps_bg_mean': eps_bg_mean,
            'd_mean': d_mean,
            'B4_mean': B4_mean,
            'B6_mean': B6_mean,
            'g_factor_mean': g_factor_mean
        }
        
        # 全領域での予測
        H_best = get_hamiltonian(B_ext_z=7.7, B4=B4_mean, B6=B6_mean, g_factor=g_factor_mean)
        G0_best = mu0 * N_spin * (g_factor_mean * muB)**2 / (2 * hbar)
        chi_best_raw_full = calculate_susceptibility(omega_rad_s_plot_full, H_best, T=35.0, gamma_array=gamma_mean)
        chi_best_full = G0_best * chi_best_raw_full
        
        if mt == 'H_form':
            mu_r_best_full = 1 + chi_best_full
        else: 
            mu_r_best_full = 1 / (1-chi_best_full)
        
        best_fit_prediction_full = calculate_normalized_transmission(omega_rad_s_plot_full, mu_r_best_full, eps_bg_mean, d_mean)
        best_predictions_full[mt] = best_fit_prediction_full
        
        # フィッティング領域での予測
        chi_best_raw_fit = calculate_susceptibility(omega_rad_s_plot_fit, H_best, T=35.0, gamma_array=gamma_mean)
        chi_best_fit = G0_best * chi_best_raw_fit

        if mt == 'H_form':
            mu_r_best_fit = 1 + chi_best_fit
        else: 
            mu_r_best_fit = 1 / (1-chi_best_fit)
        
        best_fit_prediction_fit = calculate_normalized_transmission(omega_rad_s_plot_fit, mu_r_best_fit, eps_bg_mean, d_mean)
        best_predictions_fit[mt] = best_fit_prediction_fit

    # ① フィッティング領域のプロット
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    
    # フィッティング領域データをプロット
    ax1.scatter(exp_freq_thz, exp_transmittance_normalized, 
               alpha=0.8, s=30, color='black', label='フィッティング領域データ')
    
    # フィッティング領域のベストフィット曲線
    for mt in model_types:
        ax1.plot(freq_thz_plot_fit, best_predictions_fit[mt], 
                color=colors[mt], lw=3, label=f'ベストフィット ({mt})')
    
    ax1.set_xlabel('周波数 (THz)')
    ax1.set_ylabel('正規化透過率')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('ベイズ最適化結果: フィッティング領域', fontsize=14)
    ax1.set_xlim(np.min(exp_freq_thz), np.max(exp_freq_thz))
    ax1.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('fitting_region_only.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig1)

    # ② 全領域のプロット
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    
    # 全領域データをプロット
    ax2.scatter(exp_freq_thz_full, exp_transmittance_normalized_full, 
               alpha=0.6, s=20, color='gray', label='実験データ（全領域）')
    
    # 全領域のベストフィット曲線
    for mt in model_types:
        ax2.plot(freq_thz_plot_full, best_predictions_full[mt], 
                color=colors[mt], lw=2, label=f'ベストフィット 全領域 ({mt})')
    
    # フィッティング領域境界を表示
    ax2.axvline(x=0.376, color='red', linestyle=':', alpha=0.7, label='フィッティング領域上限')
    
    ax2.set_xlabel('周波数 (THz)')
    ax2.set_ylabel('正規化透過率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('ベイズ最適化結果: 全領域予測', fontsize=14)
    ax2.set_xlim(np.min(exp_freq_thz_full), np.max(exp_freq_thz_full))
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('full_range_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig2)

    # --- 7. フィッティング品質の評価 ---
    print("\n=== フィッティング品質の評価 ===")
    for mt in model_types:
        trace = traces[mt]
        ppc = ppcs[mt]
        y_pred_mean = ppc.posterior_predictive['Y_obs'].mean(dim=['chain', 'draw']).values
        rmse_fit = np.sqrt(np.mean((exp_transmittance_normalized - y_pred_mean)**2))
        
        print(f"\n{mt} モデル:")
        print(f"  フィッティング領域 RMSE: {rmse_fit:.6f}")

    # --- 8. 診断プロットと残差分析 ---
    try:
        print("\n=== 診断プロット作成中 ===")
        create_diagnostic_plots(traces)
        
        best_model = compare_df.index[0]
        if best_model in ppcs:
            print(f"\n=== {best_model} モデルの残差分析 ===")
            ppc_best = ppcs[best_model]
            y_pred_mean = ppc_best.posterior_predictive['Y_obs'].mean(dim=['chain', 'draw']).values
            analyze_residuals(exp_transmittance_normalized, y_pred_mean, exp_freq_thz)
        print_memory_usage("診断・残差分析後")
    except Exception as e:
        print(f"診断・残差分析中にエラー: {e}")

    print("全ての処理が完了しました。結果を確認してください。")
