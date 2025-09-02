import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pytensor.tensor as pt # NumPyの代わりにPyTensorをインポート
from pytensor.tensor import linalg as pt_linalg  # 線形代数関数を明示的にインポート
from pytensor.compile.ops import as_op  # as_opをインポート
import japanize_matplotlib 

# --- 0. プロット設定 ---
plt.rcParams['figure.dpi'] = 120

# --- 1. 物理定数とパラメータ定義 ---
kB = 1.380649e-23; muB = 9.274010e-24; hbar = 1.054571e-34; c = 299792458; mu0 = 4.0 * np.pi * 1e-7
s = 3.5
N_spin = 24/1.238 * 1e27 #GGGのスピン数密度

# 手動計算で分かっている最良の値を初期値とする（実験データに基づく調整）
d_init = 157.8e-6
eps_bg_init = 15.0    # 実験データに基づく調整
g_factor_init = 2.0   # より標準的な値
B4_init = 0.8 / 240 * 0.606 * 0.5   # より小さな初期値
B6_init = 0.04 / 5040 * -1.513 * 0.5  # より小さな初期値
gamma_init = 0.05e12  # より小さなダンピング

# --- 2. PyTensorネイティブな物理モデル関数 (pm.Model内で使用) ---
@as_op(itypes=[pt.dvector, pt.dscalar], otypes=[pt.dvector])
def calculate_transmission_complex(params, model_type_flag):
    """
    bayesian_multi_test.pyと同様の完全な複素数透過率計算
    PyTensorのas_opデコレータを使用してNumPy関数をラップ
    model_type_flag: 0.0=H_form, 1.0=B_form
    """
    # パラメータを展開 - NumPy配列として処理
    params_np = np.asarray(params)
    model_type_np = float(model_type_flag)
    a, log_gamma, eps_bg, d, B4, B6, g_factor = params_np
    gamma = np.exp(log_gamma)
    
    # ハミルトニアンの計算
    H = get_hamiltonian_np(7.7, B4, B6, g_factor)  # B_val=7.7T固定
    eigenvalues, _ = np.linalg.eigh(H)
    eigenvalues -= np.min(eigenvalues)
    
    # 統計計算
    T_val = 35.0  # 温度固定
    Z = np.sum(np.exp(-eigenvalues / (kB * T_val)))
    populations = np.exp(-eigenvalues / (kB * T_val)) / Z
    
    # エネルギー差とポピュレーション差
    delta_E = eigenvalues[1:] - eigenvalues[:-1]
    delta_pop = populations[1:] - populations[:-1]
    omega_0 = delta_E / hbar
    
    # 実験データに対応する周波数配列を作成
    n_data_points = 50  # 固定サイズ
    freq_thz = np.linspace(0.34, 0.376, n_data_points)  # フィッティング領域
    omega_array = freq_thz * 1e12 * 2 * np.pi
    
    # 遷移強度の計算
    m_vals_trans = np.arange(s, -s, -1)
    transition_strength = (s + m_vals_trans) * (s - m_vals_trans + 1)
    
    # gamma配列をdelta_Eのサイズに合わせる
    if len(delta_E) > len(transition_strength):
        transition_strength = np.pad(transition_strength, (0, len(delta_E) - len(transition_strength)), 'edge')
    elif len(delta_E) < len(transition_strength):
        transition_strength = transition_strength[:len(delta_E)]
    
    # 感受率の計算
    G0 = mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
    numerator = a * G0 * delta_pop * transition_strength
    
    # gamma配列をdelta_Eと同じサイズにする
    gamma_array = np.full(len(delta_E), gamma)
    
    # 複素感受率の計算
    denominator = (omega_0[:, np.newaxis] - omega_array) - (1j * gamma_array[:, np.newaxis])
    
    # ゼロ除算を防ぐ
    safe_denominator = np.where(np.abs(denominator) < 1e-15, 1e-15 + 0j, denominator)
    chi_B_raw = np.sum(numerator[:, np.newaxis] / safe_denominator, axis=0)
    chi_B = -chi_B_raw
    
    # モデルタイプに応じた透磁率計算
    if model_type_np < 0.5:  # H_form
        mu_r_B = 1 + chi_B
    else:  # B_form
        mu_r_B = 1 / (1 - chi_B)
    
    # 複素透過率の計算
    n_complex = np.sqrt(eps_bg * mu_r_B + 0j)
    impe = np.sqrt(mu_r_B / eps_bg + 0j)
    
    lambda_0 = (2 * np.pi * c) / omega_array
    delta = (2 * np.pi * n_complex * d) / lambda_0
    
    # 透過係数の計算
    exp_j_delta = np.exp(1j * delta)
    num_t = 4 * impe * exp_j_delta
    den_t = (1 + impe)**2 - (1 - impe)**2 * np.exp(2j * delta)
    
    # ゼロ除算を防ぐ
    safe_den_t = np.where(np.abs(den_t) < 1e-15, 1e-15 + 0j, den_t)
    t = num_t / safe_den_t
    
    # 透過率（強度）の計算
    transmission = np.abs(t)**2
    
    # 有限値チェック
    transmission = np.where(np.isfinite(transmission), transmission, 0.0)
    
    # 正規化
    min_trans = np.min(transmission)
    max_trans = np.max(transmission)
    if (max_trans - min_trans) > 1e-10:
        normalized_transmission = (transmission - min_trans) / (max_trans - min_trans)
    else:
        normalized_transmission = transmission
    
    return normalized_transmission.astype(np.float64)

# --- 3. NumPyベースの物理モデル関数 (グラフ描画用) ---
# NOTE: モデルの外で、推定結果を使って計算するためのNumPy版関数

def calculate_susceptibility(omega_array, H, T, gamma_array):
    """完全な感受率計算（bayesian_multi_test.pyと同様）"""
    # 固有値と固有ベクトルの計算
    eigenvalues, _ = np.linalg.eigh(H)
    eigenvalues -= np.min(eigenvalues)
    
    # 統計力学的な計算
    Z = np.sum(np.exp(-eigenvalues / (kB * T)))
    populations = np.exp(-eigenvalues / (kB * T)) / Z
    
    # エネルギー差とポピュレーション差
    delta_E = eigenvalues[1:] - eigenvalues[:-1]
    delta_pop = populations[1:] - populations[:-1]
    omega_0 = delta_E / hbar
    
    # 遷移強度の計算
    m_vals = np.arange(s, -s - 1, -1)
    m_vals_trans = np.arange(s, -s, -1)
    transition_strength = (s + m_vals_trans) * (s - m_vals_trans + 1)
    
    # gamma_arrayがdelta_Eと同じ次元を持つように調整
    if len(gamma_array) != len(delta_E):
        print(f"警告: gamma_arrayの長さ({len(gamma_array)})がdelta_Eの長さ({len(delta_E)})と異なります。")
        if len(gamma_array) > len(delta_E):
            gamma_array = gamma_array[:len(delta_E)]
            print(f"gamma_arrayを{len(delta_E)}要素に切り詰めました。")
        else:
            # gamma_arrayが短い場合は最後の値で埋める
            gamma_array = np.pad(gamma_array, (0, len(delta_E) - len(gamma_array)), 'edge')
            print(f"gamma_arrayを{len(delta_E)}要素に拡張しました。")
    
    # G0を動的に計算
    numerator = delta_pop * transition_strength
    denominator = (omega_0[:, np.newaxis] - omega_array) - (1j * gamma_array[:, np.newaxis])
    
    # ゼロ除算チェックを追加
    safe_denominator = np.where(np.abs(denominator) < 1e-15, 1e-15, denominator)
    chi_array = np.sum(numerator[:, np.newaxis] / safe_denominator, axis=0)
    return -chi_array

def calculate_transmission_intensity(omega_array, mu_r_array, eps_bg, d):
    """強度|t|^2を計算するヘルパー関数（数値安定性を改善）"""
    # 複素数計算の安定性を向上
    eps_bg = np.maximum(eps_bg, 1e-10)  # ゼロ除算防止
    
    n_complex = np.sqrt(eps_bg * mu_r_array + 0j)
    impe = np.sqrt(mu_r_array / eps_bg + 0j)
    
    lambda_0 = np.full_like(omega_array, np.inf, dtype=float)
    nonzero_mask = omega_array > 1e-12  # より安全な閾値
    lambda_0[nonzero_mask] = (2 * np.pi * c) / omega_array[nonzero_mask]
    
    delta = 2 * np.pi * n_complex * d / lambda_0
    
    # 指数関数の計算で数値オーバーフローを防ぐ
    exp_2j_delta = np.exp(2j * delta)
    exp_j_delta = np.exp(1j * delta)
    
    numerator = 4 * impe * exp_j_delta
    denominator = (1 + impe)**2 - (1 - impe)**2 * exp_2j_delta
    
    # より安全な除算
    safe_mask = np.abs(denominator) > 1e-12
    t = np.zeros_like(denominator, dtype=complex)
    t[safe_mask] = numerator[safe_mask] / denominator[safe_mask]
    t[~safe_mask] = np.inf
    
    result = np.abs(t)**2
    # 結果の妥当性チェック
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result

def get_hamiltonian_np(B_ext_z, B4, B6, g_factor):
    m_values = np.arange(s, -s - 1, -1)
    Sz = np.diag(m_values)
    O04 = 60*np.diag([7,-13,-3,9,9,-3,-13,7])
    X_O44 = np.zeros((8,8)); X_O44[3,7],X_O44[4,0]=np.sqrt(35),np.sqrt(35); X_O44[2,6],X_O44[5,1]=5*np.sqrt(3),5*np.sqrt(3); O44=12*(X_O44+X_O44.T)
    O06 = 1260*np.diag([1,-5,9,-5,-5,9,-5,1])
    X_O46 = np.zeros((8,8)); X_O46[3,7],X_O46[4,0]=3*np.sqrt(35),3*np.sqrt(35); X_O46[2,6],X_O46[5,1]=-7*np.sqrt(3),-7*np.sqrt(3); O46=60*(X_O46+X_O46.T)
    H_cf = (B4 * kB) * (O04 + 5 * O44) + (B6 * kB) * (O06 - 21 * O46)
    H_zee = g_factor * muB * B_ext_z * Sz
    return H_cf + H_zee

def calculate_transmission_np(params, omega_array, T_val, B_val, model_type):
    """bayesian_multi_test.pyと同様の完全な透過率計算"""
    a, log_gamma, eps_bg, d, B4, B6, g_factor = params
    gamma = np.exp(log_gamma)
    
    # ハミルトニアンの取得
    H = get_hamiltonian_np(B_val, B4, B6, g_factor)
    
    # gamma配列を適切なサイズに設定
    eigenvalues, _ = np.linalg.eigh(H)
    n_transitions = len(eigenvalues) - 1
    gamma_array = np.full(n_transitions, gamma)
    
    # 感受率の計算
    G0 = mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
    chi_raw = calculate_susceptibility(omega_array, H, T_val, gamma_array)
    chi_B = a * G0 * chi_raw
    
    # 透磁率の計算
    if model_type == 'H_form':
        mu_r_B = 1 + chi_B
    else: # B_form
        mu_r_B = 1 / (1 - chi_B)
    
    # 透過率の計算
    transmission = calculate_transmission_intensity(omega_array, mu_r_B, eps_bg, d)
    
    # 正規化
    min_trans = np.min(transmission)
    max_trans = np.max(transmission)
    if (max_trans - min_trans) > 1e-10:
        normalized_transmission = (transmission - min_trans) / (max_trans - min_trans)
    else:
        normalized_transmission = transmission
    
    return normalized_transmission


# --- 4. メイン実行ブロック ---
if __name__ == '__main__':
    # --- データの読み込み ---
    print("実験データを読み込みます...")
    file_path = "C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\Circular_Polarization_B_Field.xlsx"
    sheet_name = 'Sheet2'
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    exp_freq_thz_full = df['Frequency (THz)'].to_numpy(dtype=float)
    exp_transmittance_full = df['Transmittance (7.7T)'].to_numpy(dtype=float)
    df_filtered = df[df['Frequency (THz)'] <= 0.376].copy()
    exp_freq_thz_fit = df_filtered['Frequency (THz)'].to_numpy(dtype=float)
    exp_transmittance_fit = df_filtered['Transmittance (7.7T)'].to_numpy(dtype=float)
    
    # 関数出力サイズに合わせて実験データをリサンプリング
    function_output_size = 50  # calculate_transmission_simple関数の出力サイズと一致
    if len(exp_transmittance_fit) > function_output_size:
        # データを間引く
        indices = np.linspace(0, len(exp_transmittance_fit)-1, function_output_size, dtype=int)
        exp_freq_thz_fit = exp_freq_thz_fit[indices]
        exp_transmittance_fit = exp_transmittance_fit[indices]
    elif len(exp_transmittance_fit) < function_output_size:
        # データを補間で増やす
        from scipy.interpolate import interp1d
        f = interp1d(np.arange(len(exp_transmittance_fit)), exp_transmittance_fit, kind='linear')
        new_indices = np.linspace(0, len(exp_transmittance_fit)-1, function_output_size)
        exp_transmittance_fit = f(new_indices)
        f_freq = interp1d(np.arange(len(exp_freq_thz_fit)), exp_freq_thz_fit, kind='linear')
        exp_freq_thz_fit = f_freq(new_indices)
    
    min_exp, max_exp = np.min(exp_transmittance_fit), np.max(exp_transmittance_fit)
    exp_transmittance_norm_fit = (exp_transmittance_fit - min_exp) / (max_exp - min_exp)
    exp_transmittance_norm_full = (exp_transmittance_full - min_exp) / (max_exp - min_exp)
    exp_omega_rad_s_fit = exp_freq_thz_fit * 1e12 * 2 * np.pi

    # --- モデル比較の準備 ---
    model_types = ['H_form', 'B_form']
    traces = {}
    ppcs = {}
    n_transitions = 7

    # --- 各モデルでサンプリングを実行 ---
    for mt in model_types:
        print(f"\n--- [{mt}] 複素数透過率モデルのサンプリングを開始します ---")
        with pm.Model() as model:
            # より制約的で物理的に意味のある事前分布
            a = pm.TruncatedNormal(f'a_{mt}', mu=1.0, sigma=0.2, lower=0.1, upper=3.0)  # 振幅の制約
            log_gamma = pm.Normal(f'log_gamma_{mt}', mu=np.log(gamma_init), sigma=0.5)  # より制約的
            eps_bg = pm.TruncatedNormal(f'eps_bg_{mt}', mu=eps_bg_init, sigma=0.5, lower=10.0, upper=20.0)  # 物理的範囲
            d = pm.TruncatedNormal(f'd_{mt}', mu=d_init, sigma=5e-6, lower=-50e-6, upper=50e-6)  # より狭い範囲
            B4 = pm.Normal(f'B4_{mt}', mu=B4_init, sigma=abs(B4_init) * 0.5)  # より制約的
            B6 = pm.Normal(f'B6_{mt}', mu=B6_init, sigma=abs(B6_init) * 0.5)  # より制約的  
            g_factor = pm.TruncatedNormal(f'g_factor_{mt}', mu=g_factor_init, sigma=0.02, lower=1.9, upper=2.1)  # 物理的制約
            # より適切なノイズモデル
            nu = pm.Gamma(f'nu_{mt}', alpha=3, beta=0.2)  # より制約的な自由度
            sigma_obs = pm.HalfNormal(f'sigma_{mt}', sigma=0.05)  # より小さなノイズ期待値
            
            # パラメータを配列にまとめる
            params_vector = pt.stack([a, log_gamma, eps_bg, d, B4, B6, g_factor])
            
            # モデルタイプフラグをPyTensorテンソルとして作成
            model_type_flag = pt.as_tensor(0.0 if mt == 'H_form' else 1.0, dtype='float64')
            
            # 複素数透過率計算モデルを使用
            mu = calculate_transmission_complex(params_vector, model_type_flag)
            
            # そのまま使用（関数内で適切なサイズに調整済み）
            Y_obs = pm.StudentT('Y_obs', nu=nu, mu=mu, sigma=sigma_obs, observed=exp_transmittance_norm_fit)
            
            # より堅牢なサンプリング設定（収束性向上）
            traces[mt] = pm.sample(
                draws=2000,           # サンプル数を調整
                tune=2000,            # チューニング期間を調整  
                target_accept=0.9,    # 適度な受容率
                chains=4, 
                cores=4, 
                random_seed=42,
                init='adapt_diag',    # 対角適応初期化
                idata_kwargs={"log_likelihood": True}
            )
            ppcs[mt] = pm.sample_posterior_predictive(traces[mt], random_seed=42)
        print(f"--- [{mt}] モデルのサンプリング完了 ---")
        
        # 収束診断
        summary = az.summary(traces[mt], var_names=[f'a_{mt}', f'eps_bg_{mt}', f'd_{mt}', f'B4_{mt}', f'B6_{mt}', f'g_factor_{mt}', f'nu_{mt}', f'sigma_{mt}'])
        print(summary)
        
        # R-hat収束チェック
        rhat_issues = summary[summary['r_hat'] > 1.01]
        if len(rhat_issues) > 0:
            print(f"⚠️  [{mt}] R-hat > 1.01 のパラメータがあります:")
            print(rhat_issues[['r_hat']])
            print("   → より長いサンプリングが必要です")
        else:
            print(f"✅ [{mt}] 全てのパラメータが収束しました (R-hat ≤ 1.01)")
            
        # ESS (有効サンプル数) チェック
        ess_issues = summary[summary['ess_bulk'] < 400]
        if len(ess_issues) > 0:
            print(f"⚠️  [{mt}] ESS < 400 のパラメータがあります:")
            print(ess_issues[['ess_bulk']])
            print("   → サンプル数を増やすことを推奨します")

    # --- 5. モデル比較とPareto k値の診断 ---
    print("\n--- ベイズ的モデル比較 (LOO-CV) ---")
    
    # 各モデルで対応する変数名を指定してLOO計算
    loo_results = {}
    for mt in model_types:
        var_name = f'Y_obs_{mt}'
        loo_results[mt] = az.loo(traces[mt], var_name=var_name)
    
    # 手動でモデル比較表を作成
    import pandas as pd
    comparison_data = []
    for mt, loo_result in loo_results.items():
        comparison_data.append({
            'model': mt,
            'elpd_loo': loo_result.elpd_loo,
            'p_loo': loo_result.p_loo,
            'elpd_loo_se': loo_result.se,
            'warning': loo_result.warning
        })
    
    compare_df = pd.DataFrame(comparison_data).set_index('model')
    compare_df = compare_df.sort_values('elpd_loo', ascending=False)
    print(compare_df)
    
    # LOO差分を計算
    if len(compare_df) > 1:
        best_elpd = compare_df.iloc[0]['elpd_loo']
        compare_df['delpd_loo'] = compare_df['elpd_loo'] - best_elpd
        print(f"\nベストモデル: {compare_df.index[0]}")
        print(f"ELPD差分:")
        for model in compare_df.index[1:]:
            diff = compare_df.loc[model, 'delpd_loo']
            print(f"  {model}: {diff:.2f}")
    
    # 簡易プロット
    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos = range(len(compare_df))
    ax.bar(x_pos, compare_df['elpd_loo'], yerr=compare_df['elpd_loo_se'], capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(compare_df.index)
    ax.set_ylabel('ELPD LOO')
    ax.set_title('モデル比較 (LOO-CV)')
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

    print("\n--- Pareto k 診断 ---")
    for mt in model_types:
        var_name = f'Y_obs_{mt}'
        loo_result = az.loo(traces[mt], var_name=var_name, pointwise=True)
        k_values = loo_result.pareto_k.values
        k_problem_count = np.sum(k_values > 0.7)
        print(f"[{mt}モデル] Pareto k > 0.7 のデータ点数: {k_problem_count} / {len(k_values)}")
        if k_problem_count > 0:
            print("  -> LOO-CVの信頼性に懸念があります。モデルや尤度の見直しを検討してください。")

    # --- 6. プロット1: フィッティング領域の詳細 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'H_form': 'blue', 'B_form': 'red'}
    ax.plot(exp_freq_thz_fit, exp_transmittance_norm_fit, 'o', color='black', markersize=4, label='実験データ (フィッティング領域)')
    for mt in model_types:
        trace = traces[mt]
        az.plot_hdi(exp_freq_thz_fit, ppcs[mt].posterior_predictive[f'Y_obs_{mt}'], ax=ax, color=colors[mt], hdi_prob=0.94, fill_kwargs={'alpha': 0.2})
        # ベストフィット曲線の計算
        params_mean = [
            trace.posterior[f'a_{mt}'].mean().item(),
            trace.posterior[f'log_gamma_{mt}'].mean().item(),  # log_gammaに修正
            trace.posterior[f'eps_bg_{mt}'].mean().item(),
            trace.posterior[f'd_{mt}'].mean().item(),
            trace.posterior[f'B4_{mt}'].mean().item(),
            trace.posterior[f'B6_{mt}'].mean().item(),
            trace.posterior[f'g_factor_{mt}'].mean().item()
        ]
        freq_plot = np.linspace(exp_freq_thz_fit.min(), exp_freq_thz_fit.max(), 500)
        omega_plot = freq_plot * 1e12 * 2 * np.pi
        prediction = calculate_transmission_np(params_mean, omega_plot, 35.0, 7.8, mt)
        ax.plot(freq_plot, prediction, color=colors[mt], lw=2, label=f'ベストフィット ({mt})')
    ax.set_xlabel('周波数 (THz)')
    ax.set_ylabel('正規化透過率')
    ax.set_title('ベイズ最適化結果: フィッティング領域')
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig('fitting_plot_fit_region.png')
    plt.show()

    # --- 7. プロット2: 全領域への予測（外挿）性能 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(exp_freq_thz_full, exp_transmittance_norm_full, 'o', color='lightgray', markersize=3, label='実験データ (全領域)')
    ax.plot(exp_freq_thz_fit, exp_transmittance_norm_fit, 'o', color='black', markersize=4, label='実験データ (フィッティング領域)')
    for mt in model_types:
        trace = traces[mt]
        params_mean = [
            trace.posterior[f'a_{mt}'].mean().item(),
            trace.posterior[f'log_gamma_{mt}'].mean().item(),  # log_gammaに修正
            trace.posterior[f'eps_bg_{mt}'].mean().item(),
            trace.posterior[f'd_{mt}'].mean().item(),
            trace.posterior[f'B4_{mt}'].mean().item(),
            trace.posterior[f'B6_{mt}'].mean().item(),
            trace.posterior[f'g_factor_{mt}'].mean().item()
        ]
        freq_plot = np.linspace(exp_freq_thz_full.min(), exp_freq_thz_full.max(), 1000)
        omega_plot = freq_plot * 1e12 * 2 * np.pi
        prediction = calculate_transmission_np(params_mean, omega_plot, 35.0, 7.8, mt)
        ax.plot(freq_plot, prediction, color=colors[mt], lw=2, linestyle='--', label=f'全領域予測 ({mt})')
    ax.axvline(x=exp_freq_thz_fit.max(), color='gray', linestyle=':', label='フィッティング領域上限')
    ax.set_xlabel('周波数 (THz)')
    ax.set_ylabel('正規化透過率')
    ax.set_title('ベイズ最適化結果: 全領域への予測性能')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig('fitting_plot_full_range.png')
    plt.show()

    print("\n全ての処理が完了しました。結果を確認してください。")
