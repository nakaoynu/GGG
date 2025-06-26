import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.graph import Op, Apply
import japanize_matplotlib # 日本語表示のため

# --- 0. プロット設定 ---
plt.rcParams['figure.dpi'] = 100

# --- 1. 物理定数とパラメータ定義 ---
kB = 1.380649e-23; muB = 9.274010e-24; hbar = 1.054571e-34; c = 299792458
g_factor = 1.95; eps_bg = 11.5; s = 3.5; d = 0.1578e-3; T = 35.0; B_ext = 7.8
B4_param = 0.8 / 240 * 0.606; B6_param = 0.04 / 5040 * -1.513; B4 = B4_param; B6 = B6_param
mu0 = 4.0 * np.pi * 1e-7; N_spin_exp = 24/1.238 * 1e27; N_spin = N_spin_exp * 10
G0 = mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)

# --- 2. 物理モデルの関数定義 ---
def get_hamiltonian(B_ext_z):
    m_values = np.arange(s, -s - 1, -1); Sz = np.diag(m_values)
    O04 = 60*np.diag([7,-13,-3,9,9,-3,-13,7]); X_O44 = np.zeros((8,8)); X_O44[3,7],X_O44[4,0]=np.sqrt(35),np.sqrt(35); X_O44[2,6],X_O44[5,1]=5*np.sqrt(3),5*np.sqrt(3); O44=12*(X_O44+X_O44.T)
    O06 = 1260*np.diag([1,-5,9,-5,-5,9,-5,1]); X_O46 = np.zeros((8,8)); X_O46[3,7],X_O46[4,0]=3*np.sqrt(35),3*np.sqrt(35); X_O46[2,6],X_O46[5,1]=-7*np.sqrt(3),-7*np.sqrt(3); O46=60*(X_O46+X_O46.T)
    H_cf = (B4 * kB) * (O04 + 5 * O44) + (B6 * kB) * (O06 - 21 * O46)
    H_zee = g_factor * muB * B_ext_z * Sz
    return H_cf + H_zee

def calculate_transmission(omega_array, mu_r_array):
    n_complex = np.sqrt(eps_bg * mu_r_array + 0j)
    impe = np.sqrt(mu_r_array / eps_bg + 0j)
    lambda_0 = np.full_like(omega_array, np.inf, dtype=float)
    nonzero_mask = omega_array != 0; lambda_0[nonzero_mask] = (2 * np.pi * c) / omega_array[nonzero_mask]
    delta = 2 * np.pi * n_complex * d / lambda_0
    numerator = 4 * impe * np.exp(1j * delta) / (1 + impe)**2
    denominator = 1 + ((impe - 1) / (impe + 1))**2 * np.exp(2j * delta)
    t = np.divide(numerator, denominator, where=denominator!=0, out=np.zeros_like(denominator, dtype=complex))
    return t

# --- 3. PyMCと連携するためのOpクラス (モデル選択機能を追加) ---
class PhysicsModelOp(Op):
    # ★★★ 入力にgammaを追加: [a(scalar), gamma(scalar)] ★★★
    itypes = [pt.dscalar, pt.dscalar]
    otypes = [pt.dvector]

    def __init__(self, omega_array, T_val, B_val, model_type):
        self.omega_array = omega_array; self.T = T_val; self.B = B_val
        self.model_type = model_type # 'H_form' or 'B_form'

        # --- 事前計算 ---
        H_B = get_hamiltonian(self.B)
        eigenvalues_B, _ = np.linalg.eigh(H_B)
        eigenvalues_B -= np.min(eigenvalues_B)
        Z_B = np.sum(np.exp(-eigenvalues_B / (kB * self.T)))
        self.delta_pop_B = (np.exp(-eigenvalues_B[1:]/(kB*self.T)) - np.exp(-eigenvalues_B[:-1]/(kB*self.T))) / Z_B
        self.omega_0_B = (eigenvalues_B[1:] - eigenvalues_B[:-1]) / hbar
        m_vals = np.arange(s, -s, -1)
        self.transition_strength = (s + m_vals) * (s - m_vals + 1)
        mu_r_array_0 = np.ones_like(self.omega_array); self.T0 = calculate_transmission(self.omega_array, mu_r_array_0)

    def perform(self, node, inputs, output_storage):
        # ★★★ aとgammaの両方を受け取る ★★★
        a, gamma = inputs
        chi_B = self._calculate_susceptibility(gamma) # gammaを使ってchiを計算
        
        # ★★★ モデルタイプに応じてmu_r_arrayを計算 ★★★
        if self.model_type == 'H_form':
            mu_r_array_cal = 1 + a * chi_B
        elif self.model_type == 'B_form':
            # ゼロ割を避ける
            mu_r_array_cal = np.divide(1, 1 - a * chi_B, where=(1 - a * chi_B)!=0, out=np.ones_like(chi_B, dtype=complex))
        else:
            raise ValueError("Unknown model_type")
            
        T_B = calculate_transmission(self.omega_array, mu_r_array_cal)
        delta_t = T_B - self.T0
        delta_T = np.abs(delta_t)**2
        #min_val, max_val = np.min(delta_T), np.max(delta_T)
        #normalized_delta_T = (delta_T - min_val) / (max_val - min_val) if (max_val - min_val) > 1e-9 else np.zeros_like(delta_T)
        output_storage[0][0] = delta_T
        
    def _calculate_susceptibility(self, gamma): # 引数gammaを受け取る
        numerator = G0 * self.delta_pop_B * self.transition_strength
        denominator = (self.omega_0_B[:, np.newaxis] - self.omega_array) - (1j * gamma)
        chi_array = np.sum(numerator[:, np.newaxis] / denominator, axis=0)
        return chi_array

# --- 4. メイン実行ブロック ---
if __name__ == '__main__':
    # --- データの読み込み ---
    print("実験データを読み込みます...")
    file_path = "Circular_Polarization_B_Field.xlsx"; sheet_name = 'Sheet2'
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
        exp_freq_thz = df['Frequency (THz)'].to_numpy(dtype=float)
        exp_transmittance = df['Transmittance'].to_numpy(dtype=float)
        print(f"データの読み込みに成功しました。読み込み件数: {len(df)}件")
    except Exception as e: print(f"データ読み込み中にエラー: {e}"); exit()
    exp_omega_rad_s = exp_freq_thz * 1e12 * 2 * np.pi

    # --- モデル比較の準備 ---
    model_types = ['H_form', 'B_form']
    traces = {} # サンプリング結果を格納する辞書
    ppcs = {}   # 事後予測サンプリング結果を格納

    # --- 各モデルでサンプリングを実行 ---
    for mt in model_types:
        print(f"\n--- [{mt}] モデルのサンプリングを開始します ---")
        physics_model = PhysicsModelOp(exp_omega_rad_s, T, B_ext, model_type=mt)
        
        with pm.Model() as model:
            # 事前分布
            a = pm.TruncatedNormal('a', mu=1.0, sigma=0.25, lower=0.0, upper=1.0)
            gamma_param = pm.HalfNormal('gamma', sigma=50e9)
            sigma_obs = pm.HalfCauchy('sigma', beta=1.0)
            
            # 物理モデルにaとgammaを渡す
            mu = physics_model(a, gamma_param)

            # 尤度
            Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma_obs, observed=exp_transmittance)
            
            # サンプリング実行
            traces[mt] = pm.sample(
                1000, 
                tune=1000, 
                chains=4, 
                cores=4,
                random_seed=42,
                idata_kwargs={"log_likelihood": True}
            )
            ppcs[mt] = pm.sample_posterior_predictive(traces[mt], random_seed=42)
        print(f"--- [{mt}] モデルのサンプリング完了 ---")
        print(az.summary(traces[mt], var_names=['a', 'gamma','sigma']))

    # --- 5. モデル比較の結果表示 ---
    print("\n--- ベイズ的モデル比較 (LOO-CV) ---")
    # ArviZのcompare関数でモデルの予測性能を比較
    # traces辞書をInferenceDataの辞書に変換
    idata_dict = {k: v if isinstance(v, az.InferenceData) else az.from_pymc3(trace=v) for k, v in traces.items()}
    compare_df = az.compare(idata_dict)
    print(compare_df)
    axes = az.plot_compare(compare_df, figsize=(8, 4))
    # Figureオブジェクトを取得
    fig = axes.ravel()[0].figure if hasattr(axes, "ravel") else axes.figure
    fig.suptitle('モデル比較', fontsize=16)
    fig.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.close()

    # ---【修正案】6. 比較グラフの描画 ---

    print("\nフィッティング結果をプロットします...")
    fig, ax = plt.subplots(figsize=(10, 6))

    # 実験データをプロット
    ax.plot(exp_freq_thz, exp_transmittance, 'o', color='black', markersize=4, label='実験データ')

    # 各モデルのベストフィット曲線とHDIをプロット
    colors = {'H_form': 'blue', 'B_form': 'red'}
    for mt in model_types:
        print(f"[{mt}] モデルのプロットを生成中...")
        trace = traces[mt]
        ppc = ppcs[mt]
        
        # HDI (不確かさの帯) をプロット
        az.plot_hdi(exp_freq_thz, ppc.posterior_predictive['Y_obs'], ax=ax, color=colors[mt], hdi_prob=0.94, fill_kwargs={'alpha': 0.2})
        
        # 1. そのモデルタイプのOpを再度インスタンス化
        physics_model_for_plot = PhysicsModelOp(exp_omega_rad_s, T, B_ext, model_type=mt)
        
        # 2. 事後分布の平均値を取得
        a_mean = trace.posterior['a'].mean().item()
        gamma_mean = trace.posterior['gamma'].mean().item()
        
        # 3. ヘルパー関数を使って、安全に再計算
        best_fit_chi = physics_model_for_plot._calculate_susceptibility(gamma_mean)
        
        # モデルタイプに応じてmu_rを計算
        if mt == 'H_form':
            best_fit_mu_r = 1 + a_mean * best_fit_chi
        else: # B_form
            best_fit_mu_r = np.divide(1, 1 - a_mean * best_fit_chi, where=(1 - a_mean * best_fit_chi)!=0, out=np.ones_like(best_fit_chi, dtype=complex))

        best_fit_t_B = calculate_transmission(exp_omega_rad_s, best_fit_mu_r)
        best_fit_delta_t = best_fit_t_B - physics_model_for_plot.T0
        best_fit_delta_T = np.abs(best_fit_delta_t)**2
        
        
        # 差分スペクトルを正規化
        min_val, max_val = np.min(best_fit_delta_T), np.max(best_fit_delta_T)
        if (max_val - min_val) > 1e-9:
            best_fit_prediction = (best_fit_delta_T - min_val) / (max_val - min_val)
        else:
            best_fit_prediction = np.zeros_like(best_fit_delta_T)
        
        ax.plot(exp_freq_thz, best_fit_delta_T, color=colors[mt], lw=2, label=f'ベストフィット ({mt})')

    # HDIの凡例をダミーの線で作成
    ax.plot([], [], lw=8, color='lightgray', label='95% HDI (両モデル)')

    # グラフ装飾
    ax.set_xlabel('周波数 (THz)', fontsize=12)
    ax.set_ylabel('正規化透過率変化 $|t - t_{BG}|^{2}$', fontsize=12)
    ax.set_title('H形式 vs B形式 ベイズモデルフィッティング結果', fontsize=16)
    ax.legend()
    ax.grid(True)
    ax.set_ylim(bottom=0)
    plt.savefig('fitting_result_comparison.png', dpi=300)
    plt.close()

    print("\n解析が完了しました。")
