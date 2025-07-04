import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.graph import Op, Apply

# --- 0. プロット設定 ---
plt.rcParams['font.family'] = "Meiryo"
plt.rcParams['figure.dpi'] = 100

# --- 1. 物理モデルの定義 (これまでのスクリプトから引用・整理) ---

# 物理定数とパラメータ
kB = 1.380649e-23
muB = 9.274010e-24
hbar = 1.054571e-34
c = 299792458
g_factor = 1.95 
eps_bg = 11.5 
s = 3.5
d = 0.1578e-3
T = 35.0
B_ext = 7.8
gamma = 110e9

# ハミルトニアンの計算 (一度だけ実行すれば良い)
def get_hamiltonian():
    Sz = np.diag(np.arange(s, -s - 1, -1))
    B4_param = 0.8 / 240; B6_param = 0.04 / 5040
    factor_b4 = 0.606; factor_b6 = -1.513 #ベイズ推定候補, なぜこのファクターになるのか要確認
    B4 = B4_param * factor_b4; B6 = B6_param * factor_b6
    O04 = 60 * np.diag([7, -13, -3, 9, 9, -3, -13, 7])
    X_O44 = np.zeros((8, 8)); X_O44[3, 7], X_O44[4, 0] = np.sqrt(35), np.sqrt(35); X_O44[2, 6], X_O44[5, 1] = 5*np.sqrt(3), 5*np.sqrt(3); O44 = 12 * (X_O44 + X_O44.T)
    O06 = 1260 * np.diag([1, -5, 9, -5, -5, 9, -5, 1])
    X_O46 = np.zeros((8, 8)); X_O46[3, 7], X_O46[4, 0] = 3*np.sqrt(35), 3*np.sqrt(35); X_O46[2, 6], X_O46[5, 1] = -7*np.sqrt(3), -7*np.sqrt(3); O46 = 60 * (X_O46 + X_O46.T)
    H_cf = (B4 * kB) * (O04 + 5 * O44) + (B6 * kB) * (O06 - 21 * O46)
    H_zee = g_factor * muB * B_ext * Sz
    return H_cf + H_zee

# 物理モデルをカプセル化するクラス (PyMC <-> NumPyの橋渡し)
class PhysicsModelOp(Op):
    # PyMCへの入力と出力の型を定義
    itypes = [pt.dscalar]  # Input Types（入力）: a (スカラー)
    otypes = [pt.dvector] # Output Types（出力）: 透過率スペクトル (ベクトル)

    def __init__(self, omega_array, H_matrix, T_val): # 物理モデルの初期化（メソッド, PhysicsModelOpのインスタンスが作成されるときに1度だけ呼ばれる）
        self.omega_array = omega_array # selfの属性として, omega_arrayを記憶させる
        self.H_matrix = H_matrix
        self.T = T_val

        # 物理モデルで共通に使う値を事前計算
        eigenvalues, _ = np.linalg.eigh(self.H_matrix)
        eigenvalues -= np.min(eigenvalues)
        self.eigenvalues = eigenvalues
        Z = np.sum(np.exp(-self.eigenvalues / (kB * self.T)))
        self.populations = np.exp(-self.eigenvalues / (kB * self.T)) / Z
        
        # 磁気感受率の定数部分
        mu0 = 4.0 * np.pi * 1e-7
        N_spin_exp = 24/1.238 * 1e27
        N_spin = N_spin_exp * 10 #ベイズ推定候補
        self.G0 = mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)

    def _calculate_susceptibility(self, a): #引数 a は PyMC から渡されるパラメータ
        # ベクトル化された感受率計算
        delta_E = self.eigenvalues[1:] - self.eigenvalues[:-1]
        delta_pop = self.populations[1:] - self.populations[:-1]
        omega_0 = delta_E / hbar
        m_vals = np.arange(s, -s, -1) # m = [s, s-1, ..., -s+1] 
        transition_strength = (s + m_vals) * (s - m_vals + 1)
        
        # ここで'a'を使って感受率を計算
        numerator = a * self.G0 * delta_pop * transition_strength
        denominator = (omega_0[:, np.newaxis] - self.omega_array) - (1j * gamma) 
        """
                #omega_0 は遷移周波数（配列）で、[:, np.newaxis] により縦ベクトル（列ベクトル）に変換されます。
                #omega_array は入力された周波数（縦ベクトル）で、csvファイルから読み込まれたものです。
                二つの配列をブロードキャストして、各遷移周波数と入力された周波数の差を計算します。
        """        
        chi_array = np.sum(numerator[:, np.newaxis] / denominator, axis=0)
        return chi_array

    def _calculate_transmission(self, chi_array):
        #感受率から透過率を計算する
        mu_r = 1/ (1 - chi_array)
        
        n_complex = np.sqrt(eps_bg * mu_r + 0j)
        n_complex_bg = np.sqrt(eps_bg * 1 + 0j)
        impe = np.sqrt(mu_r / eps_bg + 0j)
        impe_bg = np.sqrt(1 / eps_bg + 0j) 

        # ゼロ割を安全に回避
        omega_array = self.omega_array
        lambda_0 = np.full_like(omega_array, np.inf, dtype=float) # omega_arrayと同じ形状の配列を作成し、初期値は無限大
        nonzero_mask = omega_array != 0
        lambda_0[nonzero_mask] = (2 * np.pi * c) / omega_array[nonzero_mask]        
        delta = 2 * np.pi * n_complex * d / lambda_0
        delta_bg = 2 * np.pi * n_complex_bg * d / lambda_0
        numerator = 4 * impe * np.exp(1j * delta) / (1 + impe)**2
        numerator_bg =  4 * impe_bg * np.exp(1j * delta_bg) / (1 + impe_bg)**2
        denominator = 1 + ((impe - 1) / (impe + 1))**2 * np.exp(2j * delta)
        denominator_bg = 1 + ((impe_bg - 1) / (impe_bg + 1))**2 * np.exp(2j * delta_bg)
        t = np.divide(numerator, denominator, where=denominator!=0, out=np.zeros_like(denominator, dtype=complex)) 
        t_bg = np.divide(numerator_bg, denominator_bg, where=denominator!=0, out=np.zeros_like(denominator, dtype=complex)) 
        return  np.abs( t - t_bg )**2
    
    def perform(self, node, inputs, output_storage):
        # PyMCから渡されたパラメータ`a`
        a, = inputs # inputsリストから最初の値を取得
        
        # 物理計算の実行
        chi_array = self._calculate_susceptibility(a)
        transmittance_array = self._calculate_transmission(chi_array)
        
        # 計算結果をPyMCに返す
        output_storage[0][0] = transmittance_array
        """
        output_storage: 計算結果をPyMCの世界に返すための場所。
        output_storage[0][0] = transmittance_array は、「0番目の出力（otypesで定義）に、計算したtransmittance_arrayを格納してください」という意味。この値が、PyMCモデルの次の部分（今回は尤度関数 pm.Normal）へと渡されます。
        """
if __name__ == '__main__':
    # --- 2. データの読み込みと準備 ---
    print("実験データを読み込みます...")
    file_path = "Circular_Polarization_B_Field.xlsx"
    sheet_name = "Sheet2"
    # pandasを使ってExcelファイルからデータを読み込む
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0, names=['Frequency (THz)', 'Transmittance'])
        # A列とB列をそれぞれx, yとして読み込む
    #exp_freq_thz = df.iloc[1:, 0].values
    #exp_transmittance = df.iloc[1:, 1].values
        exp_freq_thz = df['Frequency (THz)'].to_numpy(dtype=float)
        exp_transmittance = df['Transmittance'].to_numpy(dtype=float)
        print(f"データの読み込みに成功しました。読み込み件数: {len(df)}件")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません。パスを確認してください。\nパス: {file_path}")
        exit()
    except Exception as e:
        # その他のエラー（例：シート名が違うなど）もキャッチ
        print(f"データの読み込み中にエラーが発生しました: {e}")
        exit()

    # データ単位をシミュレーションの単位に合わせる (THz -> rad/s)
    exp_omega = exp_freq_thz * 1e12 * 2 * np.pi

    # --- 3. PyMCモデルの構築 ---
    print("PyMCモデルを構築します...")
    # 物理モデルのインスタンスを作成
    H = get_hamiltonian()
    physics_model = PhysicsModelOp(exp_omega, H, T)

    with pm.Model() as model:
        # --- 事前分布 (Priors) ---
        # a: 無情報な一様分布
        a = pm.Beta('a', alpha=1.0, beta=1.0)
        
        # sigma: 測定誤差の標準偏差。幅広い値を取りうるHalfCauchy分布を仮定
        sigma = pm.HalfCauchy('sigma', beta=1.0)

        # --- 物理モデルとの連携 ---
        # `a`を物理モデルに入力し、理論的な透過率`mu`を計算
        mu = physics_model(a)

        # --- 尤度 (Likelihood) ---
        # モデルの予測値(mu)と実験データ(exp_transmittance)を正規分布で結びつける
        # 測定誤差はsigmaに従うと仮定
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=exp_transmittance)

    # --- 4. MCMCサンプリングの実行 ---
    print("MCMCサンプリングを開始します... (時間がかかる場合があります)")
    with model:
        # NUTSサンプラーを用いて事後分布からサンプリング
        trace = pm.sample(2000, tune=1000, chains=4, cores=4, random_seed=42)
        # 事後予測サンプリング
        ppc = pm.sample_posterior_predictive(trace, random_seed=42)
    print("サンプリング完了。")
    print(az.summary(trace, var_names=['a', 'sigma']))

    # --- 6. 結果の可視化と評価 ---
    print("結果をプロットします...")

    # --------------------------------------------------------------------
    # プロット1: パラメータの事後分布とトレースプロット
    # --------------------------------------------------------------------
    try:
        print("トレースプロットを生成します...")
        
        # az.plot_traceはAxesの配列を返す
        axes = az.plot_trace(trace, var_names=['a', 'sigma'], figsize=(10, 7))
        
        # Axes配列からFigureオブジェクトを取得する
        fig = axes.ravel()[0].figure
        
        # Figureオブジェクトに対してタイトルを設定
        fig.suptitle('パラメータ事後分布とサンプリングのトレース', fontsize=16)
        
        # レイアウトを自動調整
        fig.tight_layout()
        # ファイルに保存
        plt.savefig('B_trace_plot.png', dpi=300)
        print("✅ 'B_trace_plot.png' を保存しました。")

    except Exception as e:
        print(f"❌ トレースプロットの生成中にエラーが発生しました: {e}")
    finally:
        # メモリ解放のため、現在の図を閉じる
        plt.close('all')


    # --------------------------------------------------------------------
    # プロット2: 実験データとモデルの予測の比較
    # --------------------------------------------------------------------
    try:
        print("フィッティング結果のグラフを生成します...")
        fig, ax = plt.subplots(figsize=(10, 6))

        # 事後予測プロット (モデルの不確実性を示す灰色の帯)
        az.plot_hdi(exp_freq_thz, ppc.posterior_predictive['Y_obs'], ax=ax, color='lightgray', hdi_prob=0.95)
        az.plot_hdi(exp_freq_thz, ppc.posterior_predictive['Y_obs'], ax=ax, color='darkgray', hdi_prob=0.50)

        # 実験データ
        ax.plot(exp_freq_thz, exp_transmittance, 'o', color='red', markersize=4, label='実験データ')

        # 事後分布の平均値を使ったベストフィット曲線
        a_mean = trace.posterior['a'].mean().item()
        best_fit_chi = physics_model._calculate_susceptibility(a_mean)
        best_fit_transmittance = physics_model._calculate_transmission(best_fit_chi)
        ax.plot(exp_freq_thz, best_fit_transmittance, color='blue', lw=2, label=f'ベストフィット (a={a_mean:.3f})')

        # ★★★ ここを追加: HDIの凡例をダミーの線で作成 ★★★
        ax.plot([], [], lw=8, color='lightgray', label='95% HDI')
        ax.plot([], [], lw=8, color='darkgray', label='50% HDI')
        
        # グラフの装飾
        ax.set_xlabel('周波数 (THz)', fontsize=12)
        ax.set_ylabel('透過率', fontsize=12)
        ax.set_title('ベイズ推定によるモデルフィッティング結果', fontsize=16)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(bottom=0)

        # ファイルに保存
        plt.savefig('B_fitting_result.png', dpi=300)
        print("✅ 'B_fitting_result.png' を保存しました。")

    except Exception as e:
        print(f"❌ フィッティング結果のプロット中にエラーが発生しました: {e}")
    finally:
        # メモリ解放のため、現在の図を閉じる
        plt.close('all')
