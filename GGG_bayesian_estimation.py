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
    factor_b4 = 0.606; factor_b6 = -1.513
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
    itypes = [pt.dscalar]  # 入力: a (スカラー)
    otypes = [pt.dvector] # 出力: 透過率スペクトル (ベクトル)

    def __init__(self, omega_array, H_matrix, T_val): # 物理モデルの初期化（メソッド, PhysicsModelOpのインスタンスが作成されるときに1度だけ呼ばれる）
        self.omega_array = omega_array
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
        N_spin = N_spin_exp * 10
        self.G0 = mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)


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

    def _calculate_susceptibility(self, a):
        # ベクトル化された感受率計算
        delta_E = self.eigenvalues[1:] - self.eigenvalues[:-1]
        delta_pop = self.populations[1:] - self.populations[:-1]
        omega_0 = delta_E / hbar
        m_vals = np.arange(s, -s - 1, -1)
        transition_strength = (s + m_vals[:-1]) * (s - m_vals[:-1] + 1)
        
        numerator = self.G0 * delta_pop * transition_strength
        denominator = (omega_0[:, np.newaxis] - self.omega_array) - (1j * gamma) 
        """
                #omega_0 は遷移周波数（配列）で、[:, np.newaxis] により縦ベクトル（列ベクトル）に変換されます。
                #omega_array は入力された周波数（縦ベクトル）で、csvファイルから読み込まれたものです。
                二つの配列をブロードキャストして、各遷移周波数と入力された周波数の差を計算します。
        """        
        chi_array = np.sum(numerator[:, np.newaxis] / denominator, axis=0)
        return chi_array

    def _calculate_transmission(self, chi_array):
        """
        感受率から透過率を計算する
        ★★★ 比透磁率を mu_r = 1 + a * chi に変更 ★★★
        """
        mu_r = 1 + a * chi_array
        
        n_complex = np.sqrt(eps_bg * mu_r + 0j)
        impe = np.sqrt(mu_r / eps_bg + 0j)
        lambda_0 = np.divide(2 * np.pi * c, self.omega_array, where=self.omega_array!=0, out=np.inf)
        delta = 2 * np.pi * n_complex * d / lambda_0
        numerator = 4 * impe * np.exp(1j * delta) / (1 + impe)**2  
        denominator = 1 + ((impe - 1) / (impe + 1))**2 * np.exp(2j * delta)
        t = np.divide(numerator, denominator, where=denominator!=0, out=0j)
        return np.abs(t)**2

# --- 2. データの読み込みと準備 ---
print("実験データを読み込みます...")
# ファイル名を適宜確認・変更してください
try:
    df = pd.read_csv("Circular_Polarization_B_Field.xlsx - Sheet2.csv")
    # A列とB列をそれぞれx, yとして読み込む
    exp_freq_thz = df.iloc[1:, 0].values
    exp_transmittance = df.iloc[1:, 1].values
except FileNotFoundError:
    print("エラー: 'Circular_Polarization_B_Field.xlsx - Sheet2.csv' が見つかりません。")
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
    # a: 0から1の範囲の正規分布。平均0.5,標準偏差0.25で緩やかに設定
    a = pm.TruncatedNormal('a', mu=0.5, sigma=0.25, lower=0.0, upper=1.0)
    
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
    trace = pm.sample(2000, tune=1000, chains=2, cores=1)
    # 事後予測サンプリング
    ppc = pm.sample_posterior_predictive(trace)
print("サンプリング完了。")

# --- 5. 結果の可視化と評価 ---
print("結果をプロットします...")
# 1. パラメータ`a`の事後分布
az.plot_posterior(trace, var_names=['a', 'sigma'])
plt.savefig('posterior_dist.png')

# 2. 実験データとモデルの予測の比較
fig, ax = plt.subplots(figsize=(10, 6))
# 事後予測プロット (PPC)
az.plot_hdi(exp_freq_thz, ppc.posterior_predictive['Y_obs'], ax=ax, color='lightgray', hdi_prob=0.95, label='95% HDI')
az.plot_hdi(exp_freq_thz, ppc.posterior_predictive['Y_obs'], ax=ax, color='darkgray', hdi_prob=0.50, label='50% HDI')

# 実験データ
ax.plot(exp_freq_thz, exp_transmittance, 'o', color='red', markersize=4, label='実験データ')

# 事後分布の平均値を使ったベストフィット曲線
a_mean = trace.posterior['a'].mean().item()
best_fit_transmittance = physics_model.perform(None, [a_mean], [np.empty_like(exp_transmittance)])[0]
ax.plot(exp_freq_thz, best_fit_transmittance, color='blue', lw=2, label=f'ベストフィット (a={a_mean:.3f})')

ax.set_xlabel('周波数 (THz)')
ax.set_ylabel('透過率 (任意単位)')
ax.set_title('ベイズ推定によるモデルフィッティング結果')
ax.legend()
ax.grid(True)
plt.savefig('fitting_result.png')
plt.show()
plt.close()
