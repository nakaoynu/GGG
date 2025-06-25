import numpy as np
import matplotlib.pyplot as plt
#from scipy.signal import find_peaks

# 日本語フォントの設定 (ご自身の環境に合わせてフォント名を変更してください)
# Windows: "Meiryo", "Yu Gothic", "MS Gothic"
# macOS: "Hiragino Sans"
plt.rcParams['font.family'] = "Meiryo"

# --- 1. 物理定数とシミュレーションパラメータ ---
# これらの値は論文や既存コードを参考にしていますが、後で調整可能です。

# 物理定数
kB = 1.380649e-23  # ボルツマン定数 [J/K]
muB = 9.274010e-24 # ボーア磁子 [J/T]
hbar = 1.054571e-34 # ディラック定数 [J*s]
c = 299792458      # 光速 [m/s]
mu0 = 4.0 * np.pi * 1e-7 # 真空の透磁率 [H/m]

# GGGの物質パラメータ
g_factor = 1.95      # g因子
eps_bg = 11.5       # 背景比誘電率 (論文より)
s = 3.5              # Gd3+のスピン量子数 (S=7/2)
N_spin = 24/1.238 * 1e27     # スピン密度 [m^-3] (論文の情報を基に計算・仮定)
d = 0.1578e-3        # サンプルの厚み [m]
g0 = 6.0e11 * np.sqrt(7.8) / np.sqrt(21.5)

# 結晶場パラメータ (B_k^q = B_k / f_k)
# thesis_sakata_latest.pdf の値を参考に調整
B4_param = 0.8 / 240
B6_param = 0.04 / 5040
factor_b4 = 0.606
factor_b6 = -1.513
B4 = B4_param * factor_b4 # [K]
B6 = B6_param * factor_b6 # [K]

# シミュレーション条件
T = 35.0             # 温度 [K]←条件すり合わせ必要
B_ext = 7.8          # 外部静磁場 [T]←条件すり合わせ必要
gamma = 2.5e11       # 緩和周波数 [Hz] (Elijahの論文よりスペクトルの線幅を決定)

# --- 2. 演算子の定義 ---
def get_spin_operators(spin):
    """指定されたスピン量子数に対するスピン演算子行列を返す"""
    # np.arangeを使用してスピンのm値を生成
    m = np.arange(spin, -spin - 1, -1)

    # Sz: 対角行列
    Sz = np.diag(m)
    
    return Sz

def get_stevens_operators():
    """スティーブンス演算子 O_k^q を返す (S=7/2 固定)"""
    # 既存コード (f1_H.py) より引用
    O04 = 60 * np.diag([7, -13, -3, 9, 9, -3, -13, 7])
    
    X = np.zeros((8, 8))
    X[3, 7] = np.sqrt(35)
    X[4, 0] = np.sqrt(35)
    X[2, 6] = 5 * np.sqrt(3)
    X[5, 1] = 5 * np.sqrt(3)
    O44 = 12 * (X + X.T)
    
    O06 = 1260 * np.diag([1, -5, 9, -5, -5, 9, -5, 1])

    X = np.zeros((8, 8))
    X[3, 7] = 3 * np.sqrt(35)
    X[4, 0] = 3 * np.sqrt(35)
    X[2, 6] = -7 * np.sqrt(3)
    X[5, 1] = -7 * np.sqrt(3)
    O46 = 60 * (X + X.T)
    
    return O04, O44, O06, O46

# --- 3. ハミルトニアンの計算 ---

def get_hamiltonian(B_ext_z):
    """与えられた外部磁場に対する無摂動ハミルトニアンを計算する"""
    Sz= get_spin_operators(s)
    O04, O44, O06, O46 = get_stevens_operators()
    
    # 結晶場ハミルトニアン
    # 山田の卒論より、B_ext || [001] (c-axis) を想定
    H_cf = (B4 * kB) * (O04 + 5 * O44) + (B6 * kB) * (O06 - 21 * O46)
    
    # ゼーマンハミルトニアン (磁場はz方向と仮定)
    H_zee = g_factor * muB * B_ext_z * Sz
    
    return H_cf + H_zee #[J]

# --- 4. 磁気感受率と透過率の計算 ---

def calculate_susceptibility(omega, H, T):
    """
    円偏光に対する磁気感受率 chi_R, chi_L を計算する
    式は thesis_sakata_lateset.pdf (Eq. 2-25, 2-26) を参照
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H) # ハミルトニアンの固有値問題を解く
    eigenvalues -= np.min(eigenvalues) # 基底状態をエネルギーのゼロ点に設定
    #print("Eigenvalues (in J):", eigenvalues)
    
    # 占有確率 P(E_n) = exp(-E_n / kBT) / Z
    Z = np.sum(np.exp(-eigenvalues / (kB * T)))
    populations = np.exp(-eigenvalues / (kB * T)) / Z #P(E_n)
    #print("Populations:", populations)
    
    # 磁気感受率の計算
    chi = 0.0
    m_vals = np.arange(-s, s, 1)
    for i in range(len(eigenvalues)-1):
        m = m_vals[i]
        delta_E = eigenvalues[i + 1] - eigenvalues[i]
        delta_pop = populations[i + 1] - populations[i]
        omega_0 = delta_E / hbar
        chi += (4 * g0**2 /(2 * np.pi * 2.5e11)) * delta_pop * (s + m) * (s - m + 1) / ((omega_0 - omega) - (1j * gamma / 2))

    return chi

def calculate_transmission(omega, chi):
    """感受率から透過率を計算する"""
    # 比透磁率 mu_r = 1 + chi
    # thesis_sakata_lateset.pdf (Eq. 2-1) などを参照
    # B形式ではなくH形式 (mu = 1 + chi) を採用
    mu_r = 1 + chi
    
    # 複素屈折率 n_complex = sqrt(eps_bg * mu_r),インピーダンスの定義を考慮
    n_complex = np.sqrt(eps_bg * mu_r + 0j)
    impe = np.sqrt(mu_r / eps_bg + 0j)  # インピーダンスの定義
    lamda = c / (omega / (2 * np.pi))  # 波長 [m]
    
    # 透過係数 (Fabry-Perot効果を考慮)
    # thesis_sakata_lateset.pdf (Eq. 2-46)，多重反射を考慮する必要あり
    delta = 2 * np.pi * n_complex * d / lamda  # 光路差
    t = 4 * impe * np.exp(1j * delta) / ((1 + impe)**2)
    
    # 透過率 T = |t|^2
    return np.abs(t)**2

# --- 5. メイン実行ブロック ---
if __name__ == '__main__':
    # シミュレーションの周波数範囲 (THz -> Hz)
    freq_thz = np.arange(0.1e11, 9.0e11, 0.1e11)  # 0.1THzから9THzまで
    # 周波数をラジアン毎秒(ω)に変換
    omega_rad_s = freq_thz * 2 * np.pi

    # 結果を格納する配列
    trans_R = []  # 右円偏光 (RCP)
    chi_R = []  # 右円偏光の感受率

    # ハミルトニアンを計算 (周波数ループの外で一度だけ計算)
    Sz = get_spin_operators(s)
    H = get_hamiltonian(B_ext)
    #print(H)
    
    # 周波数ごとに計算を実行
    print("シミュレーションを開始します...")
    for i, omega in enumerate(omega_rad_s):
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(omega_rad_s)} 完了")
        # 1. 現在のomegaに対する感受率を計算し、一時変数に格納
        current_chi_R = calculate_susceptibility(omega, H, T)
        # 2. 感受率をリストに追加
        chi_R.append(current_chi_R)
        # 3. 現在の感受率を用いて透過率を計算
        current_T_R = calculate_transmission(omega, current_chi_R)
        # 4. 透過率をリストに追加
        trans_R.append(current_T_R)
    print("シミュレーション完了。")
    
    trans_R = (
           (np.array(trans_R) - np.min(trans_R)) / (np.max(trans_R) - np.min(trans_R))
        )

    # --- 6. グラフ描画 ---
    # 横軸ω，縦軸透過率Tのグラフを描画
    plt.figure(figsize=(10, 6))

    plt.plot(freq_thz*1e-12, trans_R, label='右円偏光 (RCP)', color='red')

    plt.xlabel('周波数 (THz)')
    plt.ylabel('透過率 $T$')
    plt.title(f'GGG 透過スペクトルシミュレーション (T={T} K, B={B_ext} T)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(min(freq_thz*1e-12), max(freq_thz*1e-12))
    plt.savefig('test_g0.png', dpi=300)
    plt.tight_layout()
