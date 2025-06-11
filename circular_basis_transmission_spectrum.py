import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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
mu0 = 4 * np.pi * 1e-7 # 真空の透磁率 [H/m]

# GGGの物質パラメータ
g_factor = 1.95      # g因子
eps_bg = 11.5        # 背景比誘電率 (論文より)
s = 3.5              # Gd3+のスピン量子数 (S=7/2)
N_spin = 4.22e27     # スピン密度 [m^-3] (論文の情報を基に計算・仮定)←確認する必要あり
d = 0.1578e-3        # サンプルの厚み [m]

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
gamma = 2.5e11       # 緩和周波数 [Hz] (スペクトルの線幅を決定)←条件すり合わせ必要

# --- 2. 演算子の定義 ---

def get_spin_operators(spin):
    """指定されたスピン量子数に対するスピン演算子行列を返す"""
    dim = int(2 * spin + 1)
    m = np.arange(spin, -spin - 1, -1)
    
    # Sz: 対角行列
    Sz = np.diag(m)
    
    # Sx, Sy (S+, S-から計算)
    sp_diag = np.sqrt(spin * (spin + 1) - m[:-1] * (m[:-1] - 1))
    sm_diag = np.sqrt(spin * (spin + 1) - m[1:] * (m[1:] + 1))
    
    Sp = np.diag(sp_diag, k=1)
    Sm = np.diag(sm_diag, k=-1)
    
    Sx = 0.5 * (Sp + Sm)
    Sy = -0.5j * (Sp - Sm)
    
    return Sx, Sy, Sz, Sp, Sm

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
    """与えられた外部磁場に対する全ハミルトニアンを計算する"""
    Sx, Sy, Sz, Sp, Sm = get_spin_operators(s)
    O04, O44, O06, O46 = get_stevens_operators()
    
    # 結晶場ハミルトニアン
    # 論文より、B_ext || [001] (c-axis) を想定
    H_cf = (B4 * kB) * (O04 + 5 * O44) + (B6 * kB) * (O06 - 21 * O46)
    
    # ゼーマンハミルトニアン (磁場はz方向と仮定)←修正必要
    H_zee = g_factor * muB * B_ext_z * Sz
    
    return H_cf + H_zee

# --- 4. 磁気感受率と透過率の計算 ---

def calculate_susceptibility(omega, H, T, Sp, Sm):
    """
    円偏光に対する磁気感受率 chi_plus, chi_minus を計算する
    式は thesis_sakata_lateset.pdf (Eq. 2-25, 2-26) を参照
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    eigenvalues -= np.min(eigenvalues) # 基底状態をエネルギーのゼロ点に設定
    
    # 占有確率 P(E_n) = exp(-E_n / kBT) / Z
    Z = np.sum(np.exp(-eigenvalues / (kB * T)))
    populations = np.exp(-eigenvalues / (kB * T)) / Z
    
    # 遷移要素 <n|S+|m>, <n|S-|m> を計算
    # U.H. @ Op @ U で、演算子を行列Hの固有基底に変換
    Sp_eig_basis = eigenvectors.conj().T @ Sp @ eigenvectors
    Sm_eig_basis = eigenvectors.conj().T @ Sm @ eigenvectors

    chi_plus = 0.0
    chi_minus = 0.0

    for n in range(len(eigenvalues)):
        for m in range(len(eigenvalues)):
            if n == m:
                continue
            
            # エネルギー差と遷移確率
            delta_E = eigenvalues[n] - eigenvalues[m]
            pop_diff = populations[m] - populations[n]
            
            # ローレンツ型の分母 (緩和項を含む)
            denominator = delta_E - hbar * (omega + 1j * gamma)

            # |<n|S+|m>|^2 と |<n|S-|m>|^2
            trans_prob_plus = np.abs(Sp_eig_basis[n, m])**2
            trans_prob_minus = np.abs(Sm_eig_basis[n, m])**2

            if np.abs(denominator) > 1e-30: # ゼロ割を避ける
                chi_plus += pop_diff * trans_prob_plus / denominator
                chi_minus += pop_diff * trans_prob_minus / denominator

    # 定数部分を乗算
    prefactor = N_spin * (g_factor * muB)**2 / hbar
    
    return prefactor * chi_plus, prefactor * chi_minus

def calculate_transmission(omega, chi):
    """感受率から透過率を計算する"""
    # 比透磁率 mu_r = 1 + chi
    # thesis_sakata_lateset.pdf (Eq. 2-1) などを参照
    # B形式ではなくH形式 (mu = 1 + chi) を採用
    mu_r = 1 + chi
    
    # 複素屈折率 n_complex = sqrt(eps_bg * mu_r)
    n_complex = np.sqrt(eps_bg * mu_r)
    
    # 透過係数 (Fabry-Perot効果を考慮)
    # thesis_sakata_lateset.pdf (Eq. 2-46)
    delta = n_complex * omega * d / c
    t = 4 * n_complex * np.exp(-1j * delta) / ((1 + n_complex)**2)
    
    # 透過率 T = |t|^2
    return np.abs(t)**2

# --- 5. メイン実行ブロック ---
if __name__ == '__main__':
    # シミュレーションの周波数範囲 (THz -> Hz)
    freq_thz = np.linspace(0.1, 2.0, 200)
    omega_rad_s = freq_thz * 1e12 * 2 * np.pi

    # 結果を格納する配列
    trans_plus = []  # 右円偏光 (RCP)
    trans_minus = [] # 左円偏光 (LCP)
    
    # ハミルトニアンを計算 (周波数ループの外で一度だけ計算)
    H = get_hamiltonian(B_ext)
    Sx, Sy, Sz, Sp, Sm = get_spin_operators(s)

    # 周波数ごとに計算を実行
    print("シミュレーションを開始します...")
    for i, omega in enumerate(omega_rad_s):
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(omega_rad_s)} 完了")
            
        chi_p, chi_m = calculate_susceptibility(omega, H, T, Sp, Sm)
        
        T_p = calculate_transmission(omega, chi_p)
        T_m = calculate_transmission(omega, chi_m)
        
        trans_plus.append(T_p)
        trans_minus.append(T_m)
    print("シミュレーション完了。")
    
    # --- 6. グラフ描画 ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(freq_thz, trans_plus, label='右円偏光 (RCP)', color='red')
    plt.plot(freq_thz, trans_minus, label='左円偏光 (LCP)', color='blue')
    
    plt.xlabel('周波数 (THz)')
    plt.ylabel('透過率 T')
    plt.title(f'GGG 透過スペクトルシミュレーション (T={T} K, B={B_ext} T)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(min(freq_thz), max(freq_thz))
    plt.savefig('ggg_transmission_spectrum.png', dpi=300)
    plt.tight_layout()
    
    #plt.show()

