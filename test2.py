import numpy as np

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

def calculate_susceptibility(omega, H, T):
    """
    円偏光に対する磁気感受率 chi_R を計算する
    式は thesis_sakata_lateset.pdf (Eq. 2-25, 2-26) を参照
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H) # ハミルトニアンの固有値問題を解く
    #eigenvalues -= np.min(eigenvalues) # 基底状態をエネルギーのゼロ点に設定

    return eigenvalues 

# --- 5. メイン実行ブロック ---
if __name__ == '__main__':
    # シミュレーションの周波数範囲 (THz -> Hz)
    freq_thz = np.arange(0.1e11, 9.0e11, 0.1e11)  # 0.1THzから9THzまで
    # 周波数をラジアン毎秒(ω)に変換
    omega_rad_s = freq_thz * 2 * np.pi
