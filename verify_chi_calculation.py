import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# --- 0. プロット設定 ---
plt.rcParams['font.family'] = "Meiryo"
plt.rcParams['figure.dpi'] = 100

# --- 1. 定数と演算子の定義 (変更なし) ---
# (省略... 前回のコードと同じ)
# 物理定数
kB = 1.380649e-23
muB = 9.274010e-24
hbar = 1.054571e-34
c = 299792458
mu0 = 4.0 * np.pi * 1e-7

# 物質・シミュレーションパラメータ
g_factor = 1.95
s = 3.5
T = 35.0
B_ext = 7.8
gamma = 110e9
N_spin_exp = 24/1.238 * 1e27
N_spin = N_spin_exp * 10
G0 = mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)

def get_hamiltonian(B_ext_z):
    Sz = np.diag(np.arange(s, -s - 1, -1))
    B4_param = 0.8 / 240 * 0.606; B6_param = 0.04 / 5040 * -1.513
    B4 = B4_param; B6 = B6_param
    O04 = 60 * np.diag([7, -13, -3, 9, 9, -3, -13, 7])
    X_O44 = np.zeros((8, 8)); X_O44[3, 7], X_O44[4, 0] = np.sqrt(35), np.sqrt(35); X_O44[2, 6], X_O44[5, 1] = 5*np.sqrt(3), 5*np.sqrt(3); O44 = 12 * (X_O44 + X_O44.T)
    O06 = 1260 * np.diag([1, -5, 9, -5, -5, 9, -5, 1])
    X_O46 = np.zeros((8, 8)); X_O46[3, 7], X_O46[4, 0] = 3*np.sqrt(35), 3*np.sqrt(35); X_O46[2, 6], X_O46[5, 1] = -7*np.sqrt(3), -7*np.sqrt(3); O46 = 60 * (X_O46 + X_O46.T)
    H_cf = (B4 * kB) * (O04 + 5 * O44) + (B6 * kB) * (O06 - 21 * O46)
    H_zee = g_factor * muB * B_ext_z * Sz
    return H_cf + H_zee

# --- 2. 2種類の計算方法をそれぞれ関数として定義  ---
def calculate_chi_loop(omega_array, H, T):
    print("方法A: forループによる計算を開始...")
    chi_R_list = []
    eigenvalues, _ = np.linalg.eigh(H)
    eigenvalues -= np.min(eigenvalues)
    Z = np.sum(np.exp(-eigenvalues / (kB * T)))
    populations = np.exp(-eigenvalues / (kB * T)) / Z
    m_vals = np.arange(-s, s, 1)
    for omega in omega_array:
        chi = 0.0j
        for i in range(len(eigenvalues) - 1):
            m = m_vals[i]
            delta_E = eigenvalues[i+1] - eigenvalues[i]
            delta_pop = populations[i+1] - populations[i]
            omega_0 = delta_E / hbar
            transition_strength = (s + m) * (s - m + 1)
            numerator = G0 * delta_pop * transition_strength
            denominator = (omega_0 - omega) - (1j * gamma)
            chi += numerator / denominator
        chi_R_list.append(chi)
    print("方法A: 計算完了。")
    return chi_R_list

def calculate_chi_vectorized(omega_array, H, T):
    print("方法B: ベクトル化による計算を開始...")
    eigenvalues, _ = np.linalg.eigh(H)
    eigenvalues -= np.min(eigenvalues)
    Z = np.sum(np.exp(-eigenvalues / (kB * T)))
    populations = np.exp(-eigenvalues / (kB * T)) / Z
    delta_E = eigenvalues[1:] - eigenvalues[:-1]
    delta_pop = populations[1:] - populations[:-1]
    omega_0 = delta_E / hbar
    m_vals = np.arange(-s, s, 1)
    transition_strength = (s + m_vals) * (s - m_vals + 1)
    numerator = G0 * delta_pop * transition_strength
    denominator = (omega_0[:, np.newaxis] - omega_array) - (1j * gamma)
    chi_array = np.sum(numerator[:, np.newaxis] / denominator, axis=0)
    print("方法B: 計算完了。")
    return chi_array


# --- 3. メイン実行・比較ブロック ---
if __name__ == '__main__':
    # 計算する周波数範囲を定義
    freq_hz = np.arange(0.1e11, 9.0e11, 0.01e11)
    omega_rad_s = freq_hz * 2 * np.pi
    freq_thz = freq_hz / 1e12 # x軸用の周波数 (THz)

    # ハミルトニアンは一度だけ計算
    H = get_hamiltonian(B_ext)

    # 2つの方法でそれぞれ計算を実行
    chi_R_from_loop = calculate_chi_loop(omega_rad_s, H, T)
    chi_array_from_vector = calculate_chi_vectorized(omega_rad_s, H, T)

    # --- 4. 結果の比較 ---
    print("\n--- 結果の比較 ---")
    chi_R_np = np.array(chi_R_from_loop)
    if np.allclose(chi_R_np, chi_array_from_vector):
        print("✅ 検証成功: 2つの計算結果は、数値誤差の範囲で完全に一致しました。")
    else:
        print("❌ 検証失敗: 2つの計算結果に差異があります。")
        difference = np.abs(chi_R_np - chi_array_from_vector)
        print(f"   最大差分: {np.max(difference)}")

    # 4.5. 結果をCSVファイルに保存
    print("\n結果をCSVファイルに保存します...")
    try:
        # 保存するデータを辞書形式で準備
        data_to_save = {
            'Frequency_THz': freq_thz,
            'Re_chi_loop': chi_R_np.real,
            'Im_chi_loop': chi_R_np.imag,
            'Re_chi_vectorized': chi_array_from_vector.real,
            'Im_chi_vectorized': chi_array_from_vector.imag,
            'Absolute_Difference': np.abs(chi_R_np - chi_array_from_vector)
        }
        
        # 辞書からpandasのDataFrameを作成
        df_results = pd.DataFrame(data_to_save)
        
        # DataFrameをCSVファイルに書き出し
        output_filename = 'verification_chi_results.csv'
        # index=False とすることで、DataFrameのインデックス(0, 1, 2...)がファイルに書き込まれるのを防ぐ
        df_results.to_csv(output_filename, index=False)
        
        print(f"✅ 正常に '{output_filename}' として保存されました。")

    except Exception as e:
        print(f"❌ CSVファイルの保存中にエラーが発生しました: {e}")

    # --- 5. グラフで視覚的に比較  ---
    print("\nグラフを生成して視覚的に比較します...")
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('2つの計算方法による磁気感受率の比較', fontsize=16)
    axs[0].plot(freq_thz, chi_R_np.real, label='方法A (ループ)', linestyle='-')
    axs[0].plot(freq_thz, chi_array_from_vector.real, label='方法B (ベクトル化)', linestyle='--')
    axs[0].set_ylabel('Re(${\chi}$)')
    axs[0].legend()
    axs[0].grid(True)
    axs[1].plot(freq_thz, chi_R_np.imag, label='方法A (ループ)', linestyle='-')
    axs[1].plot(freq_thz, chi_array_from_vector.imag, label='方法B (ベクトル化)', linestyle='--')
    axs[1].set_xlabel('周波数 (THz)')
    axs[1].set_ylabel('Im(${\chi}$)')
    axs[1].legend()
    axs[1].grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('verification_chi.png', dpi=300)
