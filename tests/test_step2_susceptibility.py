import sys
import os
import pathlib
import numpy as np
import pytensor.tensor as pt
import pytensor

# =========================================================
# 🛠️ パス設定
# =========================================================
current_dir = pathlib.Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

try:
    # 比較対象として元の関数をインポート
    from unified_weighted_bayesian_fitting import calculate_susceptibility as calc_sus_numpy
    from unified_weighted_bayesian_fitting import get_hamiltonian as get_hamiltonian_numpy
    print("✅ モジュール読み込み成功")
except ImportError as e:
    print(f"❌ エラー: モジュールが見つかりません。\n詳細: {e}")
    exit()

# =========================================================
# 🧪 定数の定義
# =========================================================
kB = 1.380649e-23
hbar = 1.054571e-34
s = 3.5

# ==============================================================================
# 🛠️ Step 2: 磁気感受率計算の PyTensor 化
# ==============================================================================
def calculate_susceptibility_pt(omega, H, T, gamma_array):
    """
    微分可能な磁気感受率計算関数 (PyTensor版)
    
    ポイント:
    1. forループを使わず、ブロードキャストで行列計算する
    2. 固有値分解 pt.linalg.eigh を使う
    """
    
    # 定数のTensor化
    kB_pt = pt.as_tensor_variable(kB)
    hbar_pt = pt.as_tensor_variable(hbar)
    
    # 1. 固有値計算 (固有ベクトルは今回は使わないので捨てる)
    # eigh は通常、固有値を昇順で返す
    eigenvalues = pt.linalg.eigh(H)[0]
    
    # エネルギーの基準を最小値に合わせる
    eigenvalues = eigenvalues - pt.min(eigenvalues)
    
    # 2. 占有確率 (ボルツマン分布)
    # 数値安定性のためクリッピングなどは必要だが、一旦シンプルに書く
    # (実際の学習時には softplus 等のテクニックが必要になる場合もある)
    Z = pt.sum(pt.exp(-eigenvalues / (kB_pt * T)))
    populations = pt.exp(-eigenvalues / (kB_pt * T)) / Z
    
    # 3. 遷移エネルギーと占有数差 (隣接準位間)
    # values[1:] - values[:-1] の操作
    delta_E = eigenvalues[1:] - eigenvalues[:-1]
    delta_pop = populations[1:] - populations[:-1]
    
    # 4. 遷移強度 (スピン行列要素)
    # m_vals: 3.5, 2.5, ..., -3.5 (8要素)
    # 遷移は 7個 (m -> m-1)
    # 対応する m は 3.5, 2.5, ..., -2.5 (最初の7個)
    m_vals = pt.as_tensor_variable(np.arange(s, -s, -1)) # 7要素
    transition_strength = (s + m_vals) * (s - m_vals + 1.0)
    
    # 5. 共鳴周波数
    omega_0 = delta_E / hbar_pt
    
    # 6. 感受率 χ(ω) の計算 (ここがブロードキャストの肝)
    # omega   : (N_freq,)
    # omega_0 : (7,)
    # gamma   : (7,)
    # numerator: (7,)
    
    numerator = delta_pop * transition_strength
    
    # 次元を合わせて引き算 (N_freq, 7) の行列を作る
    # omega[:, None] -> (N_freq, 1)
    # omega_0[None, :] -> (1, 7)
    # これで (N_freq, 7) の行列ができる
    
    denominator = omega_0[None, :] - omega[:, None] - 1j * gamma_array[None, :]
    
    # 各遷移(7個)について和を取る -> (N_freq,) になる
    # sum(..., axis=1)
    chi_components = numerator[None, :] / denominator
    chi = pt.sum(chi_components, axis=1)
    
    # 符号は元のコードに合わせる (おそらく -chi が返されている)
    return -chi

# ==============================================================================
# 🧪 Step 2: 比較検証テストの実行
# ==============================================================================
def run_test():
    print("\n=== Step 2: 磁気感受率計算の PyTensor 化テスト ===")
    
    # 1. テスト用データ作成
    # 周波数: 0.1 THz ~ 1.0 THz を 100点
    freq_thz = np.linspace(0.1, 1.0, 100)
    omega_val = freq_thz * 1e12 * 2 * np.pi
    
    # パラメータ
    B_val = 9.0
    T_val = 4.0
    g_val = 2.0
    B4_val = 0.0005
    B6_val = 0.00005
    
    # Gamma (7要素)
    gamma_val = np.full(7, 0.1e12) # 適当な値
    
    # ハミルトニアン (NumPy版で作っておく)
    H_val = get_hamiltonian_numpy(B_val, g_val, B4_val, B6_val)
    
    # ---------------------------------------------------------
    # A. NumPy版 (正解)
    # ---------------------------------------------------------
    print("計算中: NumPy版...")
    chi_numpy = calc_sus_numpy(omega_val, H_val, T_val, gamma_val)
    
    # ---------------------------------------------------------
    # B. PyTensor版 (検証)
    # ---------------------------------------------------------
    print("計算中: PyTensor版...")
    
    # 入力シンボル
    omega_sym = pt.dvector('omega')
    H_sym = pt.dmatrix('H')
    T_sym = pt.dscalar('T')
    gamma_sym = pt.dvector('gamma')
    
    # グラフ構築
    chi_graph = calculate_susceptibility_pt(omega_sym, H_sym, T_sym, gamma_sym)
    
    # コンパイル
    calc_func = pytensor.function(
        inputs=[omega_sym, H_sym, T_sym, gamma_sym],
        outputs=chi_graph
    )
    
    # 実行
    chi_pt = calc_func(omega_val, H_val, T_val, gamma_val)
    
    # ---------------------------------------------------------
    # C. 比較
    # ---------------------------------------------------------
    print("\n--- 検証結果 ---")
    
    # 複素数なので絶対値の差を見る
    diff = np.abs(chi_numpy - chi_pt)
    max_diff = np.max(diff)
    
    print(f"最大誤差: {max_diff:.3e}")
    
    # 誤差許容値 (浮動小数点計算の順序違いで極小の誤差は出る)
    if np.allclose(chi_numpy, chi_pt, atol=1e-12):
        print("✅ [OK] 磁気感受率は一致しています。")
    else:
        print("❌ [NG] 不一致があります。")
        # デバッグ用
        print(f"NumPy先頭: {chi_numpy[0]}")
        print(f"PyTensor先頭: {chi_pt[0]}")

if __name__ == "__main__":
    run_test()