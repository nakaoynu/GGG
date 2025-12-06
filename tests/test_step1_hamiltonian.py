import sys
import os
import pathlib

# ==============================================================================
# 📁 パス設定: unified_weighted_bayesian_fitting.py をインポート
current_dir = pathlib.Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

try:
    import unified_weighted_bayesian_fitting as wbf
    print(f"✅ モジュール読み込み成功: {wbf.__file__}")
except ImportError as e:
    print("❌ エラー: 親ディレクトリから unified_weighted_bayesian_fitting をインポートできません。")
    print(f"検索パス: {sys.path}")
    exit()
import numpy as np
import pytensor.tensor as pt
import pytensor
import warnings

# --- 物理定数 (wbfから取得) ---
kB = wbf.kB
muB = wbf.muB
s = 7/2  # 8次元行列用のスピン量子数 (s=7/2)

# ==============================================================================
# 🛠️ Step 1: ハミルトニアン生成関数の PyTensor 化
# ==============================================================================
def get_hamiltonian_pt(B_ext_z, g_factor, B4, B6):
    """
    微分可能なハミルトニアン生成関数 (PyTensor版)
    
    NumPy版との主な違い:
    1. 配列の代入に `A[i,j]=x` ではなく `pt.set_subtensor` を使う
    2. 定数配列は `pt.as_tensor_variable` で変換する
    """
    
    # m_values: 3.5, 2.5, ..., -3.5 (定数として扱う)
    # PyTensor計算グラフ内で定数として使用
    m_values = pt.as_tensor_variable(np.arange(s, -s - 1, -1))
    
    # --- 1. 対角行列 Sz ---
    Sz = pt.diag(m_values)
    
    # --- 2. Stevens Operator O40 (対角) ---
    # NumPy: np.diag([7, -13, -3, 9, 9, -3, -13, 7])
    O40_diag = pt.as_tensor_variable([7, -13, -3, 9, 9, -3, -13, 7])
    O40 = pt.as_tensor_variable(60.0) * pt.diag(O40_diag)
    
    # --- 3. Stevens Operator O44 (非対角) ---
    # PyTensorでは "Immutable (不変)" なので、zerosを作ってから値を埋め込む操作になる
    X_O44_base = pt.zeros((8, 8))
    
    # 値の定義 (NumPyのsqrtを使って定数計算しておく)
    val_sqrt35 = np.sqrt(35)
    val_5sqrt3 = 5 * np.sqrt(3)
    
    # set_subtensor(対象[インデックス], 値)
    # X_O44[3, 7] = ... と同じ意味
    X_O44 = pt.set_subtensor(X_O44_base[3, 7], val_sqrt35) #type: ignore
    X_O44 = pt.set_subtensor(X_O44[4, 0], val_sqrt35) #type: ignore
    X_O44 = pt.set_subtensor(X_O44[2, 6], val_5sqrt3) #type: ignore
    X_O44 = pt.set_subtensor(X_O44[5, 1], val_5sqrt3) #type: ignore
    
    # O44 = 12 * (X + X.T)
    O44 = pt.as_tensor_variable(12.0) * (X_O44 + X_O44.T) #type: ignore
    
    # --- 4. Stevens Operator O60 (対角) ---
    O60_diag = pt.as_tensor_variable([1, -5, 9, -5, -5, 9, -5, 1])
    O60 = pt.as_tensor_variable(1260.0) * pt.diag(O60_diag)
    
    # --- 5. Stevens Operator O64 (非対角) ---
    X_O64_base = pt.zeros((8, 8))
    
    val_3sqrt35 = 3 * np.sqrt(35)
    val_m7sqrt3 = -7 * np.sqrt(3)
    
    X_O64 = pt.set_subtensor(X_O64_base[3, 7], val_3sqrt35) #type: ignore
    X_O64 = pt.set_subtensor(X_O64[4, 0], val_3sqrt35) #type: ignore
    X_O64 = pt.set_subtensor(X_O64[2, 6], val_m7sqrt3) #type: ignore
    X_O64 = pt.set_subtensor(X_O64[5, 1], val_m7sqrt3) #type: ignore
    
    O64 = pt.as_tensor_variable(60.0) * (X_O64 + X_O64.T) #type: ignore
    
    # --- 6. ハミルトニアンの合算 ---
    # パラメータがスカラー(pt.dscalar)でも行列と演算できるようにブロードキャストされる
    H_cf = (B4 * kB) * (O40 + pt.as_tensor_variable(5.0) * O44) + (B6 * kB) * (O60 - pt.as_tensor_variable(21.0) * O64)
    H_zee = g_factor * muB * B_ext_z * Sz
    
    return H_cf + H_zee

# ==============================================================================
# 🧪 Step 2: 比較検証テストの実行
# ==============================================================================
def run_test():
    print("\n=== Step 1: ハミルトニアン生成の PyTensor 化テスト ===")
    
    # 1. テスト用パラメータ (物理的にありそうな値)
    B_test = 9.0
    g_test = 1.95
    B4_test = 0.000576
    B6_test = 0.000050
    
    print(f"Parameters: B={B_test}, g={g_test}, B4={B4_test}, B6={B6_test}")

    # ---------------------------------------------------------
    # A. NumPy版 (正解データ) の計算
    # ---------------------------------------------------------
    print("計算中: NumPy版 (Original)...")
    H_numpy = wbf.get_hamiltonian(B_test, g_test, B4_test, B6_test)
    E_numpy = np.linalg.eigvalsh(H_numpy) # 固有値
    E_numpy.sort()
    
    # ---------------------------------------------------------
    # B. PyTensor版 (検証対象) の計算
    # ---------------------------------------------------------
    print("計算中: PyTensor版 (New)...")
    
    # シンボル変数の定義
    b_sym = pt.dscalar('b')
    g_sym = pt.dscalar('g')
    b4_sym = pt.dscalar('b4')
    b6_sym = pt.dscalar('b6')
    
    # 計算グラフの構築
    H_graph = get_hamiltonian_pt(b_sym, g_sym, b4_sym, b6_sym)
    E_graph, _ = pt.linalg.eigh(H_graph) # 固有値と固有ベクトル
    
    # 関数としてコンパイル (数値を入力できるようにする)
    calc_func = pytensor.function(
        inputs=[b_sym, g_sym, b4_sym, b6_sym],
        outputs=[H_graph, E_graph]
    )
    
    # 実行
    H_pt, E_pt = calc_func(B_test, g_test, B4_test, B6_test)
    E_pt.sort() # 念のためソート
    
    # ---------------------------------------------------------
    # C. 結果の比較
    # ---------------------------------------------------------
    print("\n--- 検証結果 ---")
    
    # 1. 行列要素の差分
    diff_H = np.abs(H_numpy - H_pt)
    max_diff_H = np.max(diff_H)
    print(f"ハミルトニアン最大誤差: {max_diff_H:.3e}")
    
    if np.allclose(H_numpy, H_pt, atol=1e-15):
        print("✅ [OK] ハミルトニアン行列は完全に一致しています。")
    else:
        print("❌ [NG] ハミルトニアン行列に不一致があります。実装を確認してください。")

    # 2. 固有値の差分
    diff_E = np.abs(E_numpy - E_pt)
    max_diff_E = np.max(diff_E)
    print(f"固有値最大誤差        : {max_diff_E:.3e}")
    
    if np.allclose(E_numpy, E_pt, atol=1e-15):
        print("✅ [OK] 固有値は完全に一致しています。")
    else:
        print("❌ [NG] 固有値に不一致があります。")

if __name__ == "__main__":
    run_test()