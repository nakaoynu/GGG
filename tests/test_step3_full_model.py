import sys
import os
import pathlib
import numpy as np
import pytensor.tensor as pt
import pytensor

# パス設定
current_dir = pathlib.Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

try:
    from unified_weighted_bayesian_fitting import calculate_normalized_transmission as calc_trans_numpy
    from unified_weighted_bayesian_fitting import c, mu0
    print("✅ モジュール読み込み成功")
except ImportError:
    print("❌ エラー: モジュールが見つかりません")
    exit()

# ==============================================================================
# 🛠️ Step 3: 透過率計算の PyTensor 化
# ==============================================================================
def calculate_normalized_transmission_pt(omega, mu_r, d, eps_bg):
    """
    透過率計算の PyTensor版
    """
    # 定数のTensor化
    # c, mu0 はインポートしたものを使う
    c_pt = pt.as_tensor_variable(c)
    
    # 1. 複素屈折率とインピーダンス
    # mu_r は複素数 (complex128)
    
    # 安全対策: mu_r が異常値(NaN/Inf)の場合の対策は
    # PyTensorでは switch を使うが、今回は単純化してそのまま計算
    
    # eps_mu_product = eps_bg * mu_r
    # n_complex = sqrt(eps * mu)
    # impe = sqrt(mu / eps)
    
    eps_mu_product = eps_bg * mu_r
    
    # 注意: 負の実部を持つ平方根の分岐カット問題を防ぐため、
    # 元コードにあるような "0.1 + 1j*..." のような処理が必要なら switch で書く
    # 元コード: eps_mu_product = np.where(eps_mu_product.real > 0, eps_mu_product, 0.1 + 1j * eps_mu_product.imag)
    
    # PyTensorでの where (switch)
    condition = pt.gt(pt.real(eps_mu_product), 0.0)
    safe_product = pt.switch(condition, eps_mu_product, 0.1 + 1j * pt.imag(eps_mu_product))
    
    n_complex = pt.sqrt(safe_product + 0j)
    impe = pt.sqrt(mu_r / eps_bg + 0j)
    
    # 2. 位相因子 delta
    # lambda_0 = 2 * pi * c / omega
    # ゼロ除算回避: omega < 1e-12 のときは Inf にするなどの処理
    # ここでは omega > 0 前提で計算
    
    lambda_0 = (2 * np.pi * c_pt) / omega
    delta_raw = 2 * np.pi * n_complex * d / lambda_0
    
    # クリップ処理 (オーバーフロー防止)
    # pt.clip は実数部・虚数部それぞれに行う
    delta_real = pt.clip(pt.real(delta_raw), -700, 700)
    delta_imag = pt.clip(pt.imag(delta_raw), -700, 700)
    delta = delta_real + 1j * delta_imag
    
    # 3. 透過率公式
    # T = | 4n / ((1+n)^2 exp(-id) - (1-n)^2 exp(id)) |^2  ... (インピーダンス形式)
    
    numerator = 4 * impe
    exp_pos = pt.exp(-1j * delta)
    exp_neg = pt.exp(1j * delta)
    
    denominator = (1 + impe)**2 * exp_pos - (1 - impe)**2 * exp_neg
    
    t_complex = numerator / denominator
    transmission = pt.abs(t_complex)**2
    
    # 4. 正規化 (0-1)
    # min-max 正規化
    t_min = pt.min(transmission)
    t_max = pt.max(transmission)
    
    # ゼロ除算防止 (max > min)
    norm_trans = pt.switch(
        pt.gt(t_max - t_min, 1e-20),
        (transmission - t_min) / (t_max - t_min),
        0.5 # 差がない場合は0.5
    )
    
    # 物理的な範囲 (0, 1) にクリップ
    return pt.clip(norm_trans, 0.0, 1.0)

# ==============================================================================
# 🧪 テスト実行
# ==============================================================================
def run_test():
    print("\n=== Step 3: 透過率計算の PyTensor 化テスト ===")
    
    # テストデータ
    freq_thz = np.linspace(0.1, 1.0, 100)
    omega_val = freq_thz * 1e12 * 2 * np.pi
    
    # 入力としての透磁率 mu_r (適当な複素数配列を作成)
    # 共鳴っぽいうねりを入れる
    chi_dummy = 0.1 / (0.5e12 * 2 * np.pi - omega_val - 1j * 0.05e12)
    mu_r_val = 1.0 + chi_dummy
    
    d_val = 157.8e-6
    eps_bg_val = 14.2
    
    # A. NumPy版
    print("計算中: NumPy版...")
    trans_numpy = calc_trans_numpy(omega_val, mu_r_val, d_val, eps_bg_val)
    
    # B. PyTensor版
    print("計算中: PyTensor版...")
    omega_sym = pt.dvector('omega')
    mu_r_sym = pt.zvector('mu_r') # 複素数ベクトル
    d_sym = pt.dscalar('d')
    eps_sym = pt.dscalar('eps')
    
    trans_graph = calculate_normalized_transmission_pt(omega_sym, mu_r_sym, d_sym, eps_sym)
    
    calc_func = pytensor.function(
        inputs=[omega_sym, mu_r_sym, d_sym, eps_sym],
        outputs=trans_graph
    )
    
    trans_pt = calc_func(omega_val, mu_r_val, d_val, eps_bg_val)
    
    # C. 比較
    print("\n--- 検証結果 ---")
    diff = np.abs(trans_numpy - trans_pt)
    max_diff = np.max(diff)
    print(f"最大誤差: {max_diff:.3e}")
    
    if np.allclose(trans_numpy, trans_pt, atol=1e-10):
        print("✅ [OK] 透過スペクトル計算は一致しています。")
    else:
        print("❌ [NG] 不一致があります。")

if __name__ == "__main__":
    run_test()