# test_vectorized_calculation.py
# ベクトル化計算の正確性と速度をテストするスクリプト

import numpy as np
import time

# 物理定数
kB = 1.380649e-23
muB = 9.274010e-24
hbar = 1.054571e-34
mu0 = 4.0 * np.pi * 1e-7
s = 3.5

THZ_TO_RAD_S = 2.0 * np.pi * 1e12
RAD_S_TO_THZ = 1.0 / THZ_TO_RAD_S

def get_hamiltonian(B_ext_z: float, g_factor: float, B4: float, B6: float) -> np.ndarray:
    """ハミルトニアンを計算する"""
    n_states = int(2 * s + 1)
    m_values = np.arange(s, -s - 1, -1)
    Sz = np.diag(m_values)
    
    O40 = 60 * np.diag([7, -13, -3, 9, 9, -3, -13, 7])
    X_O44 = np.zeros((8, 8))
    X_O44[3, 7], X_O44[4, 0] = np.sqrt(35), np.sqrt(35)
    X_O44[2, 6], X_O44[5, 1] = 5 * np.sqrt(3), 5 * np.sqrt(3)
    O44 = 12 * (X_O44 + X_O44.T)
    O60 = 1260 * np.diag([1, -5, 9, -5, -5, 9, -5, 1])
    X_O64 = np.zeros((8, 8))
    X_O64[3, 7], X_O64[4, 0] = 3 * np.sqrt(35), 3 * np.sqrt(35)
    X_O64[2, 6], X_O64[5, 1] = -7 * np.sqrt(3), -7 * np.sqrt(3)
    O64 = 60 * (X_O64 + X_O64.T)
    
    H_cf = (B4 * kB) * (O40 + 5 * O44) + (B6 * kB) * (O60 - 21 * O64)
    H_zee = g_factor * muB * B_ext_z * Sz
    return H_cf + H_zee

def calculate_susceptibility_loop(freq_thz_array: np.ndarray, H: np.ndarray, T: float, 
                                   gamma_thz_array: np.ndarray) -> np.ndarray:
    """磁気感受率を計算（旧：ループ版）"""
    eigenvalues, _ = np.linalg.eigh(H)
    eigenvalues -= np.min(eigenvalues)
    eigenvalues = np.clip(eigenvalues / (kB * T), -700, 700)
    
    Z = np.sum(np.exp(-eigenvalues))
    populations = np.exp(-eigenvalues) / Z
    delta_E = (eigenvalues[1:] - eigenvalues[:-1]) * kB * T
    delta_pop = populations[1:] - populations[:-1]
    
    omega_0_rad = delta_E / hbar
    freq_0_thz = omega_0_rad * RAD_S_TO_THZ
    
    s_val = 3.5
    m_vals = np.arange(s_val, -s_val, -1)
    transition_strength = (s_val + m_vals) * (s_val - m_vals + 1)
    
    numerator = delta_pop * transition_strength
    finite_mask = np.isfinite(numerator) & np.isfinite(freq_0_thz) & np.isfinite(gamma_thz_array)
    numerator = numerator[finite_mask]
    freq_0_filtered = freq_0_thz[finite_mask]
    gamma_filtered = gamma_thz_array[finite_mask]
    
    if len(numerator) == 0:
        return np.zeros_like(freq_thz_array, dtype=complex)
    
    # ループ版（旧）
    chi_array = np.zeros_like(freq_thz_array, dtype=complex)
    for i, freq_thz in enumerate(freq_thz_array):
        if not np.isfinite(freq_thz):
            continue
        denominator = freq_0_filtered - freq_thz - 1j * gamma_filtered
        denominator[np.abs(denominator) < 1e-10] = 1e-10 + 1j * 1e-10
        chi_array[i] = np.sum(numerator / denominator)
    
    return -chi_array

def calculate_susceptibility_vectorized(freq_thz_array: np.ndarray, H: np.ndarray, T: float, 
                                         gamma_thz_array: np.ndarray) -> np.ndarray:
    """磁気感受率を計算（新：ベクトル化版）"""
    eigenvalues, _ = np.linalg.eigh(H)
    eigenvalues -= np.min(eigenvalues)
    eigenvalues = np.clip(eigenvalues / (kB * T), -700, 700)
    
    Z = np.sum(np.exp(-eigenvalues))
    populations = np.exp(-eigenvalues) / Z
    delta_E = (eigenvalues[1:] - eigenvalues[:-1]) * kB * T
    delta_pop = populations[1:] - populations[:-1]
    
    omega_0_rad = delta_E / hbar
    freq_0_thz = omega_0_rad * RAD_S_TO_THZ
    
    s_val = 3.5
    m_vals = np.arange(s_val, -s_val, -1)
    transition_strength = (s_val + m_vals) * (s_val - m_vals + 1)
    
    numerator = delta_pop * transition_strength
    finite_mask = np.isfinite(numerator) & np.isfinite(freq_0_thz) & np.isfinite(gamma_thz_array)
    numerator = numerator[finite_mask]
    freq_0_filtered = freq_0_thz[finite_mask]
    gamma_filtered = gamma_thz_array[finite_mask]
    
    if len(numerator) == 0:
        return np.zeros_like(freq_thz_array, dtype=complex)
    
    # ベクトル化版（新）
    freq_diff = freq_0_filtered[None, :] - freq_thz_array[:, None]
    denominator = freq_diff - 1j * gamma_filtered[None, :]
    
    small_mask = np.abs(denominator) < 1e-10
    denominator[small_mask] = 1e-10 + 1j * 1e-10
    
    chi_array = -np.sum(numerator[None, :] / denominator, axis=1)
    
    return chi_array

def test_correctness():
    """計算結果の正確性をテスト"""
    print("=" * 60)
    print("テスト1: 計算結果の正確性")
    print("=" * 60)
    
    # テストパラメータ
    B = 9.0
    T = 4.0
    g_factor = 2.0
    B4 = 0.00202
    B6 = -0.0000120
    gamma_thz = np.full(7, 0.018)
    freq_thz = np.linspace(0.2, 0.5, 500)
    
    H = get_hamiltonian(B, g_factor, B4, B6)
    
    # 両方の方法で計算
    chi_loop = calculate_susceptibility_loop(freq_thz, H, T, gamma_thz)
    chi_vec = calculate_susceptibility_vectorized(freq_thz, H, T, gamma_thz)
    
    # 差を計算
    diff = np.abs(chi_loop - chi_vec)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"  周波数点数: {len(freq_thz)}")
    print(f"  最大差分: {max_diff:.2e}")
    print(f"  平均差分: {mean_diff:.2e}")
    
    if max_diff < 1e-10:
        print("  ✅ 結果一致: 正確性テスト合格")
        return True
    else:
        print("  ❌ 結果不一致: 正確性テスト失敗")
        return False

def test_speed():
    """計算速度をテスト"""
    print("\n" + "=" * 60)
    print("テスト2: 計算速度の比較")
    print("=" * 60)
    
    # テストパラメータ
    B = 9.0
    T = 4.0
    g_factor = 2.0
    B4 = 0.00202
    B6 = -0.0000120
    gamma_thz = np.full(7, 0.018)
    freq_thz = np.linspace(0.2, 0.5, 500)
    
    H = get_hamiltonian(B, g_factor, B4, B6)
    
    n_iterations = 1000
    
    # ループ版の速度測定
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = calculate_susceptibility_loop(freq_thz, H, T, gamma_thz)
    time_loop = time.perf_counter() - start
    
    # ベクトル化版の速度測定
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = calculate_susceptibility_vectorized(freq_thz, H, T, gamma_thz)
    time_vec = time.perf_counter() - start
    
    speedup = time_loop / time_vec
    
    print(f"  反復回数: {n_iterations}")
    print(f"  周波数点数: {len(freq_thz)}")
    print(f"  ループ版: {time_loop:.3f} 秒 ({time_loop/n_iterations*1000:.3f} ms/回)")
    print(f"  ベクトル化版: {time_vec:.3f} 秒 ({time_vec/n_iterations*1000:.3f} ms/回)")
    print(f"  速度向上: {speedup:.1f}倍")
    
    if speedup > 1:
        print(f"  ✅ ベクトル化により {speedup:.1f}倍 高速化")
    else:
        print(f"  ⚠️ ベクトル化による高速化なし（{speedup:.2f}倍）")
    
    return speedup

def test_with_different_sizes():
    """異なるデータサイズでの速度比較"""
    print("\n" + "=" * 60)
    print("テスト3: データサイズ別の速度比較")
    print("=" * 60)
    
    B = 9.0
    T = 4.0
    g_factor = 2.0
    B4 = 0.00202
    B6 = -0.0000120
    gamma_thz = np.full(7, 0.018)
    
    H = get_hamiltonian(B, g_factor, B4, B6)
    
    sizes = [100, 300, 500, 1000, 2000]
    n_iterations = 500
    
    print(f"  {'周波数点数':>10} | {'ループ(ms)':>12} | {'ベクトル(ms)':>12} | {'速度向上':>8}")
    print("  " + "-" * 52)
    
    for size in sizes:
        freq_thz = np.linspace(0.2, 0.5, size)
        
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = calculate_susceptibility_loop(freq_thz, H, T, gamma_thz)
        time_loop = (time.perf_counter() - start) / n_iterations * 1000
        
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = calculate_susceptibility_vectorized(freq_thz, H, T, gamma_thz)
        time_vec = (time.perf_counter() - start) / n_iterations * 1000
        
        speedup = time_loop / time_vec
        print(f"  {size:>10} | {time_loop:>12.3f} | {time_vec:>12.3f} | {speedup:>7.1f}x")

def main():
    print("\n" + "=" * 60)
    print("ベクトル化計算のテスト")
    print("=" * 60)
    
    # 正確性テスト
    correctness_ok = test_correctness()
    
    if not correctness_ok:
        print("\n⚠️ 正確性テストに失敗したため、速度テストをスキップします")
        return
    
    # 速度テスト
    speedup = test_speed()
    
    # データサイズ別テスト
    test_with_different_sizes()
    
    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)
    
    if correctness_ok and speedup > 1:
        print("✅ すべてのテストに合格しました。AWS環境での実行準備完了です。")
    else:
        print("⚠️ 一部のテストに問題があります。確認してください。")

if __name__ == "__main__":
    main()
