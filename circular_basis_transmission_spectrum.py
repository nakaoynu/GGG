#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- フォント設定 ---
try:
    plt.rcParams['font.family'] = "Meiryo"
except RuntimeError:
    print("日本語フォント 'Meiryo' が見つかりません。デフォルトのフォントを使用します。")

# --- 1. 演算子とハミルトニアンを定義する関数 ---

def get_spin_operators(spin):
    # (変更なし)
    dim = int(2 * spin + 1)
    m = np.arange(spin, -spin - 1, -1)
    Sz = np.diag(m)
    sp_diag = np.sqrt(spin * (spin + 1) - m[:-1] * (m[:-1] - 1))
    sm_diag = np.sqrt(spin * (spin + 1) - m[1:] * (m[1:] + 1))
    Sp = np.diag(sp_diag, k=1)
    Sm = np.diag(sm_diag, k=-1)
    Sx = 0.5 * (Sp + Sm)
    Sy = -0.5j * (Sp - Sm)
    return Sx, Sy, Sz, Sp, Sm

def get_stevens_operators():
    # (変更なし)
    O04 = 60 * np.diag([7, -13, -3, 9, 9, -3, -13, 7])
    X = np.zeros((8, 8)); X[3, 7] = np.sqrt(35); X[4, 0] = np.sqrt(35); X[2, 6] = 5 * np.sqrt(3); X[5, 1] = 5 * np.sqrt(3)
    O44 = 12 * (X + X.T)
    O06 = 1260 * np.diag([1, -5, 9, -5, -5, 9, -5, 1])
    X = np.zeros((8, 8)); X[3, 7] = 3 * np.sqrt(35); X[4, 0] = 3 * np.sqrt(35); X[2, 6] = -7 * np.sqrt(3); X[5, 1] = -7 * np.sqrt(3)
    O46 = 60 * (X + X.T)
    return O04, O44, O06, O46

def get_hamiltonian(B_ext_z, s, g_factor, kB, muB, B4, B6):
    # (変更なし)
    _, _, Sz, _, _ = get_spin_operators(s)
    O04, O44, O06, O46 = get_stevens_operators()
    H_cf = (B4 * kB) * (O04 + 5 * O44) + (B6 * kB) * (O06 - 21 * O46)
    H_zee = g_factor * muB * B_ext_z * Sz
    return H_cf + H_zee

# --- 2. 物理量を計算する関数 ---

def calculate_susceptibility_tensor(omega, H, T, S_ops, N_spin, g_factor, muB, kB, hbar, gamma):
    """
    【新しい関数】磁気感受率テンソル chi_ij を計算する
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    eigenvalues -= np.min(eigenvalues)
    
    Z = np.sum(np.exp(-eigenvalues / (kB * T)))
    if Z == 0: return np.zeros((3, 3), dtype=complex)
    populations = np.exp(-eigenvalues / (kB * T)) / Z
    
    # Sx, Sy, Sz をエネルギー固有基底に変換
    S_eig_basis = [evec.conj().T @ S @ evec for S, evec in zip(S_ops, [eigenvectors]*3)]

    chi_tensor = np.zeros((3, 3), dtype=complex)
    
    for n in range(len(eigenvalues)):
        for m in range(len(eigenvalues)):
            if n == m: continue
                
            pop_diff = populations[m] - populations[n]
            denominator = (eigenvalues[n] - eigenvalues[m]) - hbar * (omega + 1j * gamma)
            if np.abs(denominator) < 1e-30: continue

            # テンソルの各成分(ij)を計算
            for i in range(3):
                for j in range(3):
                    # 遷移要素の積 <n|Si|m> * <m|Sj|n> = <n|Si|m> * (<n|Sj|m>)*
                    transition_term = S_eig_basis[i][n, m] * np.conj(S_eig_basis[j][n, m])
                    chi_tensor[i, j] += pop_diff * transition_term / denominator
    
    prefactor = N_spin * (g_factor * muB)**2 / hbar
    return prefactor * chi_tensor

def calculate_transmittance(omega, chi, eps_bg, d, c):
    # (変更なし)
    if np.abs(1 - chi) < 1e-9: mu_r = np.inf
    else: mu_r = 1 / (1 - chi)
    n_complex = np.sqrt(eps_bg * mu_r)
    if np.abs(1 + n_complex) < 1e-9: return 0.0
    delta = n_complex * omega * d / c
    t = 4 * n_complex * np.exp(-1j * delta) / ((1 + n_complex)**2)
    return np.abs(t)**2

# --- 3. メイン処理 ---
def main():
    # パラメータ設定 (変更なし)
    kB = 1.380649e-23; muB = 9.274010e-24; hbar = 1.054571e-34; c = 299792458
    g_factor = 1.95; eps_bg = 14.44; s = 3.5; N_spin = 1.26e28; d = 0.1578e-3; gamma = 5.0e11
    B4 = (0.8 / 240) * 0.606; B6 = (0.04 / 5040) * -1.513
    T_array = [35, 50, 75, 95, 200, 250]; B_ext = 7.8
    
    print("シミュレーションを開始します...")
    freq_thz = np.linspace(0.1, 1.5, 400)
    omega_rad_s = freq_thz * 1e12 * 2 * np.pi
    
    Sx, Sy, Sz, _, _ = get_spin_operators(s)
    S_ops = [Sx, Sy, Sz] # 演算子をリストにまとめる
    H = get_hamiltonian(B_ext, s, g_factor, kB, muB, B4, B6)
    
    transmittance_vectors = []

    for T in T_array:
        print(f"  T = {T} K の計算中...")
        transmittance_data = []
        for omega in omega_rad_s:
            # ★ 磁気感受率テンソルを計算
            chi_tensor = calculate_susceptibility_tensor(
                omega, H, T, S_ops, N_spin, g_factor, muB, kB, hbar, gamma)
            
            # ★ テンソルから右円偏光感受率を計算 (chi_R = chi_xx - i*chi_xy)
            chi_xx = chi_tensor[0, 0]
            chi_xy = chi_tensor[0, 1]
            chi_R = chi_xx - 1j * chi_xy
            
            T_R = calculate_transmittance(omega, chi_R, eps_bg, d, c)
            transmittance_data.append(T_R)
            
        transmittance_vectors.append(np.array(transmittance_data))
    
    # グラフ描画 (変更なし)
    print("グラフを描画し、'final_spectrum_tensor_calc.png' に保存します...")
    plt.figure(figsize=(10, 8))
    for i, T_data in enumerate(transmittance_vectors):
        offset = i * 0.5 
        plt.plot(freq_thz, T_data + offset, label=f"T={T_array[i]} K")
    plt.title(f"絶対透過率スペクトル (B={B_ext} T, B-form, 右円偏光)")
    plt.xlabel("周波数 (THz)")
    plt.ylabel("透過率 T (オフセットあり)")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.savefig('final_spectrum_tensor_calc.png', dpi=300)
    print("保存が完了しました。")

if __name__ == '__main__':
    main()