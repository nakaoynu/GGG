#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 5種類の温度に対するスペクトルの表示(B)
# 高磁場下での透過率スペクトルを計算し、グラフにプロットするコード

import numpy as np
import matplotlib
# GUIバックエンドがない環境でも動作するように 'Agg' を指定
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- フォント設定の堅牢化 ---
try:
    plt.rcParams['font.family'] = "Meiryo"
except RuntimeError:
    print("日本語フォント 'Meiryo' が見つかりません。デフォルトのフォントを使用します。")
    # 必要に応じて、ご自身の環境にインストールされている日本語フォント名に変更してください
    # (例: "Yu Gothic", "Hiragino Sans", "IPAexGothic")

# --- 定数と行列の定義 ---

# スティーブンス演算子
O04 = 60 * np.diag([7, -13, -3, 9, 9, -3, -13, 7])

X_44_upper = np.zeros((8, 8))
X_44_upper[3, 7] = np.sqrt(35)
X_44_upper[4, 0] = np.sqrt(35)
X_44_upper[2, 6] = 5 * np.sqrt(3)
X_44_upper[5, 1] = 5 * np.sqrt(3)
O44 = 12 * (X_44_upper + X_44_upper.T)

O06 = 1260 * np.diag([1, -5, 9, -5, -5, 9, -5, 1])

X_46_upper = np.zeros((8, 8))
X_46_upper[3, 7] = 3 * np.sqrt(35)
X_46_upper[4, 0] = 3 * np.sqrt(35)
X_46_upper[2, 6] = -7 * np.sqrt(3)
X_46_upper[5, 1] = -7 * np.sqrt(3)
O46 = 60 * (X_46_upper + X_46_upper.T)

# スピン演算子
Sz = np.diag([3.5, 2.5, 1.5, 0.5, -0.5, -1.5, -2.5, -3.5])

# Onnに与えられた係数
B4_base = 0.8 / 240
B6_base = 0.04 / 5040
factor_b4 = 0.6060886
factor_b6 = -1.513
B4 = B4_base * factor_b4  # [K]
B6 = B6_base * factor_b6  # [K]

# 物理定数とシミュレーションパラメータ
g = 1.95
eps_BG = 11.5
c = 299792458
kB = 1.381e-23
muB = 9.274e-24
hbar = 1.055e-34
T_array = [35, 50, 75, 95, 200, 250]
B_array = [0.0, 7.8]
g0_const = 6.0e11
gamma = 2.5e11
d = 0.1578e-3
s = 3.5

# --- ヘルパー関数 ---

def Pm(Em, Z, T):
    """ポピュレーション（占有確率）を計算する関数"""
    if Z == 0:
        return 0
    return np.exp(-Em / (kB * T)) / Z

# --- メイン処理 ---

def main():
    """シミュレーションのメイン処理を実行する関数"""
    print("シミュレーションを開始します...")
    
    omega_array = []
    value_vector = [[] for _ in range(len(T_array))]

    # 結晶場ハミルトニアンは温度に依存しないため、ループの外で計算
    H_CF = (B4 * kB) * (O04 + 5.0 * O44) + (B6 * kB) * (O06 - 21.0 * O46)
    
    # 周波数配列の作成
    freq_range = np.arange(0.1e12, 7e12, 0.01e12) # 周波数ステップを少し細かく

    for i, T in enumerate(T_array):
        print(f"  T = {T} K の計算中...")
        value_array = []
        """
        # --- 背景透過係数の計算 (B=0) ---
        H_bg = H_CF  # B=0 の場合, H_Zee は 0
        eigenvalues_bg, _ = np.linalg.eigh(H_bg)
        Z_bg = np.sum(np.exp(-eigenvalues_bg / (kB * T)))
        
        # B=0, g0=0 なので chi=0 となり、t_BG は周波数にのみ依存する
        n_bg = np.sqrt(eps_BG) # mu=1
        impe_bg = 1 / np.sqrt(eps_BG)
        
        t_BG_list = []
        for omega in freq_range:
            lamda = c / (omega / (2 * np.pi))
            delta_bg = 2 * np.pi * n_bg * d / lamda
            t_bg = np.exp(delta_bg * 1j) * 4 * impe_bg / (1 + impe_bg)**2
            t_BG_list.append(t_bg)
        """

        # --- 実際の透過率の計算 (B > 0) ---
        B = B_array[1]
        g0 = np.sqrt(B) * g0_const / np.sqrt(21.5) # この 21.5 は物理的意味の確認が必要
        H_Zee = (g * muB * B) * Sz
        H = H_CF + H_Zee
        eigenvalues, _ = np.linalg.eigh(H)
        Z = np.sum(np.exp(-eigenvalues / (kB * T)))

        for j, omega in enumerate(freq_range):
            chi = 0.0
            for num, m_val in enumerate(np.arange(-3.5, 4.0, 1.0)):
                if num + 1 < len(eigenvalues):
                    delta_E = eigenvalues[num + 1] - eigenvalues[num]
                    omega_0 = delta_E / hbar
                    denominator = (omega_0 - omega - (1j * gamma / 2))
                    if np.abs(denominator) > 1e-30:
                        term = (4 * g0**2 / (2 * np.pi * gamma) * (s + m_val) * (s - m_val + 1) * (Pm(eigenvalues[num + 1], Z, T) - Pm(eigenvalues[num], Z, T)))
                        chi += term / denominator
            
            chi = -chi
            
            if np.abs(1 - chi) < 1e-9: # ゼロ割を避ける
                mu = np.inf
            else:
                mu = 1 / (1 - chi)

            n = np.sqrt(eps_BG * mu)
            impe = np.sqrt(mu / eps_BG)
            lamda = c / (omega / (2 * np.pi))
            delta = 2 * np.pi * n * d / lamda
            t = np.exp(delta * 1j) * 4 * impe / (1 + impe)**2
            
            y = abs(t - t_BG_list[j])**2
            value_array.append(y)

            if i == 0:
                omega_array.append(omega / (2 * np.pi * 1e12)) # THzに変換

        # --- ゼロ除算エラーの防止 ---
        min_val = np.min(value_array)
        max_val = np.max(value_array)

        if max_val == min_val:
            # 計算結果が平坦な場合、ゼロ除算を避ける
            normalized_values = np.zeros_like(value_array) + i
        else:
            normalized_values = (np.array(value_array) - min_val) / (max_val - min_val) + i
            
        value_vector[i] = normalized_values

    print("グラフを描画し、'f_B_spectrum.png' に保存します...")
    # --- グラフ描画 ---
    plt.figure(figsize=(10, 8))
    for i, values in enumerate(value_vector):
        plt.plot(omega_array, values, label=f"T={T_array[i]} K")
        
        # ピーク検出
        peaks, _ = find_peaks(values, height=i+0.1, distance=5)
        peak_frequencies = [omega_array[p] for p in peaks]
        peak_values = [values[p] for p in peaks]
        
        # ピークをプロット
        plt.scatter(peak_frequencies, peak_values, color='red', zorder=5)

    plt.title(f"透過スペクトル (B={B_array[1]} T, B-form)")
    plt.xlabel("周波数 (THz)")
    plt.ylabel("$|t-t_{BG}|^2$ (規格化・オフセットあり)")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.savefig('f_B_spectrum.png', dpi=300)
    print("保存が完了しました。")

# --- スクリプト実行のエントリーポイント ---
if __name__ == '__main__':
    main()