# Zeeman相互作用のみに基づく透過スペクトル計算
import numpy as np
import sympy as sp
import scipy.constants as const
import matplotlib.pyplot as plt

# --- 物理定数 ---
eps_bg = 14.4  # 背景の誘電率
mu_B = const.physical_constants['Bohr magneton'][0]  # ボーア磁子 (J/T)
mu_0 = const.mu_0  # 真空の透磁率
c = const.c  # 光速
k_B = const.k  # ボルツマン定数
hbar = const.hbar  # ディラック定数
N_spin = 24/1.238 * 1e27  # Gd^{3+}のスピン密度 (1/m^3)
s = 3.5  # スピンサイズ
d =  0.1578e-3  # GGGの膜厚 (m)
# --- 補助関数 ---
def calculate_d2A_daB2_analytical(a_B, s):
    """
    A の a_B に関する二階微分を、分析的に導出した式に基づいて計算する。
    
    A = sinh(a_B*s/2) * cosh(a_B*(s+1)/2) / sinh(a_B/2)
    """
    
    # ゼロ除算を避けるための微小量
    if np.isclose(a_B, 0.0):
        # a_B -> 0 の極限を別途計算する必要があります（ここでは NaN を返す）
        return np.nan 

    # --- 計算の効率化と可読性のために、共通項を定義 ---
    
    # 各sinh/coshの引数
    arg_s = (a_B * s) / 2.0
    arg_s_plus_1 = (a_B * (s + 1)) / 2.0
    arg_denom = a_B / 2.0
    arg_T1_num = (a_B * (2 * s + 1)) / 2.0
    
    # 各sinh/coshの値
    sinh_s = np.sinh(arg_s)
    cosh_s = np.cosh(arg_s)
    
    sinh_s_plus_1 = np.sinh(arg_s_plus_1)
    cosh_s_plus_1 = np.cosh(arg_s_plus_1)
    
    sinh_denom = np.sinh(arg_denom)
    cosh_denom = np.cosh(arg_denom)
    
    sinh_T1_num = np.sinh(arg_T1_num)
    
    # --- 各項の計算 ---
    
    # T1
    T1_numerator = s * (s + 1) * sinh_T1_num
    T1_denominator = 2.0 * sinh_denom
    T1 = T1_numerator / T1_denominator
    
    # T2
    T2_bracket = (s * cosh_s * cosh_s_plus_1) + \
                   ((s + 1) * sinh_s * sinh_s_plus_1)
    
    T2_numerator = -cosh_denom * T2_bracket
    T2_denominator = 2.0 * (sinh_denom ** 2)
    T2 = T2_numerator / T2_denominator
    
    # T3
    T3_numerator = (cosh_denom ** 2) * sinh_s * cosh_s_plus_1
    T3_denominator = 2.0 * (sinh_denom ** 3)
    T3 = T3_numerator / T3_denominator
    
    # --- 合計 ---
    d2A_daB2 = T1 + T2 + T3
    
    return d2A_daB2

# -----------------------------------------------------------------
# 1. 感受率の計算 
# -----------------------------------------------------------------
def calculate_chi_M(omega, T, B, g, gamma):
    """
    磁気感受率 chi_M(omega) を計算する関数
    
    引数:
    omega (float): 角周波数 (rad/s)
    T (float): 温度 (K) 
    B (float): 外部磁場 (T) 
    g (float): g因子
    gamma (float): 緩和定数 (rad/s)
    s (float): スピンサイズ (例: 7/2)

    戻り値:
    complex: 複素磁気感受率 chi_M
    """
    
    
    omega_0 = g * mu_B * B / hbar  # Zeeman固有周波数 (rad/s)
    a_B = g * mu_B * B / k_B / T  # 無次元係数

    Z = sum(np.exp(a_B * m) for m in np.arange(-s, s + 1))  # 分配関数
    A = np.sinh(a_B * s / 2) * np.cosh(a_B * (s + 1) / 2) / np.sinh(a_B / 2)
    d2A_dA2 = calculate_d2A_daB2_analytical(a_B, s)
    A_coeff = 2 * ((1 + s**2) * A - d2A_dA2)
    G0 = mu_0 * N_spin * (g * mu_B)**2 / (2 * hbar) # 結合定数G0

    numerator = (1 - np.exp(a_B)) * (1 + s**2 * A_coeff) * G0
    denominator = Z * (omega_0 - (omega + 1j * gamma))

    chi_M = -1 * (numerator / denominator)

    return chi_M

def calculate_chi_E(omega):
    """
    電気感受率 chi_E(omega) を計算する関数。
    磁気感受率にのみ依存するモデルのため、ここでは0を返す。
    もし必要ならば、周波数依存の関数を実装してください。
    """
    return 0 + 0j # (定数または周波数依存の関数を実装)

# -----------------------------------------------------------------
# 2. インピーダンスと位相の計算 (式 )
# -----------------------------------------------------------------

def calculate_relative_impedance(chi_M, chi_E):
    """
    相対インピーダンス Z_r を計算 
    Z_r = sqrt( (1 + chi_M) / (1 + chi_E) )
    """
    return np.sqrt((1 + chi_M) / (eps_bg * (1 + chi_E)))

def calculate_wave_number(omega, chi_M, chi_E):
    """
    媒体中の波数 k を実部虚部ごとに計算 
    k = (omega / c) * np.sqrt((1 + chi_M) * eps_bg * (1 + chi_E))
    """
    k = (omega / c) * np.sqrt((1 + chi_M) * eps_bg * (1 + chi_E))
    k_real = np.real(k)
    k_imag = np.imag(k)
    return k_real, k_imag  # 戻り値は実部[0]と虚部[1]のタプル

def calculate_phase(omega, chi_M, chi_E):
    """
    位相 phi を計算 
    phi = k * d + Arg{...}
    """
    
    # 媒体中の波数 k
    k = (omega / c) * np.sqrt((1 + chi_M) * eps_bg * (1 + chi_E))
    k_real = np.real(k)
    # k_imag = np.imag(k)
    
    # 伝搬項
    term1 = k_real * d

    # 界面での反射による位相項
    sqrt_mu_r = np.sqrt(1 + chi_M)
    sqrt_eps_r = eps_bg * np.sqrt(1 + chi_E)
    
    # 反射係数 r に対応する項 [cite: 18]
    r_term = (sqrt_eps_r - sqrt_mu_r) / (sqrt_mu_r + sqrt_eps_r)
    
    term2 = np.angle(r_term)
    
    return term1 + term2

# -----------------------------------------------------------------
# 3. 透過率 T(omega) の計算 (式 )
# -----------------------------------------------------------------

def calculate_T_spec(Z_r, phi, k):
    """
    透過率 T(omega) を計算 
    T = 2*|Z_r|^2 / [ (1+|Z_r|^2)^2 * {1 - e^(-2k"L)cos^(2(k'L+phi))} + 4(Re{Z_r})^2 * {1 + e^(-2k"L)cos^(2(k'L+phi))} ]
    """
    
    Z_r_abs_sq = np.abs(Z_r)**2
    Re_Z_r = np.real(Z_r)
    k_imag = np.abs(k[1])
    phi_imag = np.exp(-2 * k_imag * d)
    cos = np.cos(2 * phi)

    
    numerator = 2 * Z_r_abs_sq
    
    term_1 = ((1 + Z_r_abs_sq)**2) * (1 - phi_imag * cos)
    term_2 = 4 * (Re_Z_r**2) * (1 + phi_imag * cos)
    
    denominator = term_1 + term_2
    
    # 0除算を避ける
    if denominator == 0:
        print("発散あり\n")
        return np.nan
        
    return numerator / denominator

# -----------------------------------------------------------------
# 4. メイン実行部
# -----------------------------------------------------------------

def get_T_spec_spectrum(omega_rads, T, B, g, gamma):
    """
    指定されたパラメータで透過スペクトルを計算する
    """
    
    T_spectrum = []
    
    # numpy 配列を複素数型で初期化
    chi_M_array = np.zeros_like(omega_rads, dtype=complex)
    chi_E_array = np.zeros_like(omega_rads, dtype=complex)
    Z_r_array = np.zeros_like(omega_rads, dtype=complex)
    phi_array = np.zeros_like(omega_rads, dtype=complex)
    T_array = np.zeros_like(omega_rads, dtype=float)
    
    for i, omega in enumerate(omega_rads):
        # 1. 感受率の計算 
        chi_M = calculate_chi_M(omega, T, B, g, gamma)
        chi_E = calculate_chi_E(omega)
        
        # 2. インピーダンスと位相の計算
        Z_r = calculate_relative_impedance(chi_M, chi_E)
        phi = calculate_phase(omega, chi_M, chi_E)
        k = calculate_wave_number(omega, chi_E, chi_M)
        
        # 3. 透過率の計算
        T_val = calculate_T_spec(Z_r, phi, k)
        
        T_array[i] = T_val

    return T_array

# --- シミュレーションの実行例 ---
if __name__ == "__main__":
    
    # パラメータ設定
    T = 1.0e-5                             # 温度 (K)
    B = 9.0                             # 磁場 (T)
    g = 1.95                            # g因子
    
    # 緩和定数の設定（線幅を制御）
    # 実験的な線幅から推定: FWHM ~ 0.05 THz → gamma ~ 2π × 0.015 THz
    gamma = 2 * np.pi * 0.015 * 1e12    # 緩和定数 (rad/s)
    
    # Zeeman周波数の確認
    omega_zeeman = g * mu_B * B / hbar
    freq_zeeman = omega_zeeman / (2 * np.pi * 1e12)
    print(f"Zeeman周波数: {freq_zeeman:.3f} THz")
    print(f"線幅 (gamma): {gamma / (2 * np.pi * 1e12):.4f} THz")
    
    # 計算する周波数範囲 (THz -> rad/s)
    freq_thz = np.linspace(0, 1.0, 401)
    omega_rads = freq_thz * 1e12 * (2 * np.pi)

    # スペクトル計算
    T_spec = get_T_spec_spectrum(omega_rads, T, B, g, gamma)
    
    # NaN値を除去
    T_spec = np.nan_to_num(T_spec, nan=1.0)

    # スペクトルの正規化
    min_trans, max_trans = np.min(T_spec), np.max(T_spec)
    if max_trans > min_trans and np.isfinite(max_trans) and np.isfinite(min_trans):
        T_spec = (T_spec - min_trans) / (max_trans - min_trans)
    else:
        T_spec = np.full_like(T_spec, 0.5)
    
    R_spec = 1 - T_spec
    
    # 結果のプロット
    plt.figure(figsize=(10, 6))
    plt.plot(freq_thz, T_spec)
    plt.xlabel("Frequency (THz)")
    plt.ylabel(r"Transmission T($\omega$)")
    plt.title(f"Normalised Transmission Spectrum (T={T} K, B={B} T)")
    plt.grid(True)
    plt.ylim(0, 1.1) # 透過率は 0 から 1 の範囲
    plt.savefig("NR_spectrum.png", dpi=200)
