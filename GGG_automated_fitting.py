import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.optimize import curve_fit
import configparser

# --- 0. 物理定数とパラメータ定義 ---
kB = 1.380649e-23; muB = 9.274010e-24; hbar = 1.054571e-34; c = 299792458
s = 3.5; g_factor = 1.95
mu0 = 4.0 * np.pi * 1e-7; N_spin = 24/1.238 * 1e27
G0 = mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
B4_0 = 0.8 / 240; B6_0 = 0.04 / 5040
B4_param = 0.606; B6_param = -1.513 
B4 = B4_0 * B4_param; B6 = B6_0 * B6_param
O04 = 60*np.diag([7,-13,-3,9,9,-3,-13,7]); X_O44=np.zeros((8,8)); X_O44[3,7],X_O44[4,0]=np.sqrt(35),np.sqrt(35); X_O44[2,6],X_O44[5,1]=5*np.sqrt(3),5*np.sqrt(3); O44=12*(X_O44+X_O44.T)
O06 = 1260*np.diag([1,-5,9,-5,-5,9,-5,1]); X_O46=np.zeros((8,8)); X_O46[3,7],X_O46[4,0]=3*np.sqrt(35),3*np.sqrt(35); X_O46[2,6],X_O46[5,1]=-7*np.sqrt(3),-7*np.sqrt(3); O46=60*(X_O46+X_O46.T)
m_values = np.arange(s, -s - 1, -1); Sz = np.diag(m_values)

# --- 1. 物理モデル関数 ---
def get_hamiltonian(B_ext_z):
    H_cf = (B4 * kB) * (O04 + 5 * O44) + (B6 * kB) * (O06 - 21 * O46)
    H_zee = g_factor * muB * B_ext_z * Sz
    return H_cf + H_zee

def calculate_susceptibility(omega_array, H, T, gamma):
    eigenvalues, _ = np.linalg.eigh(H); eigenvalues -= np.min(eigenvalues)
    Z = np.sum(np.exp(-eigenvalues / (kB * T))); populations = np.exp(-eigenvalues / (kB * T)) / Z
    delta_E = eigenvalues[1:]-eigenvalues[:-1]; delta_pop = populations[1:]-populations[:-1]
    omega_0 = delta_E/hbar; m_vals = np.arange(s,-s,-1); transition_strength=(s+m_vals)*(s-m_vals+1)
    numerator = G0 * delta_pop * transition_strength
    denominator = (omega_0[:, np.newaxis] - omega_array) - (1j * gamma[:, np.newaxis])
    chi_array = np.sum(numerator[:, np.newaxis]/denominator, axis=0)
    return -chi_array

def calculate_transmission_intensity(omega_array, mu_r_array, d, eps_bg):
    n_complex = np.sqrt(eps_bg * mu_r_array + 0j)
    impe = np.sqrt(mu_r_array / eps_bg + 0j)
    lambda_0 = np.full_like(omega_array, np.inf, dtype=float)
    nonzero_mask = omega_array != 0
    lambda_0[nonzero_mask] = (2*np.pi*c)/omega_array[nonzero_mask]
    delta = 2*np.pi*n_complex*d/lambda_0
    numerator = 4 * impe * np.exp(1j * delta) 
    denominator = (1 + impe)**2 - (impe - 1)**2 * np.exp(2j * delta)
    t = np.divide(numerator, denominator, where=np.abs(denominator)>1e-12, out=np.full_like(denominator, np.inf, dtype=complex)) #分母が0に近い場合は発散
    return np.abs(t)**2

# --- 2. curve_fitのためのラッパー関数 ---
def model_wrapper(omega_array, d_fit, eps_bg_fit, a, g1, g2, g3, g4, g5, g6, g7):
    # この関数は、10個の全パラメータを受け取るように固定
    global model_type_to_fit # グローバル変数から現在のモデルタイプを取得
    
    gamma_fit = np.array([g1, g2, g3, g4, g5, g6, g7])
    
    H = get_hamiltonian(B_field)
    chi = calculate_susceptibility(omega_array, H, Temp, gamma_fit)
    
    if model_type_to_fit == 'H_form':
        mu_r = 1 + a * chi
    elif model_type_to_fit == 'B_form':
        mu_r = np.divide(1, 1 - a * chi, where=(1 - a * chi)!=0, out=np.full_like(chi, np.inf, dtype=complex))
    
    return calculate_transmission_intensity(omega_array, mu_r, d_fit, eps_bg_fit)

# --- 3. メイン実行ブロック ---
if __name__ == '__main__':
    # --- 設定ファイルの読み込み ---
    config = configparser.ConfigParser()
    config.read('fitting_config.ini', encoding='utf-8')

    opt_set = config['optimization_settings']
    param_set = config['parameter_settings']
    freq_set = config['frequency_settings']
    exp_set = config['experimental_settings']
    out_set = config['output_settings']

    # --- 実験データの準備 ---
    print("実験データを読み込みます...")
    df = pd.read_excel(exp_set['excel_file'], sheet_name=exp_set['sheet_name'], header=0)
    
    # 周波数でフィルタリング
    freq_min, freq_max = freq_set.getfloat('freq_min'), freq_set.getfloat('freq_max')
    df = df[(df['Frequency (THz)'] >= freq_min) & (df['Frequency (THz)'] <= freq_max)].copy()
    print(f"周波数範囲を {freq_min} - {freq_max} THzに限定しました。")

    exp_freq_thz = df['Frequency (THz)'].to_numpy(dtype=float)
    
    # 磁場値を正しく取得
    B_field = exp_set.getfloat('magnetic_field')
    exp_transmittance = df[f"Transmittance ({B_field}T)"].to_numpy(dtype=float)
    exp_omega_rad_s = exp_freq_thz * 1e12 * 2 * np.pi
    
    Temp = exp_set.getfloat('temperature')

    # --- 最適化の実行 ---
    if opt_set.getboolean('single_model'):
        model_types_to_run = [opt_set['target_model']]
    else:
        model_types_to_run = ['H_form', 'B_form']

    best_params_all = {}
    
    for model_type in model_types_to_run:
        print(f"\n--- [{model_type}] モデルのフィッティングを開始します ---")
        model_type_to_fit = model_type # グローバル変数に設定
        
        # 最適化パラメータの初期値と範囲を定義
        gamma_initial = eval(param_set['gamma_ini'])
        p0 = [param_set.getfloat('d_ini'), param_set.getfloat('eps_bg_ini'), param_set.getfloat('a_param_ini')] + gamma_initial
        
        bounds_lower = [param_set.getfloat('d_min'), param_set.getfloat('eps_bg_min'), param_set.getfloat('a_param_min')] + [param_set.getfloat('gamma_min')] * 7
        bounds_upper = [param_set.getfloat('d_max'), param_set.getfloat('eps_bg_max'), param_set.getfloat('a_param_max')] + [param_set.getfloat('gamma_max')] * 7

        # curve_fitによる最適化
        try:
            popt, pcov = curve_fit(model_wrapper, exp_omega_rad_s, exp_transmittance, p0=p0, bounds=(bounds_lower, bounds_upper))
            best_params_all[model_type] = popt
            
            if out_set.getboolean('print_details'):
                print("\n最適化されたパラメータ:")
                print(f"  d = {popt[0]*1e6:.4f} um")
                print(f"  eps_bg = {popt[1]:.4f}")
                print(f"  a = {popt[2]:.4f}")
                print(f"  gamma = {popt[3:]}")
        except Exception as e:
            print(f"❌ フィッティング中にエラーが発生しました: {e}")
            continue

    # --- 結果の可視化 ---
    print("\nフィッティング結果をプロットします...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(exp_freq_thz, exp_transmittance, 'o', color='black', markersize=5, label='実験データ')
    
    plot_freq_thz = np.linspace(np.min(exp_freq_thz), np.max(exp_freq_thz), 500)
    plot_omega_rad_s = plot_freq_thz * 1e12 * 2 * np.pi

    colors = {'H_form': 'blue', 'B_form': 'red'}
    for model_type, popt in best_params_all.items():
        model_type_to_fit = model_type
        fit_curve = model_wrapper(plot_omega_rad_s, *popt)
        ax.plot(plot_omega_rad_s / (2 * np.pi * 1e12), fit_curve, color=colors[model_type], lw=2, label=f'ベストフィット ({model_type})')
        
    ax.set_xlabel('周波数 (THz)'); ax.set_ylabel('透過率 T(B)'); ax.legend(); ax.grid(True)
    ax.set_title(f"自動フィッティング結果 (B={B_field}T, T={Temp}K)")
    
    if out_set.getboolean('save_plot'):
        filename = out_set['output_filename_prefix']
        plt.savefig(f"{filename}.png", dpi=300)
        print(f"✅ グラフを '{filename}.png' に保存しました。")
    
    if out_set.getboolean('show_plot'):
        plt.show()
    
    plt.close()
    print("\n解析が完了しました。")
    