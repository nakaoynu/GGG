import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.signal import find_peaks
from scipy.optimize import minimize
import time
import configparser
import os

class GGGFittingConfig:
    """設定ファイルを読み込んで管理するクラス"""
    
    def __init__(self, config_file="fitting_config.ini"):
        self.config = configparser.ConfigParser()
        
        if os.path.exists(config_file):
            self.config.read(config_file, encoding='utf-8')
            print(f"✅ 設定ファイル '{config_file}' を読み込みました")
        else:
            print(f"⚠️ 設定ファイル '{config_file}' が見つかりません。デフォルト設定を使用します。")
            self._create_default_config()
        
        self._parse_config()
    
    def _create_default_config(self):
        """デフォルト設定を作成"""
        self.optimization_mode = {
            'limited_params': False,
            'limited_frequency': True,
            'single_model': False,
            'target_model': 'B_form'
        }
        
        self.param_settings = {
            # 基本パラメータ（全モードで使用）
            'd_ini': 0.1578e-3,
            'eps_bg_ini': 14.4,
            'gamma_ini': 0.11e12,
            'a_param_ini': 1.8,
            
            # 限定モード用の範囲
            'gamma_min': 1e9,
            'gamma_max': 5e12,
            'a_param_min': 1.0e-6,
            'a_param_max': 1e3,
            
            # フルモード用の範囲
            'd_min': 0.10e-3,
            'd_max': 1.0e-3,
            'eps_bg_min': 11.0,
            'eps_bg_max': 15.0,
            'gamma_full_min': 1e9,
            'gamma_full_max': 5e12,
            'a_param_full_min': 1e-6,
            'a_param_full_max': 1e3
        }
        
        self.freq_settings = {
            'freq_min': 0.15,
            'freq_max': 0.38,
            'freq_points': 300
        }
        
        self.exp_settings = {
            'temperature': 35.0,
            'magnetic_field': 7.7,
            'excel_file': 'Circular_Polarization_B_Field.xlsx',
            'sheet_name': 'Sheet2'
        }
    
    def _parse_config(self):
        """設定ファイルを解析"""
        try:
            # Optimization settings
            opt_section = self.config['optimization_settings']
            self.optimization_mode = {
                'limited_params': opt_section.getboolean('limited_params', True),
                'limited_frequency': opt_section.getboolean('limited_frequency', True),
                'single_model': opt_section.getboolean('single_model', False),
                'target_model': opt_section.get('target_model', 'B_form')
            }
            
            # Parameter settings
            param_section = self.config['parameter_settings']
            self.param_settings = {
                # 基本パラメータ（全モードで使用）
                'd_ini': float(param_section.get('d_ini', '0.1578e-3')),
                'eps_bg_ini': param_section.getfloat('eps_bg_ini', 14.4),
                'gamma_ini': float(param_section.get('gamma_ini', '0.11e12')),
                'a_param_ini': param_section.getfloat('a_param_ini', 1.8),
                
                # 限定モード用の範囲
                'gamma_min': float(param_section.get('gamma_min', '1e9')),
                'gamma_max': float(param_section.get('gamma_max', '5e12')),
                'a_param_min': param_section.getfloat('a_param_min', 1.0e-6),
                'a_param_max': param_section.getfloat('a_param_max', 1e3),
                
                # フルモード用の範囲
                'd_min': float(param_section.get('d_min', '0.10e-3')),
                'd_max': float(param_section.get('d_max', '1.0e-3')),
                'eps_bg_min': param_section.getfloat('eps_bg_min', 11.0),
                'eps_bg_max': param_section.getfloat('eps_bg_max', 15.0),
                'gamma_full_min': float(param_section.get('gamma_full_min', '1e9')),
                'gamma_full_max': float(param_section.get('gamma_full_max', '5e12')),
                'a_param_full_min': param_section.getfloat('a_param_full_min', 1e-6),
                'a_param_full_max': param_section.getfloat('a_param_full_max', 1e3)
            }
            
            # Frequency settings
            freq_section = self.config['frequency_settings']
            self.freq_settings = {
                'freq_min': freq_section.getfloat('freq_min', 0.15),
                'freq_max': freq_section.getfloat('freq_max', 0.38),
                'freq_points': freq_section.getint('freq_points', 300)
            }
            
            # Experimental settings
            exp_section = self.config['experimental_settings']
            self.exp_settings = {
                'temperature': exp_section.getfloat('temperature', 35.0),
                'magnetic_field': exp_section.getfloat('magnetic_field', 7.7),
                'excel_file': exp_section.get('excel_file', 'Circular_Polarization_B_Field.xlsx'),
                'sheet_name': exp_section.get('sheet_name', 'Sheet2')
            }
            
        except (KeyError, ValueError) as e:
            print(f"⚠️ 設定ファイルの読み込み中にエラー: {e}")
            print("デフォルト設定を使用します。")
            self._create_default_config()
    
    def print_settings(self):
        """現在の設定を表示"""
        print("\n📋 現在の最適化設定:")
        print("--- 最適化モード ---")
        for key, value in self.optimization_mode.items():
            print(f"   {key}: {value}")
        
        print("\n--- パラメータ設定 ---")
        if self.optimization_mode['limited_params']:
            print(f"   固定パラメータ: d={self.param_settings['d_ini']:.2e}, eps_bg={self.param_settings['eps_bg_ini']}")
            print(f"   gamma範囲: {self.param_settings['gamma_min']:.1e} - {self.param_settings['gamma_max']:.1e}")
            print(f"   a_param範囲: {self.param_settings['a_param_min']} - {self.param_settings['a_param_max']}")
        else:
            print(f"   初期値: d={self.param_settings['d_ini']:.2e}, eps_bg={self.param_settings['eps_bg_ini']}")
            print(f"   d範囲: {self.param_settings['d_min']:.2e} - {self.param_settings['d_max']:.2e}")
            print(f"   eps_bg範囲: {self.param_settings['eps_bg_min']} - {self.param_settings['eps_bg_max']}")
            print(f"   gamma範囲: {self.param_settings['gamma_full_min']:.1e} - {self.param_settings['gamma_full_max']:.1e}")
            print(f"   a_param範囲: {self.param_settings['a_param_full_min']} - {self.param_settings['a_param_full_max']}")
        
        print("\n--- 周波数設定 ---")
        if self.optimization_mode['limited_frequency']:
            print(f"   周波数範囲: {self.freq_settings['freq_min']} - {self.freq_settings['freq_max']} THz")
            print(f"   ポイント数: {self.freq_settings['freq_points']}")
        
        print("\n--- 実験条件 ---")
        print(f"   温度: {self.exp_settings['temperature']} K")
        print(f"   磁場: {self.exp_settings['magnetic_field']} T")

# --- グローバル物理定数と定数行列 ---
kB = 1.380649e-23; muB = 9.274010e-24; hbar = 1.054571e-34; c = 299792458
mu0 = 4.0 * np.pi * 1e-7
s = 3.5
g_factor = 1.95
N_spin = 24 / 1.238 * 1e27
G0 = mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)

# 結晶場定数
B4_0 = 0.8 / 240 ; B6_0 = 0.04 / 5040
B4_param = 0.606 ; B6_param = -1.513 
B4 = B4_0 * B4_param; B6 = B6_0 * B6_param
O04 = 60 * np.diag([7,-13,-3,9,9,-3,-13,7])
X_O44 = np.zeros((8,8)); X_O44[3,7],X_O44[4,0]=np.sqrt(35),np.sqrt(35); X_O44[2,6],X_O44[5,1]=5*np.sqrt(3),5*np.sqrt(3); O44=12*(X_O44+X_O44.T)
O06 = 1260 * np.diag([1,-5,9,-5,-5,9,-5,1]); X_O46 = np.zeros((8,8)); X_O46[3,7],X_O46[4,0]=3*np.sqrt(35),3*np.sqrt(35); X_O46[2,6],X_O46[5,1]=-7*np.sqrt(3),-7*np.sqrt(3); O46=60*(X_O46+X_O46.T)

#Sz演算子の定義
m_values = np.arange(s, -s - 1, -1)
Sz = np.diag(m_values)

class PhysicsModel:
    """物理パラメータを保持し、透過スペクトルを計算するクラス"""
    
    def __init__(self, d, eps_bg, gamma, a_param):
        self.d = d
        self.eps_bg = eps_bg
        self.gamma = gamma
        self.a_param = a_param
                
    def get_hamiltonian(self, B_ext_z):
        H_cf = (B4 * kB) * (O04 + 5 * O44) + (B6 * kB) * (O06 - 21 * O46)
        H_zee = g_factor * muB * B_ext_z * Sz
        return H_cf + H_zee
        
    def calculate_susceptibility(self, omega_array, H, T):
        eigenvalues, _ = np.linalg.eigh(H)
        eigenvalues -= np.min(eigenvalues)
        Z = np.sum(np.exp(-eigenvalues / (kB * T)))
        populations = np.exp(-eigenvalues / (kB * T)) / Z
        delta_E = eigenvalues[1:] - eigenvalues[:-1]
        delta_pop = populations[1:] - populations[:-1]
        omega_0 = delta_E / hbar
        m_vals_trans = np.arange(s, -s, -1)
        transition_strength = (s + m_vals_trans) * (s - m_vals_trans + 1)
        numerator = G0 * delta_pop * transition_strength
        denominator = (omega_0[:, np.newaxis] - omega_array) - (1j * self.gamma)
        chi_array = np.sum(numerator[:, np.newaxis] / denominator, axis=0)
        return -self.a_param * chi_array
        
    def calculate_transmission_intensity(self, omega_array, mu_r_array):
        n_complex = np.sqrt(self.eps_bg * mu_r_array + 0j)
        impe = np.sqrt(mu_r_array / self.eps_bg + 0j)
        lambda_0 = np.full_like(omega_array, np.inf, dtype=float)
        nonzero_mask = omega_array != 0
        lambda_0[nonzero_mask] = (2 * np.pi * c) / omega_array[nonzero_mask]
        delta = 2 * np.pi * n_complex * self.d / lambda_0
        r = (impe - 1) / (impe + 1)
        numerator = 4 * impe * np.exp(1j * delta) 
        denominator = (1 + impe)**2 - (impe - 1)**2 * np.exp(2j * delta)
        t = np.divide(numerator, denominator, where=np.abs(denominator)>1e-12, out=np.ones_like(denominator, dtype=complex))
        return t
        
    def get_spectrum(self, omega_array, T, B, model_type):
        """指定された条件でスペクトルを計算するメインメソッド"""
        H_B = self.get_hamiltonian(B)
        chi_B = self.calculate_susceptibility(omega_array, H_B, T)
        
        if model_type == 'H_form':
            mu_r_B = 1 + chi_B
        elif model_type == 'B_form':
            mu_r_B = np.divide(1, 1 - chi_B, where=(1 - chi_B)!=0, out=np.full_like(chi_B, np.inf, dtype=complex))
        else:
            raise ValueError("Unknown model_type")
            
        T_B = self.calculate_transmission_intensity(omega_array, mu_r_B)
        return np.abs(T_B)**2

class GGGOptimizer:
    """GGGフィッティングの最適化を管理するクラス"""
    
    def __init__(self, config_file="fitting_config.ini"):
        self.config = GGGFittingConfig(config_file)
        self.exp_data = None
        self.freq_mask = None
        
    def load_experimental_data(self):
        """実験データを読み込み"""
        print("\n📂 実験データを読み込みます...")
        
        file_path = self.config.exp_settings['excel_file']
        sheet_name = self.config.exp_settings['sheet_name']
        
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
            exp_freq_thz = df['Frequency (THz)'].to_numpy(dtype=float)
            exp_transmittance = df['Transmittance (7.7T)'].to_numpy(dtype=float)
            
            # データの正規化
            min_exp, max_exp = np.min(exp_transmittance), np.max(exp_transmittance)
            exp_transmittance_normalized = (exp_transmittance - min_exp) / (max_exp - min_exp)
            
            self.exp_data = {
                'freq_thz': exp_freq_thz,
                'transmittance': exp_transmittance_normalized
            }
            
            print(f"✅ データの読み込み完了: {len(exp_freq_thz)} データポイント")
            return True
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            return False
    
    def setup_optimization_parameters(self):
        """最適化パラメータを設定"""
        if self.config.optimization_mode['limited_params']:
            # 限定モード
            param_keys = ['gamma', 'a_param']
            p_initial = [self.config.param_settings['gamma_ini'], self.config.param_settings['a_param_ini']]
            bounds = [
                (self.config.param_settings['gamma_min'], self.config.param_settings['gamma_max']),
                (self.config.param_settings['a_param_min'], self.config.param_settings['a_param_max'])
            ]
            
            def create_model(p_array):
                p_dict = dict(zip(param_keys, p_array))
                p_dict['d'] = self.config.param_settings['d_ini']
                p_dict['eps_bg'] = self.config.param_settings['eps_bg_ini']
                return PhysicsModel(**p_dict)
                
            print("🎯 限定最適化モード: gamma と a_param のみを最適化")
            
        else:
            # フルモード  
            param_keys = ['d', 'eps_bg', 'gamma', 'a_param']
            p_initial = [self.config.param_settings['d_ini'], 
                        self.config.param_settings['eps_bg_ini'], 
                        self.config.param_settings['gamma_ini'],
                        self.config.param_settings['a_param_ini']]
            bounds = [(self.config.param_settings['d_min'], self.config.param_settings['d_max']),
                      (self.config.param_settings['eps_bg_min'], self.config.param_settings['eps_bg_max']),
                      (self.config.param_settings['gamma_full_min'], self.config.param_settings['gamma_full_max']),
                      (self.config.param_settings['a_param_full_min'], self.config.param_settings['a_param_full_max'])]
            
            def create_model(p_array):
                p_dict = dict(zip(param_keys, p_array))
                return PhysicsModel(**p_dict)
                
            print("🔄 フル最適化モード: 全パラメータを最適化")
        
        return param_keys, p_initial, bounds, create_model
    
    def setup_frequency_range(self):
        """周波数範囲を設定"""
        if self.config.optimization_mode['limited_frequency']:
            # 限定周波数範囲
            freq_min = self.config.freq_settings['freq_min']
            freq_max = self.config.freq_settings['freq_max']
            freq_points = self.config.freq_settings['freq_points']
            
            omega_hz = np.linspace(freq_min*1e12, freq_max*1e12, freq_points)
            omega_rad_s = omega_hz * 2 * np.pi
            freq_thz = omega_hz / 1e12
            
            # 実験データフィルタリング
            mask = ((self.exp_data['freq_thz'] >= freq_min) & 
                   (self.exp_data['freq_thz'] <= freq_max))
            
            print(f"🎯 限定周波数範囲: {freq_min}-{freq_max} THz")
            
        else:
            # フル周波数範囲
            omega_hz = np.linspace(0.1*1e12, np.max(self.exp_data['freq_thz'])*1e12, 500)
            omega_rad_s = omega_hz * 2 * np.pi
            freq_thz = omega_hz / 1e12
            mask = np.ones(len(self.exp_data['freq_thz']), dtype=bool)
            
            print("🔄 フル周波数範囲を使用")
        
        self.freq_mask = mask
        return omega_rad_s, freq_thz
    
    def setup_model_list(self):
        """最適化対象モデルを設定"""
        if self.config.optimization_mode['single_model']:
            model_list = [self.config.optimization_mode['target_model']]
            print(f"🎯 単一モデル最適化: {self.config.optimization_mode['target_model']}")
        else:
            model_list = ['H_form', 'B_form']
            print("🔄 両モデル（H形式・B形式）を最適化")
        
        return model_list
    
    def run_optimization(self):
        """最適化を実行"""
        if not self.load_experimental_data():
            return None
        
        # 設定を表示
        self.config.print_settings()
        
        # パラメータと範囲を設定
        param_keys, p_initial, bounds, create_model = self.setup_optimization_parameters()
        omega_rad_s, freq_thz = self.setup_frequency_range()
        model_list = self.setup_model_list()
        
        # フィルタリングされた実験データ
        exp_freq_filtered = self.exp_data['freq_thz'][self.freq_mask]
        exp_data_filtered = self.exp_data['transmittance'][self.freq_mask]
        
        T_fixed = self.config.exp_settings['temperature']
        B_field = self.config.exp_settings['magnetic_field']
        
        # コスト関数定義
        def cost_function(p_array, model_type):
            model = create_model(p_array)
            spectrum = model.get_spectrum(omega_rad_s, T_fixed, B_field, model_type)
            
            min_th, max_th = np.min(spectrum), np.max(spectrum)
            spectrum_normalized = ((spectrum - min_th) / (max_th - min_th) 
                                 if (max_th - min_th) > 1e-9 else np.zeros_like(spectrum))
            
            theory_interp = np.interp(exp_freq_filtered, freq_thz, spectrum_normalized)
            residuals = exp_data_filtered - theory_interp
            return np.sum(residuals**2)
        
        # 最適化実行
        print(f"\n⚙️ 自動最適化を開始...")
        print(f"   対象パラメータ: {param_keys}")
        print(f"   対象モデル: {model_list}")
        
        results_dict = {}
        for model_name in model_list:
            print(f"\n--- [{model_name}]モデルの最適化中 ---")
            start_time = time.time()
            
            result = minimize(cost_function, p_initial, args=(model_name,), 
                            method='L-BFGS-B', bounds=bounds)
            
            end_time = time.time()
            results_dict[model_name] = result
            
            if result.success:
                print(f"✅ 最適化成功！ 実行時間: {end_time - start_time:.2f}秒")
                print(f"   最小二乗誤差: {result.fun:.6f}")
            else:
                print(f"❌ 最適化失敗: {result.message}")
        
        # 結果を保存
        self.results = {
            'results_dict': results_dict,
            'param_keys': param_keys,
            'create_model': create_model,
            'omega_rad_s': omega_rad_s,
            'freq_thz': freq_thz,
            'exp_freq_filtered': exp_freq_filtered,
            'exp_data_filtered': exp_data_filtered,
            'T_fixed': T_fixed,
            'B_field': B_field
        }
        
        return results_dict
    
    def plot_results(self):
        """結果をプロット"""
        if not hasattr(self, 'results'):
            print("❌ 結果がありません。まず最適化を実行してください。")
            return
        
        print("\n📊 結果をプロット中...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 実験データ
        ax.plot(self.results['exp_freq_filtered'], self.results['exp_data_filtered'], 
                'o', color='black', markersize=6, label='実験データ（フィルタリング済み）')
        
        # 最適化結果
        colors = {'H_form': 'blue', 'B_form': 'darkorange'}
        
        for model_name, result in self.results['results_dict'].items():
            if result.success:
                # 最適化パラメータの詳細を表示
                print(f"\n🎉 [{model_name}] モデル最適化成功！")
                print(f"   最小二乗誤差: {result.fun:.6f}")
                
                for key, val in zip(self.results['param_keys'], result.x):
                    print(f"   {key:<10} = {val:.4e}")
                
                # 理論スペクトルを計算してプロット
                final_model = self.results['create_model'](result.x)
                spectrum = final_model.get_spectrum(self.results['omega_rad_s'], 
                                                   self.results['T_fixed'], 
                                                   self.results['B_field'], 
                                                   model_name)
                
                min_opt, max_opt = np.min(spectrum), np.max(spectrum)
                spectrum_normalized = (spectrum - min_opt) / (max_opt - min_opt)
                
                ax.plot(self.results['freq_thz'], spectrum_normalized, 
                       color=colors[model_name], linewidth=2.5, 
                       label=f'最適化後理論値 ({model_name})')
        
        # グラフ装飾
        ax.set_title('限定領域における自動最適化結果', fontsize=16)
        ax.set_xlabel('周波数 (THz)', fontsize=12)
        ax.set_ylabel('正規化透過率 $T(B)$', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 最適化範囲を強調
        if self.config.optimization_mode['limited_frequency']:
            freq_min = self.config.freq_settings['freq_min']
            freq_max = self.config.freq_settings['freq_max']
            ax.axvspan(freq_min, freq_max, alpha=0.1, color='red', 
                      label='最適化対象範囲')
            ax.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig('limited_fitting_result.png', dpi=300)
        plt.show()

# メイン実行部分
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 GGG透過スペクトル自動フィッティング（設定ファイル版）")
    print("=" * 60)
    
    # 最適化実行
    optimizer = GGGOptimizer("fitting_config.ini")
    results = optimizer.run_optimization()
    
    if results:
        optimizer.plot_results()
        print(f"\n🎯 限定領域での最適化が完了しました。")
    else:
        print("❌ 最適化に失敗しました。")
