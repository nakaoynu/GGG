"""
実用的なPyMCベイズ推定プログラム
既存の物理計算ロジックを活用し、効率的なベイズ推定とLOO-CVモデル比較を実行
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from pytensor.compile.ops import as_op
import pytensor
from typing import List, Dict, Any
import pickle
import warnings
warnings.filterwarnings('ignore')

# --- 既存の物理計算モジュールをインポート ---
# 物理定数と定数行列
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


class PhysicsCalculator:
    """
    既存の物理計算ロジックを使用したクラス
    """
    def __init__(self):
        pass

    def get_hamiltonian(self, B_ext_z):
        H_cf = (B4 * kB) * (O04 + 5 * O44) + (B6 * kB) * (O06 - 21 * O46)
        H_zee = g_factor * muB * B_ext_z * Sz
        return H_cf + H_zee

    def calculate_susceptibility(self, omega_array, H, T, gamma, a_param):
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
        denominator = (omega_0[:, np.newaxis] - omega_array) - (1j * gamma)
        chi_array = np.sum(numerator[:, np.newaxis] / denominator, axis=0)
        return -a_param * chi_array

    def calculate_transmission_intensity(self, omega_array, mu_r_array, d, eps_bg):
        n_complex = np.sqrt(eps_bg * mu_r_array + 0j)
        impe = np.sqrt(mu_r_array / eps_bg + 0j)
        lambda_0 = np.full_like(omega_array, np.inf, dtype=float)
        nonzero_mask = omega_array != 0
        lambda_0[nonzero_mask] = (2 * np.pi * c) / omega_array[nonzero_mask]
        delta = 2 * np.pi * n_complex * d / lambda_0
        numerator = 4 * impe * np.exp(1j * delta) 
        denominator = (1 + impe)**2 - (impe - 1)**2 * np.exp(2j * delta)
        t = np.divide(numerator, denominator, where=np.abs(denominator)>1e-12, out=np.ones_like(denominator, dtype=complex))
        return t

    def get_spectrum(self, omega_array, d, eps_bg, gamma, a_param, T, B, model_type):
        """スペクトル計算のメインメソッド"""
        H_B = self.get_hamiltonian(B)
        chi_B = self.calculate_susceptibility(omega_array, H_B, T, gamma, a_param)
        
        if model_type == 'H_form':
            mu_r_B = 1 + chi_B
        elif model_type == 'B_form':
            mu_r_B = np.divide(1, 1 - chi_B, where=(1 - chi_B)!=0, out=np.full_like(chi_B, np.inf, dtype=complex))
        else:
            raise ValueError("Unknown model_type")
            
        T_B = self.calculate_transmission_intensity(omega_array, mu_r_B, d, eps_bg)
        return np.abs(T_B)**2


# PyTensorオペレータとして物理計算を登録
@as_op(itypes=[pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar], otypes=[pt.dvector])
def physics_model_op(d, eps_bg, gamma, a_param, 
                    omega_array, freq_thz, T_fixed, B_field, model_type):
    """
    物理計算をPyTensorオペレータとして実装
    """
    calculator = PhysicsCalculator()
    
    # スペクトル計算
    spectrum = calculator.get_spectrum(
        omega_array, 
        float(d), 
        float(eps_bg), 
        float(gamma), 
        float(a_param), 
        T_fixed, 
        B_field, 
        model_type
    )
    
    # 正規化
    min_val, max_val = np.min(spectrum), np.max(spectrum)
    if max_val > min_val:
        spectrum_normalized = (spectrum - min_val) / (max_val - min_val)
    else:
        spectrum_normalized = np.zeros_like(spectrum)
    
    return spectrum_normalized.astype(np.float64)


class BayesianAnalyzer:
    """
    ベイズ推定とモデル比較を実行するクラス
    """
    def __init__(self, exp_freq_thz, exp_transmittance_normalized, 
                 omega_array, freq_thz, T_fixed=35.0, B_field=7.7):
        self.exp_freq_thz = exp_freq_thz
        self.exp_transmittance_normalized = exp_transmittance_normalized
        self.omega_array = omega_array
        self.freq_thz = freq_thz
        self.T_fixed = T_fixed
        self.B_field = B_field
        self.calculator = PhysicsCalculator()
        
    def create_model(self, model_type: str):
        """
        指定されたモデルタイプでPyMCモデルを作成
        """
        with pm.Model() as model:
            # 事前分布（情報的事前分布を使用）
            d = pm.TruncatedNormal('d', mu=0.16e-3, sigma=0.02e-3, lower=0.05e-3, upper=0.25e-3)
            eps_bg = pm.TruncatedNormal('eps_bg', mu=13.0, sigma=1.0, lower=10.0, upper=16.0)
            gamma = pm.TruncatedNormal('gamma', mu=0.11e12, sigma=0.2e12, lower=0.01e12, upper=2.0e12)
            a_param = pm.TruncatedNormal('a_param', mu=1.8, sigma=0.3, lower=0.1, upper=3.0)
            
            # 観測誤差
            sigma = pm.HalfNormal('sigma', sigma=0.1)
            
            # 理論計算のためのカスタムオペレータ
            def theory_calculation(d_val, eps_bg_val, gamma_val, a_param_val):
                # NumPy計算
                spectrum = self.calculator.get_spectrum(
                    self.omega_array, d_val, eps_bg_val, gamma_val, a_param_val,
                    self.T_fixed, self.B_field, model_type
                )
                
                # 正規化
                min_val, max_val = np.min(spectrum), np.max(spectrum)
                if max_val > min_val:
                    spectrum_norm = (spectrum - min_val) / (max_val - min_val)
                else:
                    spectrum_norm = np.zeros_like(spectrum)
                
                # 実験周波数で内挿
                theory_interp = np.interp(self.exp_freq_thz, self.freq_thz, spectrum_norm)
                return theory_interp
            
            # PyTensorカスタムオペレータ
            @as_op(itypes=[pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar], 
                   otypes=[pt.dvector])
            def theory_op(d, eps_bg, gamma, a_param):
                return theory_calculation(
                    float(d), float(eps_bg), float(gamma), float(a_param)
                ).astype(np.float64)
            
            # 理論値の計算
            theory_spectrum = theory_op(
                d, eps_bg, gamma, a_param
            )
            
            # 尤度
            likelihood = pm.Normal(
                'likelihood', 
                mu=theory_spectrum, 
                sigma=sigma, 
                observed=self.exp_transmittance_normalized
            )
            
        return model
    
    def run_sampling(self, model, draws=1000, tune=1000, chains=2):
        """
        MCMCサンプリングを実行
        """
        with model:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=4,
                return_inferencedata=True,
                progressbar=True,
                target_accept=0.8
            )
        return trace
    
    def compare_models(self, traces: Dict[str, Any]):
        """
        LOO-CVを使ってモデルを比較
        """
        loo_results = {}
        
        for model_type, trace in traces.items():
            try:
                loo = az.loo(trace)
                loo_results[model_type] = loo
                print(f"{model_type}モデル - LOO: {loo.loo:.2f} ± {loo.loo_se:.2f}")
            except Exception as e:
                print(f"{model_type}モデルのLOO計算でエラー: {e}")
                continue
        
        if len(loo_results) > 1:
            comparison = az.compare(loo_results)
            return comparison, loo_results
        else:
            return None, loo_results
    
    def plot_results(self, traces: Dict[str, Any], comparison=None):
        """
        結果の可視化
        """
        # 1. 事後分布の比較
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        param_names = ['d', 'eps_bg', 'gamma', 'a_param']
        
        for i, model_type in enumerate(['H_form', 'B_form']):
            if model_type not in traces:
                continue
                
            for j, param in enumerate(param_names):
                try:
                    az.plot_posterior(traces[model_type], var_names=[param], ax=axes[i, j])
                    axes[i, j].set_title(f'{model_type} - {param}')
                except:
                    axes[i, j].text(0.5, 0.5, 'No data', ha='center', va='center')
        
        plt.suptitle('パラメータの事後分布比較', fontsize=16)
        plt.tight_layout()
        plt.savefig('bayesian_parameter_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. トレースプロット
        for model_type, trace in traces.items():
            try:
                az.plot_trace(trace, var_names=param_names, compact=True)
                plt.suptitle(f'{model_type}モデル - MCMCトレース', fontsize=16)
                plt.tight_layout()
                plt.savefig(f'bayesian_trace_{model_type}.png', dpi=300, bbox_inches='tight')
                plt.show()
            except Exception as e:
                print(f"{model_type}のトレースプロット作成エラー: {e}")
        
        # 3. モデル比較
        if comparison is not None:
            try:
                az.plot_compare(comparison)
                plt.title('LOO-CVによるモデル比較')
                plt.tight_layout()
                plt.savefig('bayesian_model_comparison.png', dpi=300, bbox_inches='tight')
                plt.show()
            except Exception as e:
                print(f"モデル比較プロット作成エラー: {e}")
        
        # 4. 予測vs実験データ
        self.plot_predictions(traces)
    
    def plot_predictions(self, traces: Dict[str, Any]):
        """
        事後予測分布と実験データの比較
        """
        fig, axes = plt.subplots(1, len(traces), figsize=(6*len(traces), 6))
        if len(traces) == 1:
            axes = [axes]
        
        for i, (model_type, trace) in enumerate(traces.items()):
            try:
                # 事後サンプルから予測
                posterior = trace.posterior
                n_samples = min(100, len(posterior.chain) * len(posterior.draw))
                
                predictions = []
                param_samples = {
                    'd': posterior['d'].values.flatten()[:n_samples],
                    'eps_bg': posterior['eps_bg'].values.flatten()[:n_samples],
                    'gamma': posterior['gamma'].values.flatten()[:n_samples],
                    'a_param': posterior['a_param'].values.flatten()[:n_samples]
                }
                
                for j in range(n_samples):
                    # パラメータ制約
                    d_val = np.clip(param_samples['d'][j], 0.05e-3, 0.25e-3)
                    eps_bg_val = np.clip(param_samples['eps_bg'][j], 10.0, 16.0)
                    gamma_val = np.clip(param_samples['gamma'][j], 0.01e12, 2.0e12)
                    a_param_val = np.clip(param_samples['a_param'][j], 0.1, 3.0)
                    
                    # 予測計算
                    pred_spectrum = self.calculator.get_spectrum(
                        self.omega_array, d_val, eps_bg_val, gamma_val, a_param_val,
                        self.T_fixed, self.B_field, model_type
                    )
                    
                    # 正規化
                    min_pred, max_pred = np.min(pred_spectrum), np.max(pred_spectrum)
                    if max_pred > min_pred:
                        pred_spectrum_norm = (pred_spectrum - min_pred) / (max_pred - min_pred)
                    else:
                        pred_spectrum_norm = np.zeros_like(pred_spectrum)
                    
                    # 実験周波数で内挿
                    pred_interp = np.interp(self.exp_freq_thz, self.freq_thz, pred_spectrum_norm)
                    predictions.append(pred_interp)
                
                predictions = np.array(predictions)
                
                # 統計量の計算
                pred_mean = np.mean(predictions, axis=0)
                pred_std = np.std(predictions, axis=0)
                pred_lower = np.percentile(predictions, 2.5, axis=0)
                pred_upper = np.percentile(predictions, 97.5, axis=0)
                
                # プロット
                axes[i].plot(self.exp_freq_thz, self.exp_transmittance_normalized, 'o', 
                           color='black', markersize=4, label='実験データ')
                axes[i].plot(self.exp_freq_thz, pred_mean, '-', 
                           color='red', linewidth=2, label='事後予測平均')
                axes[i].fill_between(self.exp_freq_thz, pred_lower, pred_upper, 
                                   alpha=0.3, color='red', label='95%信頼区間')
                
                axes[i].set_title(f'{model_type}モデルの事後予測')
                axes[i].set_xlabel('周波数 (THz)')
                axes[i].set_ylabel('正規化透過率')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                
                # RMSE計算
                rmse = np.sqrt(np.mean((pred_mean - self.exp_transmittance_normalized)**2))
                axes[i].text(0.05, 0.95, f'RMSE: {rmse:.4f}', 
                           transform=axes[i].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
            except Exception as e:
                print(f"{model_type}の予測プロット作成エラー: {e}")
                axes[i].text(0.5, 0.5, f'エラー: {str(e)}', ha='center', va='center',
                           transform=axes[i].transAxes)
        
        plt.tight_layout()
        plt.savefig('bayesian_posterior_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, traces: Dict[str, Any], comparison=None):
        """
        結果の保存
        """
        # トレースデータの保存
        for model_type, trace in traces.items():
            trace.to_netcdf(f'bayesian_trace_{model_type}.nc')
            print(f"{model_type}モデルのトレースを保存: bayesian_trace_{model_type}.nc")
        
        # サマリーの保存
        with open('bayesian_analysis_summary.txt', 'w', encoding='utf-8') as f:
            f.write("=== PyMCベイズ推定結果サマリー ===\n\n")
            
            for model_type, trace in traces.items():
                f.write(f"--- {model_type}モデル ---\n")
                try:
                    summary = az.summary(trace, var_names=['d', 'eps_bg', 'gamma', 'a_param'])
                    f.write(str(summary))
                    f.write("\n\n")
                except Exception as e:
                    f.write(f"サマリー作成エラー: {e}\n\n")
            
            if comparison is not None:
                f.write("--- モデル比較結果 (LOO-CV) ---\n")
                f.write(str(comparison))
                f.write("\n")
        
        print("分析結果を保存: bayesian_analysis_summary.txt")


def main():
    """
    メイン実行関数
    """
    print("=== 実用的PyMCベイズ推定とLOO-CVモデル比較 ===\n")
    
    # --- データ読み込み ---
    print("実験データを読み込みます...")
    try:
        file_path = "Circular_Polarization_B_Field.xlsx"
        sheet_name = 'Sheet2'
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
        exp_freq_thz = df['Frequency (THz)'].to_numpy(dtype=float)
        exp_transmittance_7_7 = df['Transmittance (7.7T)'].to_numpy(dtype=float)
        
        # 正規化
        min_exp, max_exp = np.min(exp_transmittance_7_7), np.max(exp_transmittance_7_7)
        exp_transmittance_normalized = (exp_transmittance_7_7 - min_exp) / (max_exp - min_exp)
        
        print(f"データ読み込み成功。データ点数: {len(exp_freq_thz)}")
        
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return
    
    # 計算用周波数グリッド
    omega_hz = np.linspace(0.1*1e12, np.max(exp_freq_thz)*1e12, 500)
    omega_rad_s = omega_hz * 2 * np.pi
    freq_thz = omega_hz / 1e12
    
    # アナライザー初期化
    analyzer = BayesianAnalyzer(
        exp_freq_thz, exp_transmittance_normalized,
        omega_rad_s, freq_thz
    )
    
    # --- ベイズ推定実行 ---
    traces = {}
    models = {}
    
    for model_type in ['H_form', 'B_form']:
        print(f"\n--- {model_type}モデルの解析開始 ---")
        
        try:
            # モデル作成
            model = analyzer.create_model(model_type)
            models[model_type] = model
            
            # サンプリング実行
            print(f"MCMCサンプリング開始...")
            trace = analyzer.run_sampling(model, draws=800, tune=800, chains=2)
            traces[model_type] = trace
            
            print(f"{model_type}モデルの推定完了")
            
        except Exception as e:
            print(f"{model_type}モデルでエラー: {e}")
            continue
    
    if not traces:
        print("すべてのモデルでエラーが発生しました。")
        return
    
    # --- 結果分析 ---
    print(f"\n=== 推定結果の分析 ===")
    
    # パラメータサマリー
    for model_type, trace in traces.items():
        print(f"\n--- {model_type}モデルの事後統計 ---")
        try:
            summary = az.summary(trace, var_names=['d', 'eps_bg', 'gamma', 'a_param'])
            print(summary)
        except Exception as e:
            print(f"サマリー作成エラー: {e}")
    
    # --- モデル比較 ---
    print(f"\n=== LOO-CVによるモデル比較 ===")
    comparison, loo_results = analyzer.compare_models(traces)
    
    if comparison is not None:
        print("\nモデル比較結果:")
        print(comparison)
        
        best_model = comparison.index[0]
        print(f"\n🏆 最良モデル: {best_model}")
        if 'dloo' in comparison.columns:
            print(f"   LOO差分: {comparison.loc[best_model, 'dloo']:.2f}")
    else:
        print("モデル比較を実行できませんでした")
    
    # --- 可視化 ---
    print(f"\n=== 結果の可視化 ===")
    analyzer.plot_results(traces, comparison)
    
    # --- 結果保存 ---
    print(f"\n=== 結果の保存 ===")
    analyzer.save_results(traces, comparison)
    
    print(f"\n🎉 分析が完了しました！")
    print("生成されたファイル:")
    print("- bayesian_parameter_comparison.png")
    print("- bayesian_trace_[model].png")
    print("- bayesian_model_comparison.png")
    print("- bayesian_posterior_predictions.png")
    print("- bayesian_trace_[model].nc")
    print("- bayesian_analysis_summary.txt")


if __name__ == '__main__':
    main()
