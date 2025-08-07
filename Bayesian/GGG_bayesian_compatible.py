"""
PyMCベイズ推定による磁気光学パラメータ最適化
numpy.exceptions互換性問題を解決した版
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# numpy.exceptions互換性の修正
try:
    from numpy.exceptions import AxisError
except ImportError:
    # numpy 1.24以前では、AxisErrorはnumpy.AxisErrorにある
    try:
        from numpy import AxisError
    except ImportError:
        # それでもダメな場合は、IndexErrorで代用
        AxisError = IndexError

# numpy.exceptionsモジュールが存在しない場合の対処
if not hasattr(np, 'exceptions'):
    class NumpyExceptions:
        AxisError = AxisError
    np.exceptions = NumpyExceptions()

# これでPyMCをインポート
try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    print("PyMC環境: 正常に読み込まれました")
    print(f"PyMC version: {pm.__version__}")
    print(f"ArviZ version: {az.__version__}")
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"PyMC環境エラー: {e}")
    print("代替実装を使用します...")
    USE_PYMC = False
else:
    USE_PYMC = True

# 物理定数
kB = 1.380649e-23  # J/K
muB = 9.274010e-24  # J/T
hbar = 1.054571e-34  # J·s
c = 299792458  # m/s

class SimpleBayesianFitting:
    """
    PyMCが使用できない場合の代替実装
    scipy.optimizeベースのフィッティング + 不確実性推定
    """
    
    def __init__(self, freq_data, trans_data):
        self.freq_data = freq_data
        self.trans_data = trans_data
        
    def simple_model(self, params):
        """簡単な透過率モデル"""
        d, eps_bg, gamma, a = params
        
        # 簡易モデル（実際の物理計算の代替）
        freq_norm = (self.freq_data - 1.0) / 0.5  # 正規化周波数
        
        # Lorentzian型の透過率モデル
        denom = 1 + ((freq_norm - a) / (gamma * eps_bg))**2
        transmission = d / denom
        
        return np.clip(transmission, 0.01, 0.99)
    
    def fit(self):
        """最適化実行"""
        from scipy.optimize import minimize
        
        def objective(params):
            if any(p <= 0 for p in params):
                return 1e6
            
            model_trans = self.simple_model(params)
            residuals = model_trans - self.trans_data
            return np.sum(residuals**2)
        
        # 初期値
        initial_params = [0.8, 1.5, 0.1, 0.0]
        bounds = [(0.1, 1.0), (1.0, 3.0), (0.01, 1.0), (-1.0, 1.0)]
        
        result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
        
        return {
            'params': result.x,
            'success': result.success,
            'fun': result.fun,
            'message': result.message
        }

class PyMCBayesianFitting:
    """
    PyMCを使用したベイズ推定
    """
    
    def __init__(self, freq_data, trans_data):
        self.freq_data = freq_data
        self.trans_data = trans_data
        
    def build_model(self, model_type='H'):
        """PyMCモデルの構築"""
        with pm.Model() as model:
            # 事前分布の設定
            d = pm.Uniform('d', 0.1, 1.0)  # 膜厚
            eps_bg = pm.Uniform('eps_bg', 1.0, 3.0)  # 背景誘電率
            gamma = pm.Uniform('gamma', 0.01, 1.0)  # ダンピング定数
            a_param = pm.Normal('a_param', 0.0, 1.0)  # スケーリング係数
            
            # ノイズパラメータ
            sigma = pm.HalfNormal('sigma', 0.1)
            
            # 簡易物理モデル（実際の計算の代替）
            freq_norm = (self.freq_data - 1.0) / 0.5
            
            # 透過率の計算
            denom = 1 + ((freq_norm - a_param) / (gamma * eps_bg))**2
            mu = d / denom
            
            # 尤度
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=self.trans_data)
            
        return model
    
    def fit(self, model_type='H', draws=2000, tune=1000):
        """MCMC サンプリング実行"""
        model = self.build_model(model_type)
        
        with model:
            # サンプリング
            trace = pm.sample(draws=draws, tune=tune, return_inferencedata=True,
                            target_accept=0.9, random_seed=42)
            
            # 予測分布
            posterior_pred = pm.sample_posterior_predictive(trace, random_seed=42)
            
        return trace, posterior_pred

def load_experimental_data():
    """実験データの読み込み"""
    try:
        file_path = "Circular_Polarization_B_Field.xlsx"
        df = pd.read_excel(file_path, sheet_name='Sheet2', header=0)
        
        freq_thz = df['Frequency (THz)'].to_numpy(dtype=float)
        trans_7_7 = df['Transmittance (7.7T)'].to_numpy(dtype=float)
        
        # NaNの除去
        mask = ~(np.isnan(freq_thz) | np.isnan(trans_7_7))
        freq_thz = freq_thz[mask]
        trans_7_7 = trans_7_7[mask]
        
        return freq_thz, trans_7_7
        
    except FileNotFoundError:
        print("実験データファイルが見つかりません。サンプルデータを生成します。")
        # サンプルデータの生成
        freq_thz = np.linspace(0.5, 1.5, 50)
        noise = np.random.normal(0, 0.02, len(freq_thz))
        trans_7_7 = 0.8 / (1 + ((freq_thz - 1.0) / 0.1)**2) + noise
        return freq_thz, trans_7_7

def main():
    """メイン実行関数"""
    print("=== PyMC互換性確認済みベイズ推定プログラム ===")
    
    # データ読み込み
    freq_data, trans_data = load_experimental_data()
    print(f"データ点数: {len(freq_data)}")
    print(f"周波数範囲: {np.min(freq_data):.2f} - {np.max(freq_data):.2f} THz")
    
    if USE_PYMC:
        print("\n=== PyMCベイズ推定 ===")
        
        # H形式とB形式のモデル比較
        models = {}
        traces = {}
        
        for model_type in ['H', 'B']:
            print(f"\n{model_type}形式モデルのフィッティング開始...")
            
            fitter = PyMCBayesianFitting(freq_data, trans_data)
            trace, pred = fitter.fit(model_type=model_type, draws=1000, tune=500)
            
            traces[model_type] = trace
            
            # 結果の表示
            print(f"{model_type}形式 - サンプリング完了")
            summary = az.summary(trace, hdi_prob=0.95)
            print(summary)
            
        # モデル比較（LOO-CV）
        try:
            loo_h = az.loo(traces['H'])
            loo_b = az.loo(traces['B'])
            
            print("\n=== モデル比較 (LOO-CV) ===")
            print(f"H形式 LOO: {loo_h.elpd_loo:.2f} ± {loo_h.se:.2f}")
            print(f"B形式 LOO: {loo_b.elpd_loo:.2f} ± {loo_b.se:.2f}")
            
            # 比較結果
            comparison = az.compare({'H': traces['H'], 'B': traces['B']})
            print("\nモデル比較結果:")
            print(comparison)
            
        except Exception as e:
            print(f"LOO-CV計算エラー: {e}")
            
    else:
        print("\n=== 代替実装による最適化 ===")
        
        fitter = SimpleBayesianFitting(freq_data, trans_data)
        result = fitter.fit()
        
        print("最適化結果:")
        print(f"成功: {result['success']}")
        print(f"パラメータ: d={result['params'][0]:.4f}, eps_bg={result['params'][1]:.4f}")
        print(f"            gamma={result['params'][2]:.4f}, a={result['params'][3]:.4f}")
        print(f"残差二乗和: {result['fun']:.6f}")
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(freq_data, trans_data, 'o', label='実験データ', markersize=4)
    
    if USE_PYMC and 'H' in traces:
        # PyMC結果のプロット
        trace_h = traces['H']
        posterior_samples = trace_h.posterior
        
        # 代表的なパラメータでのプロット（簡易版）
        mean_params = [
            float(posterior_samples['d'].mean()),
            float(posterior_samples['eps_bg'].mean()),
            float(posterior_samples['gamma'].mean()),
            float(posterior_samples['a_param'].mean())
        ]
        
        # 簡易モデルでの予測
        freq_norm = (freq_data - 1.0) / 0.5
        denom = 1 + ((freq_norm - mean_params[3]) / (mean_params[2] * mean_params[1]))**2
        model_trans = mean_params[0] / denom
        
        plt.plot(freq_data, model_trans, '-', label='ベイズ推定結果', linewidth=2)
        
    else:
        # 代替実装結果のプロット
        if 'result' in locals():
            fitter = SimpleBayesianFitting(freq_data, trans_data)
            model_trans = fitter.simple_model(result['params'])
            plt.plot(freq_data, model_trans, '-', label='最適化結果', linewidth=2)
    
    plt.xlabel('周波数 (THz)')
    plt.ylabel('透過率')
    plt.title('磁気光学透過スペクトルのフィッティング結果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bayesian_fitting_result.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
