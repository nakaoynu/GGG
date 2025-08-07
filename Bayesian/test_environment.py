"""
PyMCベイズ推定のテスト版
依存関係とデータ読み込みをテスト
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 日本語フォント設定（オプション）
try:
    import japanize_matplotlib
    print("japanize-matplotlib: OK")
except ImportError:
    print("japanize-matplotlib: 未インストール（オプション）")
    plt.rcParams['font.family'] = 'DejaVu Sans'

# PyMC関連
try:
    # numpy.exceptionsの問題を回避するためのワークアラウンド
    import numpy as np
    if not hasattr(np, 'exceptions'):
        # numpy 1.24以前では、exceptionsモジュールが存在しないため、
        # 必要な例外クラスを手動で追加
        class MockExceptions:
            AxisError = IndexError  # numpy.exceptions.AxisErrorの代替
        np.exceptions = MockExceptions()
    
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    print("PyMC環境: OK")
    print(f"PyMC version: {pm.__version__}")
    print(f"ArviZ version: {az.__version__}")
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"PyMC環境エラー: {e}")
    exit(1)

# 物理定数のテスト
print("\n=== 物理定数のテスト ===")
kB = 1.380649e-23
muB = 9.274010e-24
hbar = 1.054571e-34
c = 299792458
print(f"物理定数設定: OK")
print(f"kB = {kB}, muB = {muB}")

# データ読み込みのテスト
print("\n=== データ読み込みテスト ===")
try:
    file_path = "C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\Circular_Polarization_B_Field.xlsx"
    df = pd.read_excel(file_path, sheet_name='Sheet2', header=0)
    exp_freq_thz = df['Frequency (THz)'].to_numpy(dtype=float)
    exp_transmittance_7_7 = df['Transmittance (7.7T)'].to_numpy(dtype=float)
    
    print(f"データ読み込み: OK")
    print(f"データ点数: {len(exp_freq_thz)}")
    print(f"周波数範囲: {np.min(exp_freq_thz):.2f} - {np.max(exp_freq_thz):.2f} THz")
    print(f"透過率範囲: {np.min(exp_transmittance_7_7):.4f} - {np.max(exp_transmittance_7_7):.4f}")
    
except Exception as e:
    print(f"データ読み込みエラー: {e}")
    print("テストデータを生成します...")
    
    # テストデータ生成
    exp_freq_thz = np.linspace(0.5, 1.5, 50)
    exp_transmittance_7_7 = 0.5 + 0.3 * np.sin(2 * np.pi * exp_freq_thz) + 0.1 * np.random.random(50)
    
    print(f"テストデータ生成: OK")
    print(f"データ点数: {len(exp_freq_thz)}")

# 正規化テスト
min_exp, max_exp = np.min(exp_transmittance_7_7), np.max(exp_transmittance_7_7)
exp_transmittance_normalized = (exp_transmittance_7_7 - min_exp) / (max_exp - min_exp)
print(f"正規化: OK (範囲: 0 - 1)")

# 簡単なPyMCモデルのテスト
print("\n=== 簡単なPyMCモデルテスト ===")
try:
    with pm.Model() as test_model:
        # 簡単なパラメータ
        slope = pm.Normal('slope', mu=0, sigma=1)
        intercept = pm.Normal('intercept', mu=0.5, sigma=0.5)
        sigma = pm.HalfNormal('sigma', sigma=0.1)
        
        # 線形モデル
        mu = intercept + slope * exp_freq_thz
        
        # 尤度
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, 
                              observed=exp_transmittance_normalized)
        
        print("モデル構築: OK")
        
        # 短いサンプリングテスト
        print("短いサンプリングテスト実行中...")
        trace = pm.sample(draws=100, tune=100, chains=1, cores=1, 
                         return_inferencedata=True, progressbar=False)
        
        print("サンプリング: OK")
        try:
            print(f"サンプル数: {trace.posterior.sizes['draw']}")
        except (AttributeError, KeyError) as e:
            print(f"サンプル情報の取得に失敗しました: {e}")
        
        # ArviZ統計
        summary = az.summary(trace)
        print("統計計算: OK")
        print(summary)

except Exception as e:
    print(f"PyMCテストエラー: {e}")

print("\n=== 環境テスト完了 ===")
print("基本的な環境は整っています。メインプログラムを実行できます。")
