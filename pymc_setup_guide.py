"""
PyMC互換性問題の解決ガイド
"""

print("=== PyMC環境セットアップガイド ===")
print()
print("現在のエラー: No module named 'numpy.exceptions'")
print("原因: PyMCが要求するnumpyバージョンと環境のnumpyバージョンの不整合")
print()
print("解決方法1: conda環境の再構築")
print("以下のコマンドを順番に実行してください:")
print()
print("1. conda deactivate")
print("2. conda create -n pymc-new python=3.10 -y")
print("3. conda activate pymc-new")  
print("4. conda install -c conda-forge pymc arviz numpy>=1.25 -y")
print("5. pip install japanize-matplotlib openpyxl")
print()
print("解決方法2: pip強制アップデート")
print("現在の環境で以下を実行:")
print()
print("1. pip install --upgrade --force-reinstall numpy>=1.25.0")
print("2. pip install --upgrade --force-reinstall pymc")
print("3. conda deactivate && conda activate pymc-env")
print()
print("解決方法3: 代替実装の使用")
print("PyMCを使わずに、scipy + arvizでベイズ的な分析を行う")
print("-> GGG_bayesian_alternative.py を使用")

# 現在の環境情報を表示
try:
    import numpy as np
    print(f"\n現在のnumpy: {np.__version__}")
except:
    print("\nnumpyが読み込めません")

try:
    import pymc as pm
    print(f"PyMC: {pm.__version__}")
except Exception as e:
    print(f"PyMCエラー: {e}")

try:
    import arviz as az
    print(f"ArviZ: {az.__version__}")
except:
    print("ArviZが読み込めません")
