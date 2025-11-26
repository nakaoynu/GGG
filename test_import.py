"""
インポート時間を測定するテストスクリプト
"""
import time

print("=== PyMC インポートテスト ===")
print("開始時刻:", time.strftime("%H:%M:%S"))
print("インポート中... (数分かかる場合があります)\n")

start = time.time()

print("[1/7] numpy をインポート中...")
import numpy as np
print(f"  ✓ 完了 ({time.time() - start:.1f}秒)")

print("[2/7] pandas をインポート中...")
import pandas as pd
print(f"  ✓ 完了 ({time.time() - start:.1f}秒)")

print("[3/7] scipy をインポート中...")
import scipy
print(f"  ✓ 完了 ({time.time() - start:.1f}秒)")

print("[4/7] matplotlib をインポート中...")
import matplotlib.pyplot as plt
print(f"  ✓ 完了 ({time.time() - start:.1f}秒)")

print("[5/7] pytensor をインポート中...")
import pytensor
print(f"  ✓ 完了 ({time.time() - start:.1f}秒)")

print("[6/7] arviz をインポート中...")
import arviz as az
print(f"  ✓ 完了 ({time.time() - start:.1f}秒)")

print("[7/7] pymc をインポート中... (これが最も時間がかかります)")
import pymc as pm
print(f"  ✓ 完了 ({time.time() - start:.1f}秒)")

total_time = time.time() - start
print(f"\n✅ 全インポート完了!")
print(f"総所要時間: {total_time:.1f}秒 ({total_time/60:.1f}分)")
print("PyMCバージョン:", pm.__version__)
