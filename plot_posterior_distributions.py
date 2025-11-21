# plot_posterior_distributions.py - 事後分布プロット作成プログラム
#
# 【概要】
# unified_weighted_bayesian_fitting.pyで保存されたtraceファイルから
# パラメータの事後分布をプロットします
#
# 【機能】
# - traceファイル(.nc)の読み込み
# - トレースプロット（サンプリング履歴）
# - 事後分布ヒストグラム
# - ペアプロット（パラメータ間の相関）
# - 診断統計の表示（R-hat, ESS）
# - 結果を画像ファイルとして保存

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import argparse
from typing import Optional, List

try:
    import japanize_matplotlib
except ImportError:
    print("警告: japanize_matplotlib が見つかりません。日本語表示に問題が生じる可能性があります。")

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10

def load_trace(trace_path: pathlib.Path) -> az.InferenceData:
    """traceファイルを読み込む"""
    print(f"\n📂 Traceファイルを読み込み中: {trace_path}")
    try:
        trace = az.from_netcdf(trace_path)
        print("✅ 読み込み完了")
        return trace
    except Exception as e:
        raise RuntimeError(f"Traceファイルの読み込みに失敗しました: {e}")

def print_summary_statistics(trace: az.InferenceData, var_names: Optional[List[str]] = None):
    """サマリー統計を表示"""
    print("\n" + "="*70)
    print("パラメータのサマリー統計")
    print("="*70)
    
    summary = az.summary(trace, var_names=var_names)
    print(summary)
    
    # R-hatの警告チェック
    if 'r_hat' in summary.columns:
        high_rhat = summary[summary['r_hat'] > 1.01]
        if len(high_rhat) > 0:
            print("\n⚠️ 警告: 以下のパラメータでR-hat > 1.01が検出されました:")
            print(high_rhat[['mean', 'r_hat']])
        else:
            print("\n✅ 全てのパラメータでR-hat ≤ 1.01（収束良好）")
    
    # ESSの警告チェック
    if 'ess_bulk' in summary.columns:
        low_ess = summary[summary['ess_bulk'] < 400]
        if len(low_ess) > 0:
            print("\n⚠️ 警告: 以下のパラメータでESS < 400が検出されました:")
            print(low_ess[['mean', 'ess_bulk', 'ess_tail']])
        else:
            print("\n✅ 全てのパラメータでESS ≥ 400（サンプル数十分）")

def plot_trace(trace: az.InferenceData, output_dir: pathlib.Path, 
               var_names: Optional[List[str]] = None, model_name: str = ""):
    """トレースプロット（サンプリング履歴と事後分布）を作成"""
    print(f"\n📊 トレースプロットを作成中...")
    
    fig, axes = plt.subplots(figsize=(12, 8))
    az.plot_trace(trace, var_names=var_names, compact=True, 
                  backend_kwargs={'figsize': (12, 8)})
    
    plt.tight_layout()
    output_file = output_dir / f'trace_plot_{model_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✅ 保存: {output_file}")
    plt.close()

def plot_posterior(trace: az.InferenceData, output_dir: pathlib.Path,
                   var_names: Optional[List[str]] = None, model_name: str = ""):
    """事後分布ヒストグラムを作成"""
    print(f"\n📊 事後分布ヒストグラムを作成中...")
    
    az.plot_posterior(trace, var_names=var_names, 
                     hdi_prob=0.95, point_estimate='mean',
                     figsize=(12, 8))
    
    plt.tight_layout()
    output_file = output_dir / f'posterior_plot_{model_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✅ 保存: {output_file}")
    plt.close()

def plot_forest(trace: az.InferenceData, output_dir: pathlib.Path,
                var_names: Optional[List[str]] = None, model_name: str = ""):
    """フォレストプロット（パラメータの信頼区間）を作成"""
    print(f"\n📊 フォレストプロットを作成中...")
    
    az.plot_forest(trace, var_names=var_names, 
                   combined=True, hdi_prob=0.95,
                   figsize=(10, 8))
    
    plt.tight_layout()
    output_file = output_dir / f'forest_plot_{model_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✅ 保存: {output_file}")
    plt.close()

def plot_pair(trace: az.InferenceData, output_dir: pathlib.Path,
              var_names: Optional[List[str]] = None, model_name: str = ""):
    """ペアプロット（パラメータ間の相関）を作成"""
    print(f"\n📊 ペアプロットを作成中...")
    
    try:
        az.plot_pair(trace, var_names=var_names, 
                     kind='kde', marginals=True,
                     figsize=(12, 12))
        
        plt.tight_layout()
        output_file = output_dir / f'pair_plot_{model_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✅ 保存: {output_file}")
        plt.close()
    except Exception as e:
        print(f"  ⚠️ ペアプロットの作成に失敗しました: {e}")

def plot_autocorr(trace: az.InferenceData, output_dir: pathlib.Path,
                  var_names: Optional[List[str]] = None, model_name: str = ""):
    """自己相関プロットを作成"""
    print(f"\n📊 自己相関プロットを作成中...")
    
    try:
        az.plot_autocorr(trace, var_names=var_names, 
                        combined=True, figsize=(12, 8))
        
        plt.tight_layout()
        output_file = output_dir / f'autocorr_plot_{model_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✅ 保存: {output_file}")
        plt.close()
    except Exception as e:
        print(f"  ⚠️ 自己相関プロットの作成に失敗しました: {e}")

def plot_energy(trace: az.InferenceData, output_dir: pathlib.Path, model_name: str = ""):
    """エネルギープロット（HMCサンプラーの診断）を作成"""
    print(f"\n📊 エネルギープロットを作成中...")
    
    try:
        az.plot_energy(trace, figsize=(10, 6))
        
        plt.tight_layout()
        output_file = output_dir / f'energy_plot_{model_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✅ 保存: {output_file}")
        plt.close()
    except Exception as e:
        print(f"  ⚠️ エネルギープロットの作成に失敗しました: {e}")

def create_all_plots(trace_path: pathlib.Path, output_dir: Optional[pathlib.Path] = None,
                     var_names: Optional[List[str]] = None):
    """全てのプロットを作成"""
    
    # traceファイルを読み込み
    trace = load_trace(trace_path)
    
    # モデル名を取得（ファイル名から）
    model_name = trace_path.stem.replace('trace_', '')
    
    # 出力ディレクトリの設定
    if output_dir is None:
        output_dir = trace_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 プロット保存先: {output_dir}")
    
    # サマリー統計を表示
    print_summary_statistics(trace, var_names=var_names)
    
    # プロットする変数名の自動設定
    if var_names is None:
        # 磁気パラメータのみをプロット（gammaは除外）
        var_names = ['a_scale', 'g_factor', 'B4', 'B6', 
                     'log_gamma_mu_base', 'temp_gamma_slope']
    
    # 各種プロットを作成
    plot_trace(trace, output_dir, var_names=var_names, model_name=model_name)
    plot_posterior(trace, output_dir, var_names=var_names, model_name=model_name)
    plot_forest(trace, output_dir, var_names=var_names, model_name=model_name)
    plot_pair(trace, output_dir, var_names=['a_scale', 'g_factor', 'B4', 'B6'], 
              model_name=model_name)
    plot_autocorr(trace, output_dir, var_names=var_names, model_name=model_name)
    plot_energy(trace, output_dir, model_name=model_name)
    
    print(f"\n🎉 全てのプロット作成が完了しました")
    print(f"📁 保存先: {output_dir}")

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description='unified_weighted_bayesian_fitting.pyで保存されたtraceファイルから事後分布プロットを作成'
    )
    parser.add_argument('trace_path', type=str, 
                       help='traceファイルのパス（.nc）')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='プロット保存先ディレクトリ（デフォルト: trace_pathと同じディレクトリ/plots）')
    parser.add_argument('--var-names', nargs='+', default=None,
                       help='プロットする変数名のリスト（デフォルト: 主要パラメータ）')
    
    args = parser.parse_args()
    
    print("="*70)
    print("事後分布プロット作成プログラム")
    print("="*70)
    
    # パスの変換
    trace_path = pathlib.Path(args.trace_path)
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else None
    
    # ファイルの存在確認
    if not trace_path.exists():
        print(f"❌ エラー: traceファイルが見つかりません: {trace_path}")
        return
    
    # プロット作成
    create_all_plots(trace_path, output_dir, var_names=args.var_names)

if __name__ == "__main__":
    # コマンドライン引数なしで実行された場合の対話的使用
    import sys
    if len(sys.argv) == 1:
        print("="*70)
        print("事後分布プロット作成プログラム（対話モード）")
        print("="*70)
        print("\n使用方法:")
        print("  python plot_posterior_distributions.py <trace_path> [--output-dir <dir>] [--var-names <vars>]")
        print("\n例:")
        print("  python plot_posterior_distributions.py analysis_results_unified/run_20251121_120000/trace_H_form.nc")
        print("\n最新の結果を自動検出して実行しますか？ (y/n): ", end="")
        
        response = input().strip().lower()
        if response == 'y':
            # 最新の結果ディレクトリを検索
            results_parent = pathlib.Path("analysis_results_unified")
            if results_parent.exists():
                run_dirs = sorted([d for d in results_parent.glob("run_*") if d.is_dir()], 
                                reverse=True)
                if run_dirs:
                    latest_dir = run_dirs[0]
                    print(f"\n📁 最新の結果ディレクトリ: {latest_dir}")
                    
                    # trace ファイルを検索
                    trace_files = list(latest_dir.glob("trace_*.nc"))
                    if trace_files:
                        print(f"\n見つかったtraceファイル:")
                        for i, tf in enumerate(trace_files):
                            print(f"  {i+1}. {tf.name}")
                        
                        print(f"\nどのファイルをプロットしますか？ (1-{len(trace_files)}, または 'all'): ", end="")
                        choice = input().strip()
                        
                        if choice.lower() == 'all':
                            for tf in trace_files:
                                create_all_plots(tf)
                        else:
                            try:
                                idx = int(choice) - 1
                                if 0 <= idx < len(trace_files):
                                    create_all_plots(trace_files[idx])
                                else:
                                    print("❌ 無効な選択です")
                            except ValueError:
                                print("❌ 無効な入力です")
                    else:
                        print("❌ traceファイルが見つかりませんでした")
                else:
                    print("❌ 結果ディレクトリが見つかりませんでした")
            else:
                print("❌ analysis_results_unified ディレクトリが見つかりませんでした")
    else:
        main()
