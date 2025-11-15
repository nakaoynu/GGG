"""LOO-CVを用いたベイズモデル比較スクリプト

保存されたPyMCトレースからLOO-CV（Leave-One-Out Cross-Validation）を計算し、
H_formとB_formモデルを比較します。

使用方法:
    python model_comparison_loo.py --results-dir "analysis_results/run_20251110_134403"
"""

import argparse
import pathlib
import warnings
from typing import Dict, Any

import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    import japanize_matplotlib
except ImportError:
    warnings.warn("japanize_matplotlib が見つかりません。日本語表示が正しく行われない可能性があります。")


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description="LOO-CVを用いてH_formとB_formモデルを比較します"
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        type=pathlib.Path,
        help="trace_H_form.ncとtrace_B_form.ncを含むディレクトリへのパス",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="結果を保存するディレクトリ（省略時は--results-dirと同じ）",
    )
    return parser.parse_args()


def load_trace(trace_path: pathlib.Path) -> az.InferenceData:
    """トレースファイルを読み込む"""
    if not trace_path.exists():
        raise FileNotFoundError(f"トレースファイルが見つかりません: {trace_path}")
    
    print(f"📖 トレースを読み込み中: {trace_path.name}")
    trace = az.from_netcdf(trace_path)
    return trace


def compute_loo(trace: az.InferenceData, model_name: str) -> az.ELPDData:
    """LOO-CVを計算"""
    print(f"\n🔬 {model_name} のLOO-CV計算中...")
    
    try:
        loo_result = az.loo(trace, pointwise=True)
        print(f"✅ {model_name} のLOO計算完了")
        return loo_result
    except Exception as e:
        print(f"❌ {model_name} のLOO計算に失敗: {e}")
        raise


def compare_models(
    loo_h: az.ELPDData, 
    loo_b: az.ELPDData,
    trace_h: az.InferenceData,
    trace_b: az.InferenceData
) -> pd.DataFrame:
    """2つのモデルをLOO-CVで比較"""
    print("\n📊 モデル比較を実行中...")
    
    # ArviZのcompare関数を使用
    model_dict = {
        "H_form": trace_h,
        "B_form": trace_b
    }
    
    comparison = az.compare(model_dict, ic="loo")
    return comparison


def print_loo_summary(loo_result: az.ELPDData, model_name: str) -> None:
    """LOO結果のサマリーを表示"""
    print(f"\n{'='*60}")
    print(f"{model_name} モデル - LOO結果サマリー")
    print(f"{'='*60}")
    print(f"ELPD_loo (expected log pointwise predictive density):")
    print(f"  推定値: {loo_result.elpd_loo:.2f}")
    print(f"  標準誤差: {loo_result.se:.2f}")
    print(f"\np_loo (effective number of parameters):")
    print(f"  {loo_result.p_loo:.2f}")
    print(f"\nLOO-IC (lower is better):")
    print(f"  {loo_result.loo:.2f}")
    
    # Pareto k診断
    if hasattr(loo_result, 'pareto_k'):
        k_values = loo_result.pareto_k
        k_bad = np.sum(k_values > 0.7)
        k_warning = np.sum((k_values > 0.5) & (k_values <= 0.7))
        k_good = np.sum(k_values <= 0.5)
        
        print(f"\nPareto k 診断:")
        print(f"  良好 (k ≤ 0.5): {k_good} 点")
        print(f"  注意 (0.5 < k ≤ 0.7): {k_warning} 点")
        print(f"  問題あり (k > 0.7): {k_bad} 点")
        
        if k_bad > 0:
            print(f"  ⚠️ {k_bad} 個のデータポイントでPareto k > 0.7")
            print(f"     LOO推定の信頼性が低い可能性があります")


def print_comparison_summary(comparison_df: pd.DataFrame) -> None:
    """モデル比較結果のサマリーを表示"""
    print(f"\n{'='*60}")
    print("モデル比較結果 (LOO-CV)")
    print(f"{'='*60}")
    print("\n", comparison_df.to_string())
    
    print(f"\n{'='*60}")
    print("解釈:")
    print(f"{'='*60}")
    
    best_model = comparison_df.index[0]
    rank_col = 'rank' if 'rank' in comparison_df.columns else comparison_df.columns[0]
    
    print(f"✅ 最良モデル: {best_model}")
    
    if len(comparison_df) > 1:
        # elpd_diff と dse を確実に float に変換して型エラーを回避する
        try:
            elpd_diff_raw = comparison_df.loc[comparison_df.index[1], 'elpd_diff']
            dse_raw = comparison_df.loc[comparison_df.index[1], 'dse']
            # numpy や pandas の特殊型にも対応するため np.asarray を経由して float にする
            elpd_diff = float(np.asarray(elpd_diff_raw))
            dse = float(np.asarray(dse_raw))
        except Exception as e:
            print(f"⚠️ elpd_diff / dse の取得または変換に失敗しました: {e}")
            print("   モデル間差の統計的判定をスキップします。")
            return
        
        print(f"\nELPD差分 (expected log pointwise predictive density difference):")
        print(f"  {comparison_df.index[1]} vs {best_model}: {elpd_diff:.2f} ± {dse:.2f}")
        
        # 比較は float 型で行う（型問題を回避）
        if abs(elpd_diff) < 2.0 * dse:
            print(f"\n💡 判定: モデル間の差は統計的に有意ではありません")
            print(f"   (|ELPD差分| < 2×標準誤差)")
        elif abs(elpd_diff) < 4.0 * dse:
            print(f"\n💡 判定: {best_model} がやや優れていますが、差は小さいです")
            print(f"   (2×標準誤差 ≤ |ELPD差分| < 4×標準誤差)")
        else:
            print(f"\n💡 判定: {best_model} が明確に優れています")
            print(f"   (|ELPD差分| ≥ 4×標準誤差)")
        
        # Weight (Akaike weight) の解釈
        if 'weight' in comparison_df.columns:
            weight_best = comparison_df.loc[best_model, 'weight']
            print(f"\nモデル重み (Akaike weight):")
            for idx in comparison_df.index:
                weight = comparison_df.loc[idx, 'weight']
                try:
                    weight_f = float(np.asarray(weight))
                    print(f"  {idx}: {weight_f:.3f} ({weight_f*100:.1f}%)")
                except Exception:
                    print(f"  {idx}: {weight} (変換不可)")


def plot_loo_comparison(
    loo_h: az.ELPDData,
    loo_b: az.ELPDData,
    output_path: pathlib.Path
) -> None:
    """LOO比較プロットを作成"""
    print("\n📈 LOO比較プロットを作成中...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Pareto k 診断プロット (H_form)
    ax1 = axes[0, 0]
    if hasattr(loo_h, 'pareto_k'):
        k_h = loo_h.pareto_k
        ax1.scatter(range(len(k_h)), k_h, alpha=0.6, s=20, color='red', label='H_form')
        ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='境界 (0.5)')
        ax1.axhline(y=0.7, color='darkred', linestyle='--', alpha=0.7, label='閾値 (0.7)')
        ax1.set_xlabel('データポイントインデックス')
        ax1.set_ylabel('Pareto k')
        ax1.set_title('H_form: Pareto k 診断')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Pareto k 診断プロット (B_form)
    ax2 = axes[0, 1]
    if hasattr(loo_b, 'pareto_k'):
        k_b = loo_b.pareto_k
        ax2.scatter(range(len(k_b)), k_b, alpha=0.6, s=20, color='blue', label='B_form')
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='境界 (0.5)')
        ax2.axhline(y=0.7, color='darkred', linestyle='--', alpha=0.7, label='閾値 (0.7)')
        ax2.set_xlabel('データポイントインデックス')
        ax2.set_ylabel('Pareto k')
        ax2.set_title('B_form: Pareto k 診断')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # ELPD比較
    ax3 = axes[1, 0]
    models = ['H_form', 'B_form']
    elpd_values = [loo_h.elpd_loo, loo_b.elpd_loo]
    se_values = [loo_h.se, loo_b.se]
    colors = ['red', 'blue']
    
    x_pos = np.arange(len(models))
    ax3.bar(x_pos, elpd_values, color=colors, alpha=0.7, yerr=se_values, capsize=10)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models)
    ax3.set_ylabel('ELPD_loo')
    ax3.set_title('ELPD比較 (高いほど良い)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # LOO-IC比較
    ax4 = axes[1, 1]
    loo_ic_values = [loo_h.loo, loo_b.loo]
    ax4.bar(x_pos, loo_ic_values, color=colors, alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(models)
    ax4.set_ylabel('LOO-IC')
    ax4.set_title('LOO-IC比較 (低いほど良い)')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ プロットを保存: {output_path.name}")


def save_results_to_csv(
    loo_h: az.ELPDData,
    loo_b: az.ELPDData,
    comparison_df: pd.DataFrame,
    output_dir: pathlib.Path
) -> None:
    """結果をCSVに保存"""
    print("\n💾 結果をCSVに保存中...")
    
    # 個別モデルのLOO結果
    loo_summary = pd.DataFrame({
        'Model': ['H_form', 'B_form'],
        'ELPD_loo': [loo_h.elpd_loo, loo_b.elpd_loo],
        'SE': [loo_h.se, loo_b.se],
        'p_loo': [loo_h.p_loo, loo_b.p_loo],
        'LOO_IC': [loo_h.loo, loo_b.loo],
    })
    
    loo_summary_path = output_dir / "loo_summary.csv"
    loo_summary.to_csv(loo_summary_path, index=False)
    print(f"✅ LOOサマリーを保存: {loo_summary_path.name}")
    
    # モデル比較結果
    comparison_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path)
    print(f"✅ モデル比較結果を保存: {comparison_path.name}")
    
    # Pareto k 値の詳細
    if hasattr(loo_h, 'pareto_k') and hasattr(loo_b, 'pareto_k'):
        pareto_df = pd.DataFrame({
            'Data_Point': range(len(loo_h.pareto_k)),
            'H_form_pareto_k': loo_h.pareto_k,
            'B_form_pareto_k': loo_b.pareto_k,
        })
        pareto_path = output_dir / "pareto_k_values.csv"
        pareto_df.to_csv(pareto_path, index=False)
        print(f"✅ Pareto k値を保存: {pareto_path.name}")


def main() -> None:
    """メイン実行関数"""
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    
    if not results_dir.exists():
        raise FileNotFoundError(f"結果ディレクトリが存在しません: {results_dir}")
    
    # 出力ディレクトリの設定
    output_dir = args.output_dir if args.output_dir else results_dir
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("LOO-CV モデル比較")
    print("="*60)
    print(f"📁 入力ディレクトリ: {results_dir}")
    print(f"📁 出力ディレクトリ: {output_dir}")
    
    # トレースの読み込み
    trace_h_path = results_dir / "trace_H_form.nc"
    trace_b_path = results_dir / "trace_B_form.nc"
    
    trace_h = load_trace(trace_h_path)
    trace_b = load_trace(trace_b_path)
    
    # LOO計算
    loo_h = compute_loo(trace_h, "H_form")
    loo_b = compute_loo(trace_b, "B_form")
    
    # 個別結果の表示
    print_loo_summary(loo_h, "H_form")
    print_loo_summary(loo_b, "B_form")
    
    # モデル比較
    comparison_df = compare_models(loo_h, loo_b, trace_h, trace_b)
    print_comparison_summary(comparison_df)
    
    # プロットの作成
    plot_path = output_dir / "loo_comparison.png"
    plot_loo_comparison(loo_h, loo_b, plot_path)
    
    # 結果の保存
    save_results_to_csv(loo_h, loo_b, comparison_df, output_dir)
    
    print("\n" + "="*60)
    print("🎉 LOO-CVモデル比較が完了しました")
    print(f"📁 結果は '{output_dir}' に保存されています")
    print("="*60)


if __name__ == "__main__":
    main()
