"""
パラメータ相関分析スクリプト v2
H_formとB_formモデルの結果を詳しく分析し、比較する

機能:
- 両モデルのパラメータ相関分析
- 温度依存gamma（完全版）と2パラメータgamma（v1版）の両方に対応
- 両モデルの比較プロット
- コマンドライン引数で結果ディレクトリを指定可能【必須】

使用例:    
    # カスタムディレクトリで2パラメータgammaモデル
    python analyze_parameter_correlation_v2.py --results-dir "path/to/results" --gamma-type 2param
"""

import argparse
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pandas as pd
from matplotlib import rcParams
from typing import Dict

# 日本語フォント設定
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic']
rcParams['axes.unicode_minus'] = False


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description="PyMCトレースからパラメータ相関を分析"
    )
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        required=True,
        help="trace_*.ncファイルを含む結果ディレクトリへのパス（必須）"
    )
    parser.add_argument(
        "--gamma-type",
        choices=["temp_dependent", "2param"],
        default="temp_dependent",
        help="gammaモデルのタイプ: temp_dependent(温度依存) or 2param(v1版)"
    )
    return parser.parse_args()


def load_trace(results_dir: pathlib.Path, model_type: str = 'B_form'):
    """
    保存されたトレースデータを読み込む
    
    Parameters
    ----------
    results_dir : pathlib.Path
        結果ディレクトリ
    model_type : str
        'H_form' または 'B_form'
    
    Returns
    -------
    trace : az.InferenceData
        PyMCのトレースデータ
    """
    trace_path = results_dir / f"trace_{model_type}.nc"
    
    if not trace_path.exists():
        raise FileNotFoundError(f"トレースファイルが見つかりません: {trace_path}")
    
    print(f"✅ トレースデータを読み込み中: {trace_path}")
    trace = az.from_netcdf(trace_path)
    return trace


def plot_pair_correlation(trace, results_dir: pathlib.Path, model_type: str = 'B_form', save: bool = True):
    """
    主要パラメータのペアプロット（相関行列）を作成
    
    Parameters
    ----------
    trace : az.InferenceData
        トレースデータ
    results_dir : pathlib.Path
        結果ディレクトリ
    model_type : str
        モデル名
    save : bool
        画像を保存するか
    """
    print(f"\n--- {model_type}モデルのパラメータ相関を可視化中 ---")
    
    # 主要な磁気パラメータ
    var_names = ['a_scale', 'g_factor', 'B4', 'B6']
    
    # ペアプロット作成
    fig = az.plot_pair(
        trace,
        var_names=var_names,
        kind='kde',  # カーネル密度推定
        marginals=True,  # 周辺分布も表示
        point_estimate='mean',  # 平均値を表示
        figsize=(12, 12)
    )
    
    plt.suptitle(f'{model_type}モデル: パラメータ相関マトリクス', fontsize=16, y=0.995)
    plt.tight_layout()
    
    if save:
        save_path = results_dir / f'parameter_correlation_{model_type}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 相関プロットを保存: {save_path}")


def plot_gamma_parameters(trace, results_dir: pathlib.Path, model_type: str = 'B_form', 
                         gamma_type: str = 'temp_dependent', save: bool = True):
    """
    Gammaパラメータの可視化（温度依存 or 2パラメータ）
    
    Parameters
    ----------
    trace : az.InferenceData
        トレースデータ
    results_dir : pathlib.Path
        結果ディレクトリ
    model_type : str
        モデル名
    gamma_type : str
        'temp_dependent' または '2param'
    save : bool
        画像を保存するか
    """
    print(f"\n--- {model_type}モデルのGammaパラメータを可視化中 ---")
    
    # gamma_typeに応じてパラメータ名を選択
    if gamma_type == '2param':
        gamma_vars = ['log_gamma_min', 'log_gamma_other']
    else:
        gamma_vars = ['log_gamma_mu_base', 'log_gamma_sigma_base', 'temp_gamma_slope']
    
    # 存在するパラメータのみ抽出
    available_vars = [v for v in gamma_vars if v in trace.posterior.data_vars]
    
    if not available_vars:
        print("⚠️ Gammaパラメータが見つかりません")
        return
    
    # ペアプロット
    fig = az.plot_pair(
        trace,
        var_names=available_vars,
        kind='kde',
        marginals=True,
        point_estimate='mean',
        figsize=(10, 10)
    )
    
    gamma_label = '2パラメータGamma' if gamma_type == '2param' else '温度依存Gamma'
    plt.suptitle(f'{model_type}モデル: {gamma_label}パラメータ相関', fontsize=16, y=0.995)
    plt.tight_layout()
    
    if save:
        save_path = results_dir / f'gamma_correlation_{model_type}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gamma相関プロットを保存: {save_path}")
    

def plot_autocorrelation(trace, results_dir: pathlib.Path, model_type: str = 'B_form', save: bool = True):
    """
    自己相関プロット（サンプリングの独立性確認）
    
    Parameters
    ----------
    trace : az.InferenceData
        トレースデータ
    results_dir : pathlib.Path
        結果ディレクトリ
    model_type : str
        モデル名
    save : bool
        画像を保存するか
    """
    print(f"\n--- {model_type}モデルの自己相関を確認中 ---")
    
    var_names = ['a_scale', 'g_factor', 'B4', 'B6']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, var in enumerate(var_names):
        az.plot_autocorr(trace, var_names=[var], ax=axes[i], max_lag=100)
        axes[i].set_title(f'{var} の自己相関', fontsize=12)
    
    plt.suptitle(f'{model_type}モデル: サンプリング自己相関', fontsize=16)
    plt.tight_layout()
    
    if save:
        save_path = results_dir / f'autocorrelation_{model_type}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 自己相関プロットを保存: {save_path}")
    
    plt.show()


def calculate_correlation_matrix(trace, results_dir: pathlib.Path, model_type: str = 'B_form'):
    """
    パラメータ間の相関係数行列を計算
    
    Parameters
    ----------
    trace : az.InferenceData
        トレースデータ
    results_dir : pathlib.Path
        結果ディレクトリ
    model_type : str
        モデル名
    
    Returns
    -------
    corr_df : pd.DataFrame
        相関係数行列
    """
    print(f"\n--- {model_type}モデルの相関係数を計算中 ---")
    
    var_names = ['a_scale', 'g_factor', 'B4', 'B6']
    
    # 事後分布サンプルを抽出
    samples = {}
    for var in var_names:
        if var in trace.posterior.data_vars:
            # 全チェーンを結合してフラット化
            samples[var] = trace.posterior[var].values.flatten()
    
    # DataFrameに変換
    df = pd.DataFrame(samples)
    
    # 相関係数行列を計算
    corr_matrix = df.corr()
    
    print("\n📊 パラメータ相関係数行列:")
    print(corr_matrix.round(4))
    
    # CSV保存
    csv_path = results_dir / f'correlation_matrix_{model_type}.csv'
    corr_matrix.to_csv(csv_path)
    print(f"\n✅ 相関係数行列を保存: {csv_path}")
    
    # ヒートマップ表示
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # 軸ラベル設定
    ax.set_xticks(np.arange(len(var_names)))
    ax.set_yticks(np.arange(len(var_names)))
    ax.set_xticklabels(var_names)
    ax.set_yticklabels(var_names)
    
    # 相関係数を表示
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=11)
    
    ax.set_title(f'{model_type}モデル: パラメータ相関係数', fontsize=14)
    plt.colorbar(im, ax=ax, label='相関係数')
    plt.tight_layout()
    
    # 保存
    heatmap_path = results_dir / f'correlation_heatmap_{model_type}.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✅ 相関ヒートマップを保存: {heatmap_path}")
    
    plt.show()
    
    return corr_matrix


def print_summary_statistics(trace, results_dir: pathlib.Path, model_type: str = 'B_form', 
                            gamma_type: str = 'temp_dependent'):
    """
    パラメータの統計サマリーを表示
    
    Parameters
    ----------
    trace : az.InferenceData
        トレースデータ
    results_dir : pathlib.Path
        結果ディレクトリ
    model_type : str
        モデル名
    gamma_type : str
        'temp_dependent' または '2param'
    """
    print(f"\n{'='*60}")
    print(f"📊 {model_type}モデル: パラメータ統計サマリー")
    print(f"{'='*60}\n")
    
    # gamma_typeに応じてパラメータリストを選択
    base_vars = ['a_scale', 'g_factor', 'B4', 'B6']
    if gamma_type == '2param':
        var_names = base_vars + ['log_gamma_min', 'log_gamma_other']
    else:
        var_names = base_vars + ['log_gamma_mu_base', 'log_gamma_sigma_base', 'temp_gamma_slope']
    
    # 存在するパラメータのみ
    available_vars = [v for v in var_names if v in trace.posterior.data_vars]
    
    summary = az.summary(trace, var_names=available_vars, round_to=6)
    print(summary)
    
    # CSV保存
    csv_path = results_dir / f'parameter_summary_{model_type}.csv'
    summary.to_csv(csv_path)
    print(f"\n✅ サマリーを保存: {csv_path}")


def compare_models(trace_h, trace_b, results_dir: pathlib.Path, save: bool = True):
    """
    H_formとB_formの両モデルを比較
    
    Parameters
    ----------
    trace_h : az.InferenceData
        H_formのトレースデータ
    trace_b : az.InferenceData
        B_formのトレースデータ
    results_dir : pathlib.Path
        結果ディレクトリ
    save : bool
        画像を保存するか
    """
    print("\n" + "="*70)
    print("🔍 H_form vs B_form モデル比較")
    print("="*70)
    
    var_names = ['a_scale', 'g_factor', 'B4', 'B6']
    
    # 各パラメータの事後分布を比較
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()
    
    for i, var in enumerate(var_names):
        ax = axes[i]
        
        # H_formの分布
        if var in trace_h.posterior.data_vars:
            samples_h = trace_h.posterior[var].values.flatten()
            ax.hist(samples_h, bins=50, alpha=0.6, color='red', label='H_form', density=True)
        
        # B_formの分布
        if var in trace_b.posterior.data_vars:
            samples_b = trace_b.posterior[var].values.flatten()
            ax.hist(samples_b, bins=50, alpha=0.6, color='blue', label='B_form', density=True)
        
        ax.set_xlabel(var, fontsize=12, labelpad=8)
        ax.set_ylabel('密度', fontsize=12, labelpad=8)
        ax.set_title(f'{var}の事後分布比較', fontsize=13, pad=12)
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=10)
    
    plt.suptitle('H_form vs B_form: パラメータ事後分布比較', fontsize=17, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0, w_pad=2.5)
    
    if save:
        save_path = results_dir / 'model_comparison_posterior.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ モデル比較プロットを保存: {save_path}")
    
    plt.show()
    
    # パラメータ統計の比較表を作成
    print("\n📊 パラメータ平均値の比較:")
    comparison_data = []
    
    for var in var_names:
        row = {'Parameter': var}
        
        if var in trace_h.posterior.data_vars:
            samples_h = trace_h.posterior[var].values.flatten()
            row['H_form_mean'] = np.mean(samples_h)
            row['H_form_std'] = np.std(samples_h)
        else:
            row['H_form_mean'] = np.nan
            row['H_form_std'] = np.nan
        
        if var in trace_b.posterior.data_vars:
            samples_b = trace_b.posterior[var].values.flatten()
            row['B_form_mean'] = np.mean(samples_b)
            row['B_form_std'] = np.std(samples_b)
        else:
            row['B_form_mean'] = np.nan
            row['B_form_std'] = np.nan
        
        # 差分を計算
        if not np.isnan(row['H_form_mean']) and not np.isnan(row['B_form_mean']):
            row['Difference'] = row['H_form_mean'] - row['B_form_mean']
            row['Relative_diff_%'] = (row['Difference'] / row['B_form_mean']) * 100
        else:
            row['Difference'] = np.nan
            row['Relative_diff_%'] = np.nan
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # CSV保存
    csv_path = results_dir / 'model_comparison_summary.csv'
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n✅ モデル比較サマリーを保存: {csv_path}")


def main():
    """メイン実行関数"""
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    gamma_type = args.gamma_type
    
    if not results_dir.exists():
        raise FileNotFoundError(f"結果ディレクトリが存在しません: {results_dir}")
    
    print("="*70)
    print("🔬 パラメータ相関分析スクリプト v2")
    print(f"📁 結果ディレクトリ: {results_dir}")
    print(f"🔧 Gammaモデルタイプ: {gamma_type}")
    print("="*70)
    
    traces: Dict[str, az.InferenceData] = {}
    
    # 両モデルのトレースを読み込み・分析
    for model_type in ['B_form', 'H_form']:
        try:
            print(f"\n【{model_type}モデルの読み込みと分析】")
            trace = load_trace(results_dir, model_type)
            traces[model_type] = trace
            
            # 統計サマリー
            print_summary_statistics(trace, results_dir, model_type, gamma_type)
            
            # 相関係数行列
            calculate_correlation_matrix(trace, results_dir, model_type)
            
            # パラメータ相関プロット
            plot_pair_correlation(trace, results_dir, model_type)
            
            # Gamma相関プロット
            plot_gamma_parameters(trace, results_dir, model_type, gamma_type)
            
            # 自己相関プロット
            plot_autocorrelation(trace, results_dir, model_type)
                        
        except Exception as e:
            print(f"❌ {model_type}モデルの分析でエラー: {e}")
    
    # 両モデルの比較
    if 'H_form' in traces and 'B_form' in traces:
        try:
            compare_models(traces['H_form'], traces['B_form'], results_dir)
        except Exception as e:
            print(f"❌ モデル比較でエラー: {e}")
    
    print("\n" + "="*70)
    print("✅ 全ての相関分析が完了しました")
    print(f"📁 結果は {results_dir} に保存されています")
    print("="*70)


if __name__ == "__main__":
    main()
