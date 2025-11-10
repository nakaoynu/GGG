"""
パラメータ相関分析スクリプト
B_formモデルの成功した結果を詳しく分析
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pandas as pd
from matplotlib import rcParams

# 日本語フォント設定
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic']
rcParams['axes.unicode_minus'] = False

# 結果ディレクトリのパス
RESULTS_DIR = pathlib.Path(r"c:\Users\taich\OneDrive - YNU(ynu.jp)\master\磁性\GGG\Programs\Bayesian\202510\analysis_results\run_20251104_165830")

def load_trace(model_type='B_form'):
    """
    保存されたトレースデータを読み込む
    
    Parameters:
    -----------
    model_type : str
        'H_form' または 'B_form'
    
    Returns:
    --------
    trace : az.InferenceData
        PyMCのトレースデータ
    """
    trace_path = RESULTS_DIR / f"trace_{model_type}.nc"
    
    if not trace_path.exists():
        raise FileNotFoundError(f"トレースファイルが見つかりません: {trace_path}")
    
    print(f"✅ トレースデータを読み込み中: {trace_path}")
    trace = az.from_netcdf(trace_path)
    return trace


def plot_pair_correlation(trace, model_type='B_form', save=True):
    """
    主要パラメータのペアプロット（相関行列）を作成
    
    Parameters:
    -----------
    trace : az.InferenceData
        トレースデータ
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
        save_path = RESULTS_DIR / f'parameter_correlation_{model_type}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 相関プロットを保存: {save_path}")
    
    plt.show()


def plot_gamma_parameters(trace, model_type='B_form', save=True):
    """
    Gammaパラメータ（温度依存性）の可視化
    
    Parameters:
    -----------
    trace : az.InferenceData
        トレースデータ
    model_type : str
        モデル名
    save : bool
        画像を保存するか
    """
    print(f"\n--- {model_type}モデルのGammaパラメータを可視化中 ---")
    
    # Gammaパラメータ
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
    
    plt.suptitle(f'{model_type}モデル: Gammaパラメータ相関', fontsize=16, y=0.995)
    plt.tight_layout()
    
    if save:
        save_path = RESULTS_DIR / f'gamma_correlation_{model_type}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gamma相関プロットを保存: {save_path}")
    
    plt.show()


def plot_autocorrelation(trace, model_type='B_form', save=True):
    """
    自己相関プロット（サンプリングの独立性確認）
    
    Parameters:
    -----------
    trace : az.InferenceData
        トレースデータ
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
        save_path = RESULTS_DIR / f'autocorrelation_{model_type}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 自己相関プロットを保存: {save_path}")
    
    plt.show()


def calculate_correlation_matrix(trace, model_type='B_form'):
    """
    パラメータ間の相関係数行列を計算
    
    Parameters:
    -----------
    trace : az.InferenceData
        トレースデータ
    model_type : str
        モデル名
    
    Returns:
    --------
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
    csv_path = RESULTS_DIR / f'correlation_matrix_{model_type}.csv'
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
    heatmap_path = RESULTS_DIR / f'correlation_heatmap_{model_type}.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✅ 相関ヒートマップを保存: {heatmap_path}")
    
    plt.show()
    
    return corr_matrix


def plot_posterior_predictive(trace, model_type='B_form', save=True):
    """
    事後予測分布の可視化
    
    Parameters:
    -----------
    trace : az.InferenceData
        トレースデータ
    model_type : str
        モデル名
    save : bool
        画像を保存するか
    """
    print(f"\n--- {model_type}モデルの事後予測チェック ---")
    
    if 'posterior_predictive' not in trace.groups():
        print("⚠️ 事後予測分布が保存されていません")
        return
    
    # 事後予測プロット
    fig, ax = plt.subplots(figsize=(12, 6))
    az.plot_ppc(trace, ax=ax, num_pp_samples=100)
    ax.set_title(f'{model_type}モデル: 事後予測チェック', fontsize=14)
    plt.tight_layout()
    
    if save:
        save_path = RESULTS_DIR / f'posterior_predictive_{model_type}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 事後予測プロットを保存: {save_path}")
    
    plt.show()


def print_summary_statistics(trace, model_type='B_form'):
    """
    パラメータの統計サマリーを表示
    
    Parameters:
    -----------
    trace : az.InferenceData
        トレースデータ
    model_type : str
        モデル名
    """
    print(f"\n{'='*60}")
    print(f"📊 {model_type}モデル: パラメータ統計サマリー")
    print(f"{'='*60}\n")
    
    var_names = ['a_scale', 'g_factor', 'B4', 'B6', 
                 'log_gamma_mu_base', 'log_gamma_sigma_base', 'temp_gamma_slope']
    
    # 存在するパラメータのみ
    available_vars = [v for v in var_names if v in trace.posterior.data_vars]
    
    summary = az.summary(trace, var_names=available_vars, round_to=6)
    print(summary)
    
    # CSV保存
    csv_path = RESULTS_DIR / f'parameter_summary_{model_type}.csv'
    summary.to_csv(csv_path)
    print(f"\n✅ サマリーを保存: {csv_path}")


def main():
    """メイン実行関数"""
    print("="*70)
    print("🔬 パラメータ相関分析スクリプト")
    print("="*70)
    
    # B_formモデルの分析（成功例）
    try:
        print("\n【B_formモデルの分析】")
        trace_b = load_trace('B_form')
        
        # 統計サマリー
        print_summary_statistics(trace_b, 'B_form')
        
        # 相関係数行列
        corr_matrix = calculate_correlation_matrix(trace_b, 'B_form')
        
        # パラメータ相関プロット
        plot_pair_correlation(trace_b, 'B_form')
        
        # Gamma相関プロット
        plot_gamma_parameters(trace_b, 'B_form')
        
        # 自己相関プロット
        plot_autocorrelation(trace_b, 'B_form')
        
        # 事後予測チェック
        plot_posterior_predictive(trace_b, 'B_form')
        
    except Exception as e:
        print(f"❌ B_formモデルの分析でエラー: {e}")
    
    # H_formモデルの分析（比較用）
    try:
        print("\n" + "="*70)
        print("\n【H_formモデルの分析】")
        trace_h = load_trace('H_form')
        
        # 統計サマリー
        print_summary_statistics(trace_h, 'H_form')
        
        # 相関係数行列
        corr_matrix_h = calculate_correlation_matrix(trace_h, 'H_form')
        
        # パラメータ相関プロット
        plot_pair_correlation(trace_h, 'H_form')
        
        # Gamma相関プロット
        plot_gamma_parameters(trace_h, 'H_form')
        
        # 自己相関プロット
        plot_autocorrelation(trace_h, 'H_form')
        
        # 事後予測チェック
        plot_posterior_predictive(trace_h, 'H_form')
        
    except Exception as e:
        print(f"❌ H_formモデルの分析でエラー: {e}")
    
    print("\n" + "="*70)
    print("✅ 全ての相関分析が完了しました")
    print(f"📁 結果は {RESULTS_DIR} に保存されています")
    print("="*70)


if __name__ == "__main__":
    main()
