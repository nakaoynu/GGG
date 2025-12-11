"""保存済みPyMCトレースから透過スペクトルの95%信用区間をエクスポート（統合版）

unified_weighted_bayesian_fitting.pyで生成された保存されたトレースを読み込み、
各(B, T)ペアで磁気応答を再構成し、平均透過率とMAP推定値、95%信用区間の
プロットとCSV要約を保存します。

【主な機能】
- 複数の(B, T)ペアに対応
- 温度変化データ・磁場変化データの両方をサポート
- 事後サンプルから透過率スペクトルの不確実性を計算
- MAP推定値と95%信用区間を可視化
"""

from __future__ import annotations

import argparse
import pathlib
import warnings
import datetime
from typing import Dict, Tuple, Any, Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml

import unified_weighted_bayesian_fitting as uwbf

# 物理定数 (SI単位系)
MU0 = uwbf.mu0
MUB = uwbf.muB
HBAR = uwbf.hbar
KB = uwbf.kB
BASE_TEMPERATURE = 4.0  # 温度依存gamma計算の基準温度
S = uwbf.s # スピン量子数（統合版モジュールから取得）
THZ_TO_RAD_S = uwbf.THZ_TO_RAD_S  # THz → rad/s 変換係数

try:
    import japanize_matplotlib
except ImportError:
    print("警告: japanize_matplotlib が見つかりません。")

plt.rcParams['figure.dpi'] = 120


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description=(
            "unified_weighted_bayesian_fitting.pyで生成された保存済みPyMCトレースから、"
            "透過スペクトルの95%信用区間を計算します。"
        )
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        type=pathlib.Path,
        help="trace_<model>.ncと補助CSVファイルを含む実行ディレクトリへのパス",
    )
    parser.add_argument(
        "--model",
        choices=["H_form", "B_form", "both"],
        default="both",
        help="読み込むモデルトレースを選択 (デフォルト: both, both: 両モデルを比較)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=300,
        help="予測スペクトル用に抽出する事後サンプル数 (デフォルト: 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="事後サンプル選択用の乱数シード (デフォルト: 42)",
    )
    parser.add_argument(
        "--freq-points",
        type=int,
        default=None,
        help=(
            "評価用の周波数点数（オプション）。省略した場合は"
            "元の実験グリッドを使用します。"
        ),
    )
    return parser.parse_args()


def load_runtime_config(results_dir: pathlib.Path) -> Dict:
    """サンプリング時に使用された設定を読み込む"""
    config_path = results_dir / "config_used.yml"
    if config_path.exists():
        config = uwbf.load_config(config_path)
        print(f"✅ 設定ファイルを読み込みました: {config_path}")
    else:
        warnings.warn(
            "config_used.yml が結果ディレクトリに見つかりません。デフォルトの config_unified.yml にフォールバックします。",
            RuntimeWarning,
            stacklevel=1,
        )
        config = uwbf.load_config()
    return config


def load_unified_data_for_export(config: Dict[str, Any]) -> Dict[Tuple[float, float], Dict[str, Any]]:
    """全(B, T)データセットを読み込み、重み配列を生成する"""
    print("\n--- データ読み込み ---")
    
    unified_data = uwbf.load_unified_data(config)
    
    # 全データセットを(B, T)キーの辞書に変換
    datasets = {}
    all_datasets = unified_data['temp_variable'] + unified_data['field_variable']
    
    for dataset in all_datasets:
        B = dataset['b_field']
        T = dataset['temperature']
        bt_key = (B, T)
        
        # 重み配列を生成
        weights = uwbf.create_frequency_weights(dataset, config['analysis_settings'])
        dataset['weights'] = weights
        
        datasets[bt_key] = dataset
        print(f"  読み込み: B={B:.1f}T, T={T:.1f}K (データ点数: {len(dataset['frequency'])})")
    
    print(f"✅ 合計 {len(datasets)} 個の(B, T)データセットを読み込みました")
    return datasets


def load_bt_parameters(results_dir: pathlib.Path, model_type: str) -> Dict[Tuple[float, float], Dict]:
    """(B, T)ペア別の光学パラメータをCSVから読み込む"""
    params_path = results_dir / f"bt_optical_parameters_{model_type}.csv"
    if not params_path.exists():
        raise FileNotFoundError(
            f"{model_type} の(B, T)パラメータが見つかりません: {params_path}"
        )
    
    df = pd.read_csv(params_path)
    required_cols = {"b_field", "temperature", "eps_bg", "d"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{params_path.name} に必須列 {sorted(missing)} がありません"
        )
    
    param_map: Dict[Tuple[float, float], Dict] = {}
    for _, row in df.iterrows():
        b_field = float(row["b_field"])
        temperature = float(row["temperature"])
        bt_key = (b_field, temperature)
        
        param_map[bt_key] = {
            "eps_bg": float(row["eps_bg"]),
            "d": float(row["d"]),
        }
    
    print(f"✅ {len(param_map)} 個の(B, T)パラメータを読み込みました")
    return param_map


def prepare_posterior_samples(
    posterior: xr.Dataset, n_samples: int, seed: int | None
) -> xr.Dataset:
    """事後分布サンプルを準備（計算効率のためサブサンプリング）
    
    Parameters
    ----------
    posterior : xr.Dataset
        PyMCの事後分布 (16,000サンプル = 4000 draws × 4 chains)
    n_samples : int
        抽出するサンプル数 (推奨: 300)
    seed : int or None
        乱数シード
    
    Returns
    -------
    xr.Dataset
        サブサンプリングされた事後分布
    """
    posterior_ds = posterior.stack(sample=("chain", "draw"))
    total_samples = posterior_ds.sizes["sample"]
    n_select = min(n_samples, total_samples) if n_samples else total_samples

    print(f"  事後サンプル: {total_samples}個 → {n_select}個を抽出")

    if n_select < total_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(total_samples, size=n_select, replace=False))
        subset = posterior_ds.isel(sample=indices)
    else:
        subset = posterior_ds
    return subset


def normalize_transmittance(values: np.ndarray) -> np.ndarray:
    """透過率データを0-1に正規化"""
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr)
    arr_finite = arr[finite]
    minimum, maximum = arr_finite.min(), arr_finite.max()
    if maximum > minimum:
        normalized = (arr - minimum) / (maximum - minimum)
    else:
        normalized = np.full_like(arr, 0.5)
    return np.clip(normalized, 0.0, 1.0)


def compute_map_estimates(posterior_subset: xr.Dataset) -> Dict[str, Any]:
    """事後分布からMAP（最大事後確率）推定値を計算
    
    対称な事後分布の場合、平均値をMAP推定値として使用します。
    """
    map_params: Dict[str, Any] = {
        'a_scale': float(posterior_subset['a_scale'].mean().item()),
        'g_factor': float(posterior_subset['g_factor'].mean().item()),
        'B4': float(posterior_subset['B4'].mean().item()),
        'B6': float(posterior_subset['B6'].mean().item()),
        'log_gamma_mu_base': float(posterior_subset['log_gamma_mu_base'].mean().item()),
        'temp_gamma_slope': float(posterior_subset['temp_gamma_slope'].mean().item()),
        'log_gamma_sigma_base': float(posterior_subset['log_gamma_sigma_base'].mean().item()),
        'log_gamma_offset_base': posterior_subset['log_gamma_offset_base'].mean(dim='sample').values,
    }
    return map_params


def calculate_transmission_single(
    freq_thz_array: np.ndarray,
    b_field: float,
    temperature: float,
    eps_bg: float,
    thickness: float,
    a_scale: float,
    g_factor: float,
    param_b4: float,
    param_b6: float,
    gamma_array: np.ndarray,
    model_type: str,
    n_spin: float,
    S: float = S,
) -> np.ndarray:
    """単一のパラメータセットで透過スペクトルを計算（THz単位系対応版）
    
    Parameters
    ----------
    freq_thz_array : np.ndarray
        周波数配列 (THz)
    b_field : float
        磁場 (T)
    temperature : float
        温度 (K)
    eps_bg : float
        背景誘電率
    thickness : float
        試料厚さ (m)
    a_scale : float
        磁気感受率のスケーリング係数
    g_factor : float
        ランデのg因子
    param_b4 : float
        結晶場パラメータ B4 (K)
    param_b6 : float
        結晶場パラメータ B6 (K)
    gamma_array : np.ndarray
        遷移の減衰パラメータ配列 (THz)
    model_type : str
        モデルタイプ ('H_form' または 'B_form')
    n_spin : float
        スピン密度 (m^-3)
    s : float
        スピン量子数 (デフォルト: 3.5)
    
    Returns
    -------
    np.ndarray
        正規化された透過スペクトル
    """
    # ハミルトニアンを計算
    hamiltonian = uwbf.get_hamiltonian(b_field, g_factor, param_b4, param_b6)
    
    # 磁気感受率を計算（THz単位系）
    chi_raw = uwbf.calculate_susceptibility(freq_thz_array, hamiltonian, temperature, gamma_array)
    # 【修正】THz単位系での次元合わせ: chi_rawは1/THz次元 = THZ_TO_RAD_S/(rad/s)
    # 旧版のchi_rawは1/(rad/s)次元なので、chi_raw_new = chi_raw_old * THZ_TO_RAD_S
    # chi = G0 * chi_raw を一致させるには G0_new = G0_old / THZ_TO_RAD_S
    g0 = a_scale * MU0 * n_spin * (g_factor * MUB) ** 2 / (2 * HBAR) / THZ_TO_RAD_S
    chi = g0 * chi_raw

    # モデルに応じて比透磁率を計算
    if model_type == "B_form":
        mu_r = 1.0 / (1.0 - chi)
    else:  # H_form
        mu_r = 1.0 + chi

    # 透過率を計算（THz単位系）
    trans = uwbf.calculate_normalized_transmission(freq_thz_array, mu_r, thickness, eps_bg)
    return np.clip(np.real_if_close(trans), 0.0, 1.0)


def simulate_predictions(
    dataset: Dict[str, Any],
    b_field: float,
    temperature: float,
    params: Dict[str, float],
    posterior_subset: xr.Dataset,
    model_type: str,
    n_spin: float,
    s: float,
    freq_points: int | None,
) -> Dict[str, np.ndarray]:
    """事後サンプルから透過率スペクトルの予測を計算
    
    Parameters
    ----------
    dataset : Dict
        実験データセット
    b_field : float
        磁場 (T)
    temperature : float
        温度 (K)
    params : Dict
        eps_bg, d などの(B, T)固有パラメータ
    posterior_subset : xr.Dataset
        事後サンプルのサブセット
    model_type : str
        モデルタイプ
    n_spin : float
        スピン密度
    s : float
        スピン量子数
    freq_points : int or None
        評価する周波数点数
    
    Returns
    -------
    Dict[str, np.ndarray]
        周波数、平均、MAP、信用区間、実験データを含む辞書
    """
    freq_exp = np.asarray(dataset["frequency"], dtype=float)
    if freq_points and freq_points > 0:
        freq_eval = np.linspace(freq_exp.min(), freq_exp.max(), freq_points)
    else:
        freq_eval = freq_exp

    eps_bg = params["eps_bg"]
    thickness = params["d"]

    # 事後サンプルからパラメータを取得
    a_scale = np.asarray(posterior_subset["a_scale"].values, dtype=float)
    g_factor = np.asarray(posterior_subset["g_factor"].values, dtype=float)
    param_b4 = np.asarray(posterior_subset["B4"].values, dtype=float)
    param_b6 = np.asarray(posterior_subset["B6"].values, dtype=float)
    log_gamma_mu = np.asarray(posterior_subset["log_gamma_mu_base"].values, dtype=float)
    temp_gamma_slope = np.asarray(posterior_subset["temp_gamma_slope"].values, dtype=float)
    log_gamma_sigma = np.asarray(posterior_subset["log_gamma_sigma_base"].values, dtype=float)
    log_gamma_offset = np.asarray(posterior_subset["log_gamma_offset_base"].values, dtype=float)

    n_draws = a_scale.shape[0]
    predictions = np.zeros((n_draws, freq_eval.size), dtype=float)

    # 温度依存gammaを計算
    temp_diff = temperature - BASE_TEMPERATURE
    log_gamma_mu_temp = log_gamma_mu + temp_gamma_slope * temp_diff
    
    # log_gamma_offsetの形状を確認して適切にブロードキャスト
    if log_gamma_offset.ndim == 1:
        # 1次元の場合: (7,) → (n_draws, 7)
        gamma_samples = np.exp(
            log_gamma_mu_temp[:, None] + log_gamma_offset[None, :] * log_gamma_sigma[:, None]
        )
    elif log_gamma_offset.shape[0] == 7:
        # 形状が (7, n_draws) の場合 → 転置して (n_draws, 7)
        log_gamma_offset_T = log_gamma_offset.T
        gamma_samples = np.exp(
            log_gamma_mu_temp[:, None] + log_gamma_offset_T * log_gamma_sigma[:, None]
        )
    else:
        # 形状が (n_draws, 7) の場合
        gamma_samples = np.exp(
            log_gamma_mu_temp[:, None] + log_gamma_offset * log_gamma_sigma[:, None]
        )

    # 全事後サンプルで透過率を計算（THz単位系）
    print(f"    事後サンプル {n_draws}個で透過率を計算中...", end="", flush=True)
    
    # デバッグ: 最初のサンプルのパラメータ値を表示
    if n_draws > 0:
        print(f"\n    [デバッグ] パラメータ例 (sample 0):")
        print(f"      a_scale={a_scale[0]:.4f}, g_factor={g_factor[0]:.4f}")
        print(f"      B4={param_b4[0]:.6f} K, B6={param_b6[0]:.8f} K")
        print(f"      gamma (THz): {gamma_samples[0][:3]}... (最初の3つ)")
        print(f"      eps_bg={eps_bg:.4f}, thickness={thickness*1e6:.2f} um")
        print(f"    ", end="", flush=True)
    
    for idx in range(n_draws):
        predictions[idx] = calculate_transmission_single(
            freq_eval, b_field, temperature, eps_bg, thickness,
            a_scale[idx], g_factor[idx], param_b4[idx], param_b6[idx],
            gamma_samples[idx], model_type, n_spin, s
        )
    print(" 完了")
    
    # デバッグ: 予測値の統計を表示
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    pred_all_same = np.allclose(predictions, 0.5, atol=0.01)
    if pred_all_same:
        print(f"    [警告] 予測が全て0.5付近: 物理モデル計算に問題がある可能性")
    else:
        print(f"    [デバッグ] 予測統計: mean={pred_mean:.4f}, std={pred_std:.4f}, range=[{predictions.min():.4f}, {predictions.max():.4f}]")

    # 統計量を計算
    mean_pred = predictions.mean(axis=0)
    lower, upper = np.percentile(predictions, [2.5, 97.5], axis=0)

    # MAP（最大事後確率）予測を計算（THz単位系）
    map_params = compute_map_estimates(posterior_subset)
    temp_diff_map = temperature - BASE_TEMPERATURE
    log_gamma_mu_temp_map = map_params['log_gamma_mu_base'] + map_params['temp_gamma_slope'] * temp_diff_map
    gamma_map = np.exp(
        log_gamma_mu_temp_map + map_params['log_gamma_offset_base'] * map_params['log_gamma_sigma_base']
    )
    
    map_pred = calculate_transmission_single(
        freq_eval, b_field, temperature, eps_bg, thickness,
        map_params['a_scale'], map_params['g_factor'], map_params['B4'], map_params['B6'],
        gamma_map, model_type, n_spin, s
    )

    # 重み情報も返す
    weights = dataset.get("weights", np.ones_like(freq_eval))
    
    return {
        "frequency_thz": freq_eval,
        "mean": mean_pred,
        "map": map_pred,
        "lower": lower,
        "upper": upper,
        "observed": normalize_transmittance(dataset.get("transmittance_full", freq_eval * 0.0)),
        "weights": weights,
    }


def save_outputs(
    output_dir: pathlib.Path,
    model_type: str,
    b_field: float,
    temperature: float,
    summary: Dict[str, np.ndarray],
) -> None:
    """結果をCSVとPNGで保存"""
    freq = summary["frequency_thz"]
    df = pd.DataFrame(
        {
            "frequency_thz": freq,
            "mean_transmission": summary["mean"],
            "map_transmission": summary["map"],
            "lower_95": summary["lower"],
            "upper_95": summary["upper"],
            "observed_normalized": summary["observed"],
        }
    )
    csv_path = output_dir / f"transmission_ci_{model_type}_B{b_field:.1f}T_T{int(temperature)}K.csv"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(freq, summary["lower"], summary["upper"], 
                     color="tab:blue", alpha=0.3, label="95%信用区間")
    ax.plot(freq, summary["mean"], color="tab:blue", linewidth=2.0, 
            label="事後平均", linestyle='--')
    ax.plot(freq, summary["map"], color="tab:red", linewidth=2.5, 
            label="MAP推定値")
    
    # 重み情報がある場合、重みに応じて色分けして表示
    weights = summary.get("weights", np.ones_like(freq))
    mask_weight_1 = np.abs(weights - 1.0) < 1e-6
    mask_weight_mid = np.abs(weights - 0.1) < 1e-6
    mask_weight_other = ~(mask_weight_1 | mask_weight_mid)
    
    if np.any(mask_weight_other):
        ax.scatter(freq[mask_weight_other], summary["observed"][mask_weight_other], 
                  color="lightgray", s=12, alpha=0.5, label="実験データ(背景)", zorder=3)
    if np.any(mask_weight_mid):
        ax.scatter(freq[mask_weight_mid], summary["observed"][mask_weight_mid], 
                  color="orange", s=16, alpha=0.7, label="実験データ(LP-UP間)", zorder=4)
    if np.any(mask_weight_1):
        ax.scatter(freq[mask_weight_1], summary["observed"][mask_weight_1], 
                  color="black", s=20, alpha=0.8, label="実験データ(ピーク)", zorder=5)
    
    ax.set_xlabel("周波数 (THz)")
    ax.set_ylabel("正規化透過率")
    ax.set_title(f"{model_type} 透過スペクトル @ B={b_field:.1f}T, T={temperature:.0f}K")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    png_path = output_dir / f"transmission_ci_{model_type}_B{b_field:.1f}T_T{int(temperature)}K.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def save_comparison_outputs(
    output_dir: pathlib.Path,
    b_field: float,
    temperature: float,
    summary_h: Dict[str, np.ndarray],
    summary_b: Dict[str, np.ndarray],
    config: Dict[str, Any],
) -> None:
    """H_formとB_formの比較結果をCSVとPNGで保存"""
    freq = summary_h["frequency_thz"]
    
    # 比較データをCSVに保存
    df = pd.DataFrame(
        {
            "frequency_thz": freq,
            "H_form_mean": summary_h["mean"],
            "H_form_map": summary_h["map"],
            "H_form_lower_95": summary_h["lower"],
            "H_form_upper_95": summary_h["upper"],
            "B_form_mean": summary_b["mean"],
            "B_form_map": summary_b["map"],
            "B_form_lower_95": summary_b["lower"],
            "B_form_upper_95": summary_b["upper"],
            "observed_normalized": summary_h["observed"],
        }
    )
    csv_path = output_dir / f"transmission_comparison_B{b_field:.1f}T_T{int(temperature)}K.csv"
    df.to_csv(csv_path, index=False)

    # 比較プロットを作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 上段: 両モデルのMAP推定値と実験データ
    ax1.fill_between(freq, summary_h["lower"], summary_h["upper"], 
                     color="tab:red", alpha=0.2, label="H_form 95%信用区間")
    ax1.fill_between(freq, summary_b["lower"], summary_b["upper"], 
                     color="tab:blue", alpha=0.2, label="B_form 95%信用区間")
    ax1.plot(freq, summary_h["map"], color="tab:red", linewidth=2.5, 
             label="H_form MAP推定値", linestyle='-')
    ax1.plot(freq, summary_b["map"], color="tab:blue", linewidth=2.5, 
             label="B_form MAP推定値", linestyle='-')
    
    # 重み情報がある場合、重みに応じて色分けして表示
    weight_settings = config['analysis_settings'].get('weight_settings', {})
    w_high = float(weight_settings['lp_up_peak_weight'])
    w_mid = float(weight_settings['between_peaks_weight'])
    
    weights = summary_h.get("weights", np.ones_like(freq))
    mask_weight_1 = np.abs(weights - w_high) < 1e-6
    mask_weight_mid = np.abs(weights - w_mid) < 1e-6
    mask_weight_other = ~(mask_weight_1 | mask_weight_mid)
    
    if np.any(mask_weight_other):
        ax1.scatter(freq[mask_weight_other], summary_h["observed"][mask_weight_other], 
                   color="lightgray", s=12, alpha=0.5, label="実験データ(背景)", zorder=3)
    if np.any(mask_weight_mid):
        ax1.scatter(freq[mask_weight_mid], summary_h["observed"][mask_weight_mid], 
                   color="orange", s=16, alpha=0.7, label=f"実験データ(LP-UP間 w={w_mid})", zorder=4)
    if np.any(mask_weight_1):
        ax1.scatter(freq[mask_weight_1], summary_h["observed"][mask_weight_1], 
                   color="black", s=20, alpha=0.8, label=f"実験データ(ピーク w={w_high})", zorder=5)
    
    ax1.set_ylabel("正規化透過率")
    ax1.set_title(f"モデル比較: 透過スペクトル @ B={b_field:.1f}T, T={temperature:.0f}K")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_ylim(0.0, 1.05)
    ax1.legend(loc='best', fontsize=9)
    
    # 下段: 両モデルの差分（H_form - B_form）
    diff_map = summary_h["map"] - summary_b["map"]
    diff_mean = summary_h["mean"] - summary_b["mean"]
    ax2.plot(freq, diff_map, color="tab:green", linewidth=2.0, 
             label="MAP差分 (H_form - B_form)")
    ax2.plot(freq, diff_mean, color="tab:orange", linewidth=1.5, 
             linestyle='--', label="平均差分 (H_form - B_form)")
    ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel("周波数 (THz)")
    ax2.set_ylabel("透過率差分")
    ax2.set_title("モデル間差分")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc='best', fontsize=9)
    
    png_path = output_dir / f"transmission_comparison_B{b_field:.1f}T_T{int(temperature)}K.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"    比較プロットを保存: {png_path.name}")


def main() -> None:
    """メイン実行関数"""
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"結果ディレクトリが存在しません: {results_dir}")

    print("="*70)
    print("統合版: 透過スペクトル95%信用区間エクスポート")
    print("="*70)
    print(f"結果ディレクトリ: {results_dir}")
    print(f"モデル: {args.model}")
    print(f"サンプル数: {args.samples}")

    # 設定ファイルを読み込み
    config = load_runtime_config(results_dir)

    # データセットを読み込み
    datasets = load_unified_data_for_export(config)
    
    # 物理パラメータを取得
    n_spin = config['physical_parameters']['N_spin']
    s = config['physical_parameters']['s']
    
    # 両モデル比較モードの処理
    if args.model == "both":
        print("\n=== 両モデル比較モード ===")
        
        # 両モデルのトレースとパラメータを読み込み
        models_data = {}
        for model_type in ["H_form", "B_form"]:
            trace_path = results_dir / f"trace_{model_type}.nc"
            if not trace_path.exists():
                raise FileNotFoundError(f"トレースファイルが見つかりません: {trace_path}")
            
            print(f"\n📂 {model_type} モデルを読み込み中...")
            trace = az.from_netcdf(trace_path)
            posterior_subset = prepare_posterior_samples(trace.posterior, args.samples, args.seed)  # type: ignore[attr-defined]
            bt_params = load_bt_parameters(results_dir, model_type)
            
            models_data[model_type] = {
                'posterior_subset': posterior_subset,
                'bt_params': bt_params,
            }
        
        output_dir = results_dir / "transmission_intervals_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 出力先: {output_dir}")
        
        missing_bt = []
        for bt_key in sorted(datasets.keys()):
            B, T = bt_key
            
            # 両モデルでパラメータが存在するかチェック
            if (bt_key not in models_data["H_form"]['bt_params'] or
                bt_key not in models_data["B_form"]['bt_params']):
                missing_bt.append(bt_key)
                continue
            
            print(f"\n  処理中: B={B:.1f}T, T={T:.1f}K")
            
            # H_formの予測を計算
            summary_h = simulate_predictions(
                dataset=datasets[bt_key],
                b_field=B,
                temperature=T,
                params=models_data["H_form"]['bt_params'][bt_key],
                posterior_subset=models_data["H_form"]['posterior_subset'],
                model_type="H_form",
                n_spin=n_spin,
                s=s,
                freq_points=args.freq_points,
            )
            
            # B_formの予測を計算
            summary_b = simulate_predictions(
                dataset=datasets[bt_key],
                b_field=B,
                temperature=T,
                params=models_data["B_form"]['bt_params'][bt_key],
                posterior_subset=models_data["B_form"]['posterior_subset'],
                model_type="B_form",
                n_spin=n_spin,
                s=s,
                freq_points=args.freq_points,
            )
            
            # 比較プロットを保存
            save_comparison_outputs(output_dir, B, T, summary_h, summary_b, config)
            
            # 個別モデルの結果も保存
            save_outputs(output_dir, "H_form", B, T, summary_h)
            save_outputs(output_dir, "B_form", B, T, summary_b)
            
            print(f"    ✅ B={B:.1f}T, T={T:.1f}K の比較結果を保存しました")
        
        if missing_bt:
            warnings.warn(
                "以下の(B, T)ペアのパラメータが見つかりませんでした: "
                + ", ".join(f"({B:.1f}T, {T:.0f}K)" for B, T in missing_bt),
                RuntimeWarning,
                stacklevel=1,
            )
        
        print(f"\n🎉 全ての比較出力を保存しました: {output_dir}")
    
    # 単一モデルモードの処理
    else:
        print(f"\n=== {args.model} モデル処理モード ===")
        
        # トレースとパラメータを読み込み
        trace_path = results_dir / f"trace_{args.model}.nc"
        if not trace_path.exists():
            raise FileNotFoundError(f"トレースファイルが見つかりません: {trace_path}")
        
        print(f"\n📂 トレースファイルを読み込み中...")
        trace = az.from_netcdf(trace_path)
        posterior_subset = prepare_posterior_samples(trace.posterior, args.samples, args.seed)  # type: ignore[attr-defined]
        bt_params = load_bt_parameters(results_dir, args.model)

        output_dir = results_dir / f"transmission_intervals_{args.model}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 出力先: {output_dir}")

        missing_bt = []
        for bt_key in sorted(datasets.keys()):
            B, T = bt_key
            
            if bt_key not in bt_params:
                missing_bt.append(bt_key)
                continue
            
            print(f"\n  処理中: B={B:.1f}T, T={T:.1f}K")
            
            summary = simulate_predictions(
                dataset=datasets[bt_key],
                b_field=B,
                temperature=T,
                params=bt_params[bt_key],
                posterior_subset=posterior_subset,
                model_type=args.model,
                n_spin=n_spin,
                s=s,
                freq_points=args.freq_points,
            )
            save_outputs(output_dir, args.model, B, T, summary)
            print(f"    ✅ B={B:.1f}T, T={T:.1f}K の透過スペクトル信用区間を保存しました")

        if missing_bt:
            warnings.warn(
                "以下の(B, T)ペアのパラメータが見つかりませんでした: "
                + ", ".join(f"({B:.1f}T, {T:.0f}K)" for B, T in missing_bt),
                RuntimeWarning,
                stacklevel=1,
            )

        print(f"\n🎉 全ての出力を保存しました: {output_dir}")


if __name__ == "__main__":
    main()
