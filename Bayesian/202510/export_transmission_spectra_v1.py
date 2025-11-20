"""保存済みPyMCトレースから透過スペクトルの95%信用区間をエクスポート (v1用)
weighted_bayesian_fitting_v1.pyで生成されたトレースを読み込み、各温度で磁気応答を
再構成し、平均透過率とMAP推定値、95%信用区間のプロットとCSV要約を保存します。

v1との主な違い:
- 2パラメータgammaモデル (log_gamma_min, log_gamma_other)
- 温度依存性のないgamma配列
"""

from __future__ import annotations

import argparse
import pathlib
import warnings
from typing import Dict, Iterable, Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import weighted_bayesian_fitting_v1 as wbf

# 物理定数 (SI単位系)
MU0 = wbf.mu0
MUB = wbf.muB
HBAR = wbf.hbar


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description=(
            "weighted_bayesian_fitting_v1.pyで生成された保存済みPyMCトレースから、"
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
        default="H_form",
        help="読み込むモデルトレースを選択 (デフォルト: H_form, both: 両モデルを比較)",
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
    """サンプリング時に使用された設定を読み込む

    Parameters
    ----------
    results_dir: pathlib.Path
        ``config_used.yml`` を含むディレクトリ

    Returns
    -------
    dict
        解決済み設定辞書
    """
    config_path = results_dir / "config_used.yml"
    if config_path.exists():
        config = wbf.load_config(config_path)
    else:
        warnings.warn(
            "config_used.yml が結果ディレクトリに見つかりません。デフォルトの config_v1.yml にフォールバックします。",
            RuntimeWarning,
            stacklevel=1,
        )
        config = wbf.load_config(pathlib.Path(__file__).parent / "config_v1.yml")
    return config


def update_module_state(config: Dict) -> None:
    """インポートしたモジュールの状態を読み込んだ設定で同期"""
    wbf.CONFIG = config
    wbf.TEMPERATURE_COLUMNS = config["analysis_settings"]["temperature_columns"]
    wbf.LOW_FREQUENCY_CUTOFF = config["analysis_settings"]["low_freq_cutoff"]
    wbf.HIGH_FREQUENCY_CUTOFF = config["analysis_settings"]["high_freq_cutoff"]
    wbf.DATA_FILE_PATH = config["file_paths"]["data_file"]
    wbf.DATA_SHEET_NAME = config["file_paths"]["sheet_name"]
    wbf.s = config["physical_parameters"]["s"]
    wbf.N_spin = config["physical_parameters"]["N_spin"]
    wbf.B_FIXED = config["physical_parameters"]["B_fixed"]
    wbf.d_fixed = config["physical_parameters"]["d_fixed"]


def load_full_datasets(config: Dict) -> Dict[float, Dict]:
    """全温度データセットを読み込み、重み配列を生成する"""
    data_bundle = wbf.load_and_split_data_three_regions_temperature(
        file_path=config["file_paths"]["data_file"],
        sheet_name=config["file_paths"]["sheet_name"],
        low_cutoff=config["analysis_settings"]["low_freq_cutoff"],
        high_cutoff=config["analysis_settings"]["high_freq_cutoff"],
    )
    datasets = {}
    for dataset in data_bundle["all_full"]:
        temp = float(dataset["temperature"])
        # 重み配列を生成して追加
        weights = wbf.create_frequency_weights(dataset, config["analysis_settings"])
        dataset["weights"] = weights
        datasets[temp] = dataset
    return datasets


def load_temperature_parameters(results_dir: pathlib.Path, model_type: str) -> Dict[float, Dict]:
    """温度別の光学パラメータをCSVから読み込む"""
    params_path = results_dir / f"temperature_optical_parameters_{model_type}.csv"
    if not params_path.exists():
        raise FileNotFoundError(
            f"{model_type} の温度パラメータが見つかりません: {params_path}"
        )
    df = pd.read_csv(params_path)
    required_cols = {"temperature", "eps_bg", "d"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{params_path.name} に必須列 {sorted(missing)} がありません"
        )
    param_map: Dict[float, Dict] = {}
    for _, row in df.iterrows():
        temp_val = float(row["temperature"])  # type: ignore[arg-type]
        eps_bg_val = float(row["eps_bg"])  # type: ignore[arg-type]
        d_val = float(row["d"])  # type: ignore[arg-type]
            
        param_map[temp_val] = {
            "eps_bg": eps_bg_val,
            "d": d_val,
        }
    return param_map


def prepare_posterior_samples(
    posterior: Any, n_samples: int, seed: int | None
) -> xr.Dataset:
    """事後分布サンプルを準備（計算効率のためサブサンプリング）"""
    posterior_ds = posterior.stack(sample=("chain", "draw"))
    total_samples = posterior_ds.sizes["sample"]
    n_select = min(n_samples, total_samples) if n_samples else total_samples

    if n_select < total_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(total_samples, size=n_select, replace=False))
        subset = posterior_ds.isel(sample=indices)
    else:
        subset = posterior_ds
    return subset


def normalize_transmittance(values: Iterable[float]) -> np.ndarray:
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
    
    v1では2パラメータgammaモデルを使用
    
    Parameters
    ----------
    posterior_subset : xr.Dataset
        事後サンプルのサブセット
    
    Returns
    -------
    dict
        全パラメータのMAP推定値を含む辞書
    """
    map_params: Dict[str, Any] = {
        'a_scale': float(posterior_subset['a_scale'].mean().item()),
        'g_factor': float(posterior_subset['g_factor'].mean().item()),
        'B4': float(posterior_subset['B4'].mean().item()),
        'B6': float(posterior_subset['B6'].mean().item()),
        'log_gamma_min': float(posterior_subset['log_gamma_min'].mean().item()),
        'log_gamma_other': float(posterior_subset['log_gamma_other'].mean().item()),
    }
    return map_params


def construct_gamma_array_v1(log_gamma_min: float, log_gamma_other: float) -> np.ndarray:
    """v1の2パラメータgammaモデルからgamma配列を構築
    
    gamma配列は [gamma_min, gamma_other, gamma_other, ...]の7要素
    
    Parameters
    ----------
    log_gamma_min : float
        最低次遷移の対数減衰パラメータ
    log_gamma_other : float
        その他の遷移の対数減衰パラメータ
    
    Returns
    -------
    np.ndarray
        7要素のgamma配列
    """
    gamma_min = np.exp(log_gamma_min)
    gamma_other = np.exp(log_gamma_other)
    return np.array([gamma_min] + [gamma_other] * 6)


def calculate_transmission_single(
    omega_eval: np.ndarray,
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
) -> np.ndarray:
    """単一のパラメータセットで透過スペクトルを計算
    
    Parameters
    ----------
    omega_eval : np.ndarray
        角周波数配列 (rad/s)
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
        遷移の減衰パラメータ配列
    model_type : str
        モデルタイプ ('H_form' または 'B_form')
    n_spin : float
        スピン密度 (m^-3)
    
    Returns
    -------
    np.ndarray
        正規化された透過スペクトル
    """
    hamiltonian = wbf.get_hamiltonian(wbf.B_FIXED, g_factor, param_b4, param_b6)
    chi_raw = wbf.calculate_susceptibility(omega_eval, hamiltonian, temperature, gamma_array)
    g0 = a_scale * MU0 * n_spin * (g_factor * MUB) ** 2 / (2 * HBAR)
    chi = g0 * chi_raw

    if model_type == "B_form":
        mu_r = 1.0 / (1.0 - chi)
    else:
        mu_r = 1.0 + chi

    trans = wbf.calculate_normalized_transmission(omega_eval, mu_r, thickness, eps_bg)
    return np.clip(np.real_if_close(trans), 0.0, 1.0)


def simulate_predictions(
    dataset: Dict,
    params: Dict[str, float],
    posterior_subset: xr.Dataset,
    model_type: str,
    n_spin: float,
    freq_points: int | None,
) -> Dict[str, np.ndarray]:
    """事後サンプルから透過スペクトルの予測分布を計算"""
    freq_exp = np.asarray(dataset["frequency"], dtype=float)
    if freq_points and freq_points > 0:
        freq_eval = np.linspace(freq_exp.min(), freq_exp.max(), freq_points)
    else:
        freq_eval = freq_exp
    omega_eval = freq_eval * 1e12 * 2 * np.pi

    temperature = float(dataset["temperature"])
    eps_bg = params["eps_bg"]
    thickness = params["d"]

    # 事後サンプルを抽出
    a_scale = np.asarray(posterior_subset["a_scale"].values, dtype=float)
    g_factor = np.asarray(posterior_subset["g_factor"].values, dtype=float)
    param_b4 = np.asarray(posterior_subset["B4"].values, dtype=float)
    param_b6 = np.asarray(posterior_subset["B6"].values, dtype=float)
    log_gamma_min = np.asarray(posterior_subset["log_gamma_min"].values, dtype=float)
    log_gamma_other = np.asarray(posterior_subset["log_gamma_other"].values, dtype=float)

    n_draws = a_scale.shape[0]
    predictions = np.zeros((n_draws, freq_eval.size), dtype=float)

    # 各事後サンプルでgamma配列を構築して予測を計算
    for idx in range(n_draws):
        gamma_array = construct_gamma_array_v1(log_gamma_min[idx], log_gamma_other[idx])
        predictions[idx] = calculate_transmission_single(
            omega_eval, temperature, eps_bg, thickness,
            a_scale[idx], g_factor[idx], param_b4[idx], param_b6[idx],
            gamma_array, model_type, n_spin
        )

    mean_pred = predictions.mean(axis=0)
    lower, upper = np.percentile(predictions, [2.5, 97.5], axis=0)

    # MAP（最大事後確率）予測を計算
    map_params = compute_map_estimates(posterior_subset)
    gamma_map = construct_gamma_array_v1(map_params['log_gamma_min'], map_params['log_gamma_other'])
    
    map_pred = calculate_transmission_single(
        omega_eval, temperature, eps_bg, thickness,
        map_params['a_scale'], map_params['g_factor'], map_params['B4'], map_params['B6'],
        gamma_map, model_type, n_spin
    )

    # 重み情報も返す(存在する場合)
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
    csv_path = output_dir / f"transmission_ci_{model_type}_{int(temperature)}K.csv"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(freq, summary["lower"], summary["upper"], color="tab:blue", alpha=0.3, label="95%信用区間")
    ax.plot(freq, summary["mean"], color="tab:blue", linewidth=2.0, label="事後平均", linestyle='--')
    ax.plot(freq, summary["map"], color="tab:red", linewidth=2.5, label="MAP推定値")
    
    # 重み情報がある場合、重みに応じて色分けして表示
    weights = summary.get("weights", np.ones_like(freq))
    mask_weight_1 = np.abs(weights - 1.0) < 1e-6  # 重み=1.0の点
    mask_weight_06 = np.abs(weights - 0.6) < 1e-6  # 重み=0.6の点
    mask_weight_other = ~(mask_weight_1 | mask_weight_06)  # その他の重みの点
    
    if np.any(mask_weight_other):
        ax.scatter(freq[mask_weight_other], summary["observed"][mask_weight_other], 
                  color="lightgray", s=12, alpha=0.5, label="実験データ(背景)", zorder=3)
    if np.any(mask_weight_06):
        ax.scatter(freq[mask_weight_06], summary["observed"][mask_weight_06], 
                  color="orange", s=16, alpha=0.7, label="実験データ(重み=0.6)", zorder=4)
    if np.any(mask_weight_1):
        ax.scatter(freq[mask_weight_1], summary["observed"][mask_weight_1], 
                  color="black", s=20, alpha=0.8, label="実験データ(重み=1.0)", zorder=5)
    
    ax.set_xlabel("周波数 (THz)")
    ax.set_ylabel("正規化透過率")
    ax.set_title(f"{model_type} 透過スペクトル @ {temperature:.0f} K (v1: 2パラメータgamma)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    png_path = output_dir / f"transmission_ci_{model_type}_{int(temperature)}K.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def save_comparison_outputs(
    output_dir: pathlib.Path,
    temperature: float,
    summary_h: Dict[str, np.ndarray],
    summary_b: Dict[str, np.ndarray],
) -> None:
    """H_formとB_formの比較結果をCSVとPNGで保存
    
    Parameters
    ----------
    output_dir : pathlib.Path
        出力ディレクトリ
    temperature : float
        温度 (K)
    summary_h : Dict[str, np.ndarray]
        H_formモデルの予測サマリー
    summary_b : Dict[str, np.ndarray]
        B_formモデルの予測サマリー
    """
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
    csv_path = output_dir / f"transmission_comparison_{int(temperature)}K.csv"
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
    weights = summary_h.get("weights", np.ones_like(freq))
    mask_weight_1 = np.abs(weights - 1.0) < 1e-6  # 重み=1.0の点
    mask_weight_06 = np.abs(weights - 0.6) < 1e-6  # 重み=0.6の点
    mask_weight_other = ~(mask_weight_1 | mask_weight_06)  # その他の重みの点
    
    if np.any(mask_weight_other):
        ax1.scatter(freq[mask_weight_other], summary_h["observed"][mask_weight_other], 
                   color="lightgray", s=12, alpha=0.5, label="実験データ(背景)", zorder=3)
    if np.any(mask_weight_06):
        ax1.scatter(freq[mask_weight_06], summary_h["observed"][mask_weight_06], 
                   color="orange", s=16, alpha=0.7, label="実験データ(重み=0.6)", zorder=4)
    if np.any(mask_weight_1):
        ax1.scatter(freq[mask_weight_1], summary_h["observed"][mask_weight_1], 
                   color="black", s=20, alpha=0.8, label="実験データ(重み=1.0)", zorder=5)
    ax1.set_ylabel("正規化透過率")
    ax1.set_title(f"モデル比較: 透過スペクトル @ {temperature:.0f} K (v1: 2パラメータgamma)")
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
    
    png_path = output_dir / f"transmission_comparison_{int(temperature)}K.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"  比較プロットを保存: {png_path.name}")


def main() -> None:
    """メイン実行関数"""
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"結果ディレクトリが存在しません: {results_dir}")

    config = load_runtime_config(results_dir)
    update_module_state(config)

    datasets = load_full_datasets(config)
    
    # 両モデル比較モードの処理
    if args.model == "both":
        print("=== 両モデル比較モード (v1: 2パラメータgamma) ===")
        
        # 両モデルのトレースとパラメータを読み込み
        models_data = {}
        for model_type in ["H_form", "B_form"]:
            trace_path = results_dir / f"trace_{model_type}.nc"
            if not trace_path.exists():
                raise FileNotFoundError(f"トレースファイルが見つかりません: {trace_path}")
            
            trace = az.from_netcdf(trace_path)
            posterior_subset = prepare_posterior_samples(trace.posterior, args.samples, args.seed)  # type: ignore[attr-defined]
            temperature_params = load_temperature_parameters(results_dir, model_type)
            
            models_data[model_type] = {
                'posterior_subset': posterior_subset,
                'temperature_params': temperature_params,
            }
            print(f"  {model_type} モデルを読み込みました")
        
        output_dir = results_dir / "transmission_intervals_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        missing_temps = []
        for temperature in sorted(datasets.keys()):
            # 両モデルでパラメータが存在するかチェック
            if (temperature not in models_data["H_form"]['temperature_params'] or
                temperature not in models_data["B_form"]['temperature_params']):
                missing_temps.append(temperature)
                continue
            
            # H_formの予測を計算
            summary_h = simulate_predictions(
                dataset=datasets[temperature],
                params=models_data["H_form"]['temperature_params'][temperature],
                posterior_subset=models_data["H_form"]['posterior_subset'],
                model_type="H_form",
                n_spin=config["physical_parameters"]["N_spin"],
                freq_points=args.freq_points,
            )
            
            # B_formの予測を計算
            summary_b = simulate_predictions(
                dataset=datasets[temperature],
                params=models_data["B_form"]['temperature_params'][temperature],
                posterior_subset=models_data["B_form"]['posterior_subset'],
                model_type="B_form",
                n_spin=config["physical_parameters"]["N_spin"],
                freq_points=args.freq_points,
            )
            
            # 比較プロットを保存
            save_comparison_outputs(output_dir, temperature, summary_h, summary_b)
            
            # 個別モデルの結果も保存
            save_outputs(output_dir, "H_form", temperature, summary_h)
            save_outputs(output_dir, "B_form", temperature, summary_b)
            
            print(f"{temperature:.0f} K の比較結果を保存しました")
        
        if missing_temps:
            warnings.warn(
                "以下の温度のパラメータが見つかりませんでした: "
                + ", ".join(f"{temp:.0f}K" for temp in missing_temps),
                RuntimeWarning,
                stacklevel=1,
            )
        
        print(f"\n全ての比較出力を保存しました: {output_dir}")
    
    # 単一モデルモードの処理
    else:
        temperature_params = load_temperature_parameters(results_dir, args.model)

        trace_path = results_dir / f"trace_{args.model}.nc"
        if not trace_path.exists():
            raise FileNotFoundError(f"トレースファイルが見つかりません: {trace_path}")
        trace = az.from_netcdf(trace_path)
        posterior_subset = prepare_posterior_samples(trace.posterior, args.samples, args.seed)  # type: ignore[attr-defined]

        output_dir = results_dir / f"transmission_intervals_{args.model}"
        output_dir.mkdir(parents=True, exist_ok=True)

        missing_temps = []
        for temperature in sorted(datasets.keys()):
            if temperature not in temperature_params:
                missing_temps.append(temperature)
                continue
            summary = simulate_predictions(
                dataset=datasets[temperature],
                params=temperature_params[temperature],
                posterior_subset=posterior_subset,
                model_type=args.model,
                n_spin=config["physical_parameters"]["N_spin"],
                freq_points=args.freq_points,
            )
            save_outputs(output_dir, args.model, temperature, summary)
            print(
                f"{temperature:.0f} K の透過スペクトル信用区間を保存しました -> {output_dir}"
            )

        if missing_temps:
            warnings.warn(
                "以下の温度のパラメータが見つかりませんでした: "
                + ", ".join(f"{temp:.0f}K" for temp in missing_temps),
                RuntimeWarning,
                stacklevel=1,
            )

        print(f"全ての出力を保存しました: {output_dir}")


if __name__ == "__main__":
    main()
