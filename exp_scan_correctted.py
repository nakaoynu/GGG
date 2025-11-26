import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
import os

# --- プロット設定 ---
try:
    import japanize_matplotlib
except ImportError:
    pass  # japanize_matplotlibがなくても動作するように

plt.rcParams['font.family'] = "Meiryo" # 必要に応じてフォントを変更してください
plt.rcParams['figure.dpi'] = 100

def analyze_transmittance(df, freq_col, transmittance_col):
    """
    指定された透過率データに対して、正規化、ピーク検出、背景補正(2次多項式)、FWHM計算、膜厚計算、プロットを行う。
    """
    print(f"--- {transmittance_col} の分析を開始 ---")

    # データが存在しない、またはNaNが含まれる行を除外するなどのクリーニング
    df_clean = df[[freq_col, transmittance_col]].dropna()
    if df_clean.empty:
        print(f"警告: {transmittance_col} のデータが空です。スキップします。")
        return None, None, None

    exp_freq_thz = df_clean[freq_col].to_numpy(dtype=float)
    exp_transmittance_original = df_clean[transmittance_col].to_numpy(dtype=float)

    # --- データ正規化 (Min-Max Normalization) ---\n
    min_val = np.min(exp_transmittance_original)
    max_val = np.max(exp_transmittance_original)
    if np.isclose(max_val, min_val):
        print("エラー: データの値がほぼ一定であるため、正規化できません。分析をスキップします。")
        return None, None, None
        
    exp_transmittance_normalized = (exp_transmittance_original - min_val) / (max_val - min_val)
    print("データの正規化が完了しました。")

    # 1. バレーの検出 (正規化データに対して実行)
    y_inverted_all = -exp_transmittance_normalized
    valley_indices_all, _ = find_peaks(y_inverted_all, prominence=0.01)
    
    if len(valley_indices_all) < 2:
        print("エラー: 背景補正に必要な数の谷が検出されていません。")
        return None, None, None

    # 2. 背景補正 (2次多項式フィッティング)
    valley_freqs = exp_freq_thz[valley_indices_all]
    valley_transmittance_norm = exp_transmittance_normalized[valley_indices_all]
    
    # 2次多項式でフィッティング
    coeffs = np.polyfit(valley_freqs, valley_transmittance_norm, 2)
    
    # 全周波数範囲で背景を計算
    background = np.polyval(coeffs, exp_freq_thz)
    
    # 背景を差し引いて補正
    exp_transmittance_corrected = exp_transmittance_normalized - background
    
    # --- 補正後データのリスケール [0, 1] ---
    corrected_min = np.min(exp_transmittance_corrected)
    corrected_max = np.max(exp_transmittance_corrected)
    if not np.isclose(corrected_max, corrected_min):
        exp_transmittance_corrected = (exp_transmittance_corrected - corrected_min) / (corrected_max - corrected_min)
    print("補正後データを[0, 1]の範囲にリスケールしました。")
    
    # 3. 補正後データのピーク再検出とFWHM計算
    corrected_peak_indices, _ = find_peaks(exp_transmittance_corrected, height=0.01)
    
    # --- 結果の可視化 (簡易版) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(exp_freq_thz, exp_transmittance_normalized, label='Normalized Raw', alpha=0.5)
    ax.plot(exp_freq_thz, background, label='Background (Poly)', linestyle='--')
    ax.plot(exp_freq_thz, exp_transmittance_corrected, label='Corrected', linewidth=2)
    
    if len(corrected_peak_indices) > 0:
         ax.plot(exp_freq_thz[corrected_peak_indices], exp_transmittance_corrected[corrected_peak_indices], "x", label="Peaks")

    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('Normalized Transmittance')
    ax.set_title(f'Analysis Result: {transmittance_col}')
    ax.legend()
    plt.close()

    print(f"--- {transmittance_col} の分析が完了 ---\n")
    return exp_freq_thz, exp_transmittance_corrected, background


def analyze_transmittance_linear_bg(df, freq_col, transmittance_col):
    """
    指定された透過率データに対して、正規化、ピーク検出、線形補間による背景補正を行う。
    """
    print(f"--- {transmittance_col} の分析を開始 (線形補間背景補正) ---\n")

    df_clean = df[[freq_col, transmittance_col]].dropna()
    if df_clean.empty:
        print(f"警告: {transmittance_col} のデータが空です。スキップします。")
        return None, None, None

    exp_freq_thz = df_clean[freq_col].to_numpy(dtype=float)
    exp_transmittance_original = df_clean[transmittance_col].to_numpy(dtype=float)

    # --- データ正規化 ---
    min_val = np.min(exp_transmittance_original)
    max_val = np.max(exp_transmittance_original)
    if np.isclose(max_val, min_val):
        return None, None, None
        
    exp_transmittance_normalized = (exp_transmittance_original - min_val) / (max_val - min_val)

    # 1. バレーの検出
    y_inverted_all = -exp_transmittance_normalized
    valley_indices_all, _ = find_peaks(y_inverted_all, prominence=0.01)
    
    if len(valley_indices_all) < 2:
        print("エラー: 背景補正に必要な数の谷が検出されていません。")
        return None, None, None

    # 2. 背景補正 (線形補間)
    valley_freqs = exp_freq_thz[valley_indices_all]
    valley_transmittance_norm = exp_transmittance_normalized[valley_indices_all]
    
    # 線形補間で背景を計算
    background = np.interp(exp_freq_thz, valley_freqs, valley_transmittance_norm)
    
    # 背景を差し引いて補正
    exp_transmittance_corrected = exp_transmittance_normalized - background
    
    # --- 補正後データのリスケール [0, 1] ---
    corrected_min = np.min(exp_transmittance_corrected)
    corrected_max = np.max(exp_transmittance_corrected)
    if not np.isclose(corrected_max, corrected_min):
        exp_transmittance_corrected = (exp_transmittance_corrected - corrected_min) / (corrected_max - corrected_min)
    
    # 結果プロット
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(exp_freq_thz, exp_transmittance_normalized, label='Normalized Raw', alpha=0.5)
    ax.plot(exp_freq_thz, background, label='Background (Linear)', linestyle='--')
    ax.plot(exp_freq_thz, exp_transmittance_corrected, label='Corrected', linewidth=2)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('Normalized Transmittance')
    ax.set_title(f'Analysis Result (Linear BG): {transmittance_col}')
    ax.legend()
    plt.close()

    print(f"--- {transmittance_col} の分析が完了 ---\n")
    return exp_freq_thz, exp_transmittance_corrected, background


def detect_columns(df, freq_col):
    """
    データフレームから 'K' または 'T' で終わる列を自動検出するヘルパー関数。
    """
    # 'K'で終わる列 (温度)
    temp_cols = [col for col in df.columns if str(col).strip().endswith('K') and col != freq_col]
    
    # 'T'で終わる列 (磁場)
    field_cols = [col for col in df.columns if str(col).strip().endswith('T') and col != freq_col]
    
    # 重複を除いて結合
    detected_cols = sorted(list(set(temp_cols + field_cols)))
    
    # 数値順にソートするためのロジック (例: 4K, 10K, 100K となるように)
    def sort_key(val):
        import re
        # 数字部分を抽出
        match = re.search(r'(\d+(\.\d+)?)', str(val))
        if match:
            return float(match.group(1))
        return 0.0

    detected_cols.sort(key=sort_key)
    
    return detected_cols


def process_file_and_save(file_path, sheet_name,target_columns, freq_col='Frequency (THz)', output_filename=None):
    """
    Excelファイルを読み込み、2次多項式背景補正を行って保存する。
    target_columns が None または 'auto' の場合、自動検出を行う。
    """
    print(f"====== ファイル '{file_path}' の処理を開始 ======")
    
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
        print(f"データの読み込みに成功しました。読み込み件数: {len(df)}件")
        print("全列名:", df.columns.tolist())
    except Exception as e:
        print(f"データの読み込み中にエラーが発生しました: {e}")
        return

    # --- カラム自動検出機能 ---
    if target_columns is None or target_columns == 'auto':
        print("ℹ️ 解析対象カラムを自動検出します...")
        target_columns = detect_columns(df, freq_col)
        if not target_columns:
            print("警告: 'K' または 'T' で終わるカラムが見つかりませんでした。")
            return
        print(f"✅ 検出されたカラム: {target_columns}")

    # --- 各列の分析を実行 ---
    results = {}
    for col in target_columns:
        if col in df.columns:
            freq, corrected_trans, background = analyze_transmittance(df, freq_col, col)
            if freq is not None:
                results[col] = {
                    'frequency': freq,
                    'corrected_transmittance': corrected_trans,
                    'background': background
                }
        else:
            print(f"警告: 列 '{col}' がデータフレームに見つかりません。スキップします。")

    # --- 保存処理 ---
    if results:
        first_key = next(iter(results))
        df_to_save = pd.DataFrame({freq_col: results[first_key]['frequency']})

        for col_name, data in results.items():
            df_to_save[f'{col_name}'] = data['corrected_transmittance']

        # 出力ファイル名が指定されていない場合は自動生成
        if output_filename is None:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_filename = f"Corrected_{base_name}.xlsx"

        output_dir = "fullsize_corrected_exp_datasets"
        output_filepath = os.path.join(output_dir, output_filename)

        try:
            os.makedirs(output_dir, exist_ok=True)
            df_to_save.to_excel(output_filepath, index=False, sheet_name='Corrected Data')
            print(f"\n補正後のデータを '{output_filepath}' に保存しました。")
            print("保存された列:", df_to_save.columns.tolist())
        except Exception as e:
            print(f"\nファイルの保存中にエラーが発生しました: {e}")
    else:
        print("\n保存するデータがありません。")
    
    print(f"====== ファイル '{file_path}' の処理が完了 ======\\n")


def process_file_and_save_linear_bg(file_path, sheet_name, target_columns, freq_col='Frequency (THz)', output_filename=None):
    """
    Excelファイルを読み込み、線形補間背景補正を行って保存する。
    target_columns が None または 'auto' の場合、自動検出を行う。
    """
    print(f"====== ファイル '{file_path}' の処理を開始 (線形補間背景補正) ======")
    
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
        print(f"データの読み込みに成功しました。読み込み件数: {len(df)}件")
    except Exception as e:
        print(f"データの読み込み中にエラーが発生しました: {e}")
        return

    # --- カラム自動検出機能 ---
    if target_columns is None or target_columns == 'auto':
        print("ℹ️ 解析対象カラムを自動検出します...")
        target_columns = detect_columns(df, freq_col)
        if not target_columns:
            print("警告: 'K' または 'T' で終わるカラムが見つかりませんでした。")
            return
        print(f"✅ 検出されたカラム: {target_columns}")

    results = {}
    for col in target_columns:
        if col in df.columns:
            freq, corrected_trans, background = analyze_transmittance_linear_bg(df, freq_col, col)
            if freq is not None:
                results[col] = {
                    'frequency': freq,
                    'corrected_transmittance': corrected_trans,
                    'background': background
                }
        else:
            print(f"警告: 列 '{col}' がデータフレームに見つかりません。スキップします。")

    if results:
        first_key = next(iter(results))
        df_to_save = pd.DataFrame({freq_col: results[first_key]['frequency']})

        for col_name, data in results.items():
            df_to_save[f'{col_name}'] = data['corrected_transmittance']

        if output_filename is None:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_filename = f"Corrected_{base_name}_Linear.xlsx"

        output_dir = "fullsize_corrected_exp_datasets_linear"
        output_filepath = os.path.join(output_dir, output_filename)

        try:
            os.makedirs(output_dir, exist_ok=True)
            df_to_save.to_excel(output_filepath, index=False, sheet_name='Corrected Data Linear')
            print(f"\n補正後のデータを '{output_filepath}' に保存しました。")
        except Exception as e:
            print(f"\nファイルの保存中にエラーが発生しました: {e}")
    else:
        print("\n保存するデータがありません。")
    
    print(f"====== ファイル '{file_path}' の処理が完了 (線形補間背景補正) ======\\n")


process_file_and_save(
     file_path="C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\Circular_Polarization_B_Field.xlsm", 
     sheet_name="Sheet1", 
     target_columns="auto"
    )

process_file_and_save(
     file_path="C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\Circular_Polarization_Temparature.xlsx", 
     sheet_name="Circular_Polarization_Temp", 
     target_columns="auto"
    )