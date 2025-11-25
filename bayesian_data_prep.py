import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# --- 設定 ---
plt.rcParams['font.family'] = "Meiryo"
plt.rcParams['figure.dpi'] = 100

def detect_columns(df, freq_col):
    """
    データフレームから 'K' (温度) または 'T' (磁場) で終わる列を自動検出・ソートする。
    """
    # 数値以外の列を除外
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 'K'で終わる列 (温度)
    temp_cols = [col for col in df.columns if str(col).strip().endswith('K') and col != freq_col]
    
    # 'T'で終わる列 (磁場)
    field_cols = [col for col in df.columns if str(col).strip().endswith('T') and col != freq_col]
    
    # 重複を除いて結合
    detected_cols = sorted(list(set(temp_cols + field_cols)))
    
    # 数値順にソート (例: 4K, 10K, 100K)
    def sort_key(val):
        match = re.search(r'(\d+(\.\d+)?)', str(val))
        if match:
            return float(match.group(1))
        return 0.0

    detected_cols.sort(key=sort_key)
    return detected_cols

def normalize_data(series):
    """
    0-1 Min-Max正規化を行う
    """
    vals = series.to_numpy(dtype=float)
    min_val = np.min(vals)
    max_val = np.max(vals)
    
    if np.isclose(max_val, min_val):
        print("  ⚠️ 警告: 値が一定のため正規化できません (All 0になります)")
        return np.zeros_like(vals)
        
    return (vals - min_val) / (max_val - min_val)

def generate_bayesian_input(file_path, sheet_name, freq_col='Frequency (THz)', output_filename=None):
    """
    ベイズ推定専用の正規化データを作成し、保存する。
    背景補正は一切行わない。
    """
    print(f"====== ベイズ推定用データ作成: '{os.path.basename(file_path)}' ======")
    
    # 1. 読み込み
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
        print(f"✅ 読み込み成功 ({len(df)}行)")
    except Exception as e:
        print(f"❌ 読み込みエラー: {e}")
        return

    # 2. カラム検出
    target_columns = detect_columns(df, freq_col)
    if not target_columns:
        print("❌ 解析対象のカラム('K' or 'T')が見つかりませんでした。")
        return
    print(f"ℹ️  検出されたカラム: {target_columns}")

    # 3. データ処理（正規化のみ）
    # 結果格納用DataFrame
    df_clean = df[[freq_col] + target_columns].dropna()
    df_output = pd.DataFrame()
    df_output[freq_col] = df_clean[freq_col]

    print("ℹ️  正規化処理を実行中...")
    for col in target_columns:
        df_output[col] = normalize_data(df_clean[col])

    # 4. 保存設定
    output_dir = "bayesian_inputs" # ベイズ推定用入力フォルダ
    os.makedirs(output_dir, exist_ok=True)

    if output_filename is None:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"BayesianInput_{base_name}.xlsx"
    
    output_path = os.path.join(output_dir, output_filename)

    # 5. Excel保存
    try:
        # シート名は統一して "Normalized Data" とする
        df_output.to_excel(output_path, index=False, sheet_name='Normalized Data')
        print(f"\n🎉 保存完了: {output_path}")
        print(f"   シート名: 'Normalized Data'")
    except Exception as e:
        print(f"❌ 保存エラー: {e}")
        return

    # 6. 確認用プロット (PNG保存)
    try:
        plt.figure(figsize=(10, 6))
        for col in target_columns:
            plt.plot(df_output[freq_col], df_output[col], label=col, alpha=0.8)
        
        plt.xlabel(freq_col)
        plt.ylabel("Normalized Transmittance")
        plt.title(f"Bayesian Input Check: {output_filename}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_filename = output_filename.replace(".xlsx", ".png")
        plt.savefig(os.path.join(output_dir, plot_filename))
        print(f"📊 確認用グラフを保存しました: {plot_filename}")
        plt.close()
    except Exception as e:
        print(f"⚠️ グラフ作成エラー(データ保存には影響しません): {e}")

    print(f"====== 完了 ======\n")

# --- 実行ブロック ---
if __name__ == "__main__":
    # ここに処理したいファイルパスを記述してください
    
    # 温度依存データ
    generate_bayesian_input(
         file_path="C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\Raw_Transmittance_Temperature.xlsx",
         sheet_name="Circular_Polarization_Temp"
    )

    # 磁場依存データ
    generate_bayesian_input(
        file_path="C:\\Users\\taich\\OneDrive - YNU(ynu.jp)\\master\\磁性\\GGG\\Programs\\Raw_Transmittance_Field.xlsm", 
        sheet_name="Sheet1"
    )
