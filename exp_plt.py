import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 0. プロット設定 ---
plt.rcParams['font.family'] = "Meiryo"
plt.rcParams['figure.dpi'] = 100


# --- 2. データの読み込みと準備 ---
print("実験データを読み込みます...")
file_path = "Circular_Polarization_B_Field.xlsx"
sheet_name = "Sheet2"
# pandasを使ってExcelファイルからデータを読み込む
# ファイル名を適宜確認・変更してください
try:
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0, names=['Frequency (THz)', 'Transmittance'])
    # A列とB列をそれぞれx, yとして読み込む
   #exp_freq_thz = df.iloc[1:, 0].values
   #exp_transmittance = df.iloc[1:, 1].values
    exp_freq_thz = df['Frequency (THz)'].to_numpy(dtype=float)
    exp_transmittance = df['Transmittance'].to_numpy(dtype=float)
    print(f"データの読み込みに成功しました。読み込み件数: {len(df)}件")
except FileNotFoundError:
    print(f"エラー: ファイルが見つかりません。パスを確認してください。\nパス: {file_path}")
    exit()
except Exception as e:
    # その他のエラー（例：シート名が違うなど）もキャッチ
    print(f"データの読み込み中にエラーが発生しました: {e}")
    exit()

print(df)

plt.xlabel('周波数 (THz)')
plt.ylabel('透過率 ')
plt.title('実験結果')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.plot(exp_freq_thz, exp_transmittance, 'o', color='red', markersize=4, label='実験データ')
plt.show()
plt.close()
