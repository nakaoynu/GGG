from graphviz import Digraph

def create_fitting_workflow_chart():
    """
    物理モデルの自動最適化プログラムのワークフロー図を生成する。
    日本語フォントを指定して文字化けを防ぐ。
    """
    # 日本語表示のためのフォント設定 (Windows環境を想定)
    jp_font = "Yu Gothic UI"

    # グラフオブジェクトの初期化 (有向グラフ)
    dot = Digraph(
        comment='Physics Model Fitting Workflow',
        graph_attr={
            'rankdir': 'TB',  # Top to Bottom (上から下へ)
            'label': '物理モデル自動最適化ワークフロー',
            'fontsize': '20',
            'fontname': jp_font
        }
    )

    # ノード(箱)とエッジ(矢印)のデフォルトスタイル設定
    dot.attr('node', shape='box', style='rounded,filled', fontname=jp_font)
    dot.attr('edge', fontname=jp_font)

    # --- 1. 準備フェーズのノード定義 ---
    with dot.subgraph(name='cluster_preparation') as c:
        c.attr(label='1-3. 準備フェーズ', style='rounded', color='gray', fontname=jp_font)
        c.node('Constants', '1. グローバル定数と行列の定義', shape='folder', fillcolor='#D6EAF8')
        c.node('ClassDef', '2. PhysicsModel クラスの定義', shape='box3d', fillcolor='#D6EAF8')
        c.node('DataInput', '3. 実験データ読み込み\n(Excel & 正規化)', shape='note', fillcolor='#FCF3CF')

    # --- 2. 最適化フェーズのノード定義 ---
    with dot.subgraph(name='cluster_optimization') as c:
        c.attr(label='4. 自動最適化フェーズ', style='rounded', color='gray', fontname=jp_font)
        c.node('CostFunc', 'コスト関数 cost_function の定義\n(理論値と実験値の誤差を計算)', fillcolor='#FDEDEC')
        
        # ループ処理を表現
        c.node('LoopStart', "最適化ループ開始\nfor model in ['H_form', 'B_form']", shape='Mdiamond', fillcolor='#FADBD8')
        c.node('Minimize', 'scipy.optimize.minimize実行\n(誤差最小化)', shape='diamond', fillcolor='#E8DAEF')
        c.node('ResultDict', '結果を辞書に格納\nresults_dict', shape='cylinder', fillcolor='#FEF9E7')

    # --- 3. 評価フェーズのノード定義 ---
    with dot.subgraph(name='cluster_evaluation') as c:
        c.attr(label='5. 評価フェーズ (ループ後)', style='rounded', color='gray', fontname=jp_font)
        c.node('PrintResults', '最適化結果をターミナルに表示', shape='parallelogram', fillcolor='#D5F5E3')
        c.node('FinalPlot', '最終比較グラフを作成・表示', shape='image', fillcolor='#F6DDCC')

    # --- エッジ(矢印)で各ステップを接続 ---
    dot.edge('Constants', 'ClassDef', label='利用')
    dot.edge('ClassDef', 'CostFunc', label='利用')
    dot.edge('DataInput', 'CostFunc', label='比較対象')
    
    dot.edge('CostFunc', 'LoopStart', label='使用')
    dot.edge('LoopStart', 'Minimize')
    dot.edge('Minimize', 'ResultDict')
    dot.edge('ResultDict', 'LoopStart', style='dashed', label='次のモデルへ')

    dot.edge('ResultDict', 'PrintResults', label='全モデル完了後')
    dot.edge('PrintResults', 'FinalPlot')

    # --- ファイルとして保存 ---
    output_filename = 'auto_fitting_workflow_chart'
    try:
        # cleanup=Trueで、dotソースファイルなどの中間ファイルを削除
        dot.render(output_filename, format='png', view=False, cleanup=True)
        print(f"✅ ワークフロー図 '{output_filename}.png' を生成しました。")
    except Exception as e:
        print(f"❌ ワークフロー図の生成中にエラーが発生しました。")
        print("   Graphvizがシステムにインストールされ、環境変数PATHが設定されているか確認してください。")
        print(f"   詳細: {e}")

# --- スクリプトとして実行 ---
if __name__ == '__main__':
    create_fitting_workflow_chart()