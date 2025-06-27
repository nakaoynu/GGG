from graphviz import Digraph

def create_workflow_chart():
    """
    GGG_model_comparision.py の解析ワークフロー図を生成する関数
    Graphvizで日本語が文字化けしないようにフォントを明示的に指定
    """
    
    # --- ★★★ ここが重要: Graphvizが認識できる日本語フォント名を指定 ★★★ ---
    # Windows環境では "Yu Gothic UI" や "Meiryo UI" が安定しています。
    # もしこれらでうまくいかない場合は、"MS Gothic" などを試してください。
    jp_font = "Yu Gothic UI"

    # グラフオブジェクトを作成 (DiGraph = Directed Graph, 有向グラフ)
    dot = Digraph(
        comment='GGG Model Comparison Workflow',
        graph_attr={
            'rankdir': 'TB',  # 上から下へ (Top to Bottom)
            'label': 'GGGモデル比較解析ワークフロー',
            'fontsize': '20',
            'fontname': jp_font  # グラフ全体のタイトルフォント
        }
    )

    # ノード（箱）とエッジ（矢印）のデフォルトスタイルにフォントを指定
    dot.attr('node', shape='box', style='rounded,filled', fontname=jp_font)
    dot.attr('edge', fontname=jp_font) # 矢印のラベルにもフォントを適用

    # --- ノード（各ステップ）を定義 ---

    with dot.subgraph(name='cluster_preparation') as c:
        c.attr(label='1. 準備フェーズ', style='rounded', color='gray', fontname=jp_font)
        c.node('DataInput', '実験データ\nCircular_Polarization_B_Field.xlsx', shape='note', fillcolor='#FCF3CF')
        c.node('FuncDef', '物理モデルの定義\nハミルトニアンや透過率の計算関数', fillcolor='#D6EAF8')

    dot.node('LoopStart', "モデル比較ループの開始\nfor model in ['H_form', 'B_form']", shape='hexagon', fillcolor='#FADBD8')
    
    with dot.subgraph(name='cluster_loop_body') as c:
        c.attr(label='ループ内の処理 (各モデルで実行)', style='rounded', color='gray', fontname=jp_font)
        c.node('ModelDef', '3a. PyMCモデル構築\n事前分布と尤度を定義\n(PhysicsModelOp)', shape='diamond', fillcolor='#FDEDEC')
        c.node('Sampling', '3b. MCMCサンプリング\npm.sample()で事後分布を計算', fillcolor='#E8DAEF')
        c.node('Storage', '3c. 結果を格納\ntracesとppcs辞書に保存', shape='cylinder', fillcolor='#FEF9E7')

    with dot.subgraph(name='cluster_evaluation') as c:
        c.attr(label='4. 解析と評価フェーズ (ループ後)', style='rounded', color='gray', fontname=jp_font)
        c.node('ModelCompare', '4a. モデル比較\naz.compare()でLOO-CVを計算', shape='stadium', fillcolor='#D5F5E3')
        c.node('Plotting', '4b. 結果の可視化\n比較表とフィット曲線をプロット', shape='parallelogram', fillcolor='#F6DDCC')


    # --- エッジ（矢印）を定義して、フローを接続 ---
    dot.edge('DataInput', 'LoopStart', label='入力')
    dot.edge('FuncDef', 'LoopStart', label='定義')
    dot.edge('LoopStart', 'ModelDef')
    dot.edge('ModelDef', 'Sampling')
    dot.edge('Sampling', 'Storage')
    dot.edge('Storage', 'LoopStart', style='dashed', label='次のモデルへ')
    dot.edge('Storage', 'ModelCompare', label='全モデル完了後')
    dot.edge('ModelCompare', 'Plotting')

    # --- ファイルとして保存 ---
    output_filename = 'GGG_model_comparison_workflow'
    try:
        dot.render(output_filename, format='png', view=False, cleanup=True)
        print(f"✅ ワークフロー図 '{output_filename}.png' を生成しました。")
    except Exception as e:
        print(f"❌ ワークフロー図の生成中にエラーが発生しました。")
        print("   Graphvizがシステムにインストールされているか確認してください。")
        print(f"   詳細: {e}")

# --- スクリプトとして実行 ---
if __name__ == '__main__':
    create_workflow_chart()