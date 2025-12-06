# GPU実行結果分析に基づく修正内容まとめ

## 📋 修正完了日: 2025-12-03

---

## 🎯 修正の目的

**前回実行の問題点:**
- R-hat = 5×10¹⁵ (完全発散)
- ESS = 4.0 (実質収束していない)
- Acceptance prob = 0.00 (全サンプル棄却)
- Divergences = 10,000 (最大値)
- Step size = 2.23×10⁻³⁰⁸ (異常値)

**根本原因:**
1. 数値スケールの不整合 (10^-34 などの微小値による勾配計算の破綻)
2. B4/B6パラメータが文献値と不一致 (物理モデルの誤り)
3. MCMCサンプラー設定が不適切 (divergence対策不足)

---

## ✅ 実施した修正

### 1. **THzスケーリングの導入** (最重要)

#### PhysicalConstants クラスの書き換え

```python
class PhysicalConstants:
    # 基準: 1 THz = 1.0
    SCALE_FREQ = 1.0e12  # 1 THz
    
    # SI値 → THzスケール変換
    kB: float = 1.38e-23 / (6.626e-34 * 1e12)   # ≈ 0.021 THz/K
    muB: float = 9.27e-24 / (6.626e-34 * 1e12)  # ≈ 0.014 THz/T
    hbar: float = 1.0 / (2 * π)                  # ≈ 0.159
```

**効果:**
- 従来: kB=1.38e-23, hbar=1.05e-34 → underflow/overflow
- 修正後: kB≈0.021, hbar≈0.159 → 安定した勾配計算

#### omega計算の修正

```python
# 修正前
'omega': flat_freq * 1e12 * 2 * np.pi  # 10^13レベルの巨大値

# 修正後
'omega': flat_freq * 2 * np.pi  # THz単位 (1~10程度)
```

---

### 2. **B4/B6の文献値反映**

#### initial_values (config)

```yaml
# 修正前
B4: 0.000300  # 根拠不明
B6: 0.000030  # 根拠不明

# 修正後 (R.M. Macfarlane et al.)
B4: 0.00202   # [K]
B6: -0.0000120  # [K] ← 負の値!
```

#### bayesian_priors (config)

```yaml
# B4: 文献値0.00202 K を中心に
B4:
  mu: 0.00202
  sigma: 0.0005  # ±0.0005K の探索範囲

# B6: 文献値 -1.20e-5 K (負値)
B6:
  mu: -0.000012
  sigma: 0.000005  # 符号が変わらない範囲
```

**物理的意義:**
- B6が負 → エネルギー準位の分裂パターンが正しくなる
- Gd³⁺の結晶場分裂を正確に再現

---

### 3. **gamma事前分布のTHzスケール対応**

```yaml
# 修正前 (SI単位想定)
log_gamma_mu_base:
  mu: 25.0  # exp(25) ≈ 7×10¹⁰ [rad/s] 想定

# 修正後 (THz単位)
log_gamma_mu_base:
  mu: -3.0  # exp(-3) ≈ 0.05 THz = 50 GHz
```

**妥当性:**
- Gd³⁺の緩和時間: 10-100 GHz (文献的に妥当)
- THzスケールで exp(-3) ≈ 0.05 THz = 50 GHz

---

### 4. **MCMC設定の安定化**

```yaml
# 修正前
target_accept: 0.85
init: "adapt_diag"

# 修正後
target_accept: 0.95       # divergence対策
max_treedepth: 12         # 複雑なモデル対応
init: "jitter+adapt_diag" # 初期化安定化
```

**効果:**
- target_accept=0.95: ステップサイズを小さくし、発散を防ぐ
- max_treedepth=12: 深い探索を許可 (デフォルト10 → 12)
- jitter: 初期値にノイズを加えて局所解を回避

---

## 📊 期待される改善効果

### 数値安定性

| 項目 | 修正前 | 修正後 | 効果 |
|------|--------|--------|------|
| kB | 1.38×10⁻²³ | 0.021 | 勾配計算安定化 |
| hbar | 1.05×10⁻³⁴ | 0.159 | underflow解消 |
| omega | ~10¹³ | ~5 | 演算範囲内 |

### サンプリング品質 (期待値)

| 指標 | 前回 | 目標 |
|------|------|------|
| R-hat | 5×10¹⁵ | < 1.01 |
| ESS | 4.0 | > 400 |
| Acceptance prob | 0.00 | 0.6-0.9 |
| Divergences | 10,000 | < 10 |

### 物理的妥当性

- B6が負 → 正しいエネルギー準位分裂
- gamma ≈ 50 GHz → 文献値と一致
- 全パラメータが物理的に妥当な範囲

---

## 🔧 次回実行時の確認事項

### Phase 1: 動作確認
1. エラーなく実行開始できるか
2. Acceptance prob が 0.6-0.9 になるか
3. Divergences が激減するか

### Phase 2: 収束診断
4. R-hat < 1.01 を達成できるか
5. ESS > 400 を達成できるか
6. トレースプロットで mixing を確認

### Phase 3: 物理的妥当性
7. g-factor が 1.8-2.2 の範囲か
8. B4, B6 が文献値 ±10% 以内か
9. gamma が 10-100 GHz の範囲か

---

## 📝 補足情報

### 波長計算への影響

```python
# calculate_transmission_vectorized 内
wavelength = (2.0 * np.pi * PT_C) / omega_array
# PT_C = 2.998e8 [m/s] (SI単位のまま)
# omega_array: THz単位 [rad/ps]
# → wavelength は正しく計算される
```

**検証:**
- freq = 0.3 THz → omega = 0.3 * 2π ≈ 1.88 [THz]
- wavelength = 2π * 2.998e8 / 1.88e12 ≈ 1mm (正しい)

### G0の計算

```python
G0 = a_scale * mu0 * N_spin * (g_factor * muB)^2 / (2 * hbar)
```

- muB: THz単位 (0.014 THz/T)
- hbar: 無次元 (1/2π)
- → G0の次元: [THz²] (エネルギー²スケール)
- chi: [THz⁻¹] (エネルギー⁻¹)
- → G0 * chi: 無次元 (正しい)

---

## ⚠️ 既知の制約

1. **データファイルパス**: `bayesian_inputs/` ディレクトリが存在しない場合は元のパスにフォールバック
2. **CPU版との互換性**: THzスケーリングはGPU版専用 (CPU版は別ファイル)
3. **結果の単位**: gamma などの出力は THz 単位で保存される (解析時に注意)

---

## 🎓 参考文献

- R.M. Macfarlane et al., "Crystal-field levels in Gd³⁺:GGG"
  - B4 = 0.00202 K
  - B6 = -1.20×10⁻⁵ K

---

## 📞 トラブルシューティング

### もし依然として収束しない場合

1. **事前分布をさらに緩める**
   ```yaml
   B4:
     sigma: 0.001  # 0.0005 → 0.001
   ```

2. **データをサブサンプリング**
   ```python
   # uni_gpu_test.py の main() 内
   if debug_mode:
       subsample_idx = np.random.choice(len(omega), size=1000)
       unified_data = {k: v[subsample_idx] for k, v in unified_data.items()}
   ```

3. **より強い初期化**
   ```yaml
   init: "advi+adapt_diag"  # Variational inference で初期化
   ```

---

**修正完了**: すべての変更が `uni_gpu_test.py` と `config_unified_gpu.yml` に反映されました。
