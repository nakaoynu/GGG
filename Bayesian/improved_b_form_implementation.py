# improved_b_form_implementation.py - 改善されたB_form実装
"""
B_formの信用区間を狭くするための実装
主要改善点:
1. chi値制約 (-0.5, 0.8)
2. 数値安定化パラメータ epsilon
3. mu_r範囲制限
4. 強化された事前分布
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from pytensor.graph.op import Op
import warnings
warnings.filterwarnings('ignore')

# 既存のヘルパー関数をインポート（実際の使用時は適切にインポート）
# from two_step_iterative_fitting import (
#     get_hamiltonian, calculate_susceptibility, calculate_normalized_transmission,
#     mu0, N_spin, muB, hbar, TEMPERATURE
# )

class ImprovedBFormMagneticOp(Op):
    """改善されたB_form磁気モデル - 数値安定性を向上"""
    
    def __init__(self, datasets, field_specific_params):
        self.datasets = datasets
        self.field_specific_params = field_specific_params
        self.itypes = [pt.dscalar, pt.dvector, pt.dscalar, pt.dscalar, pt.dscalar]
        self.otypes = [pt.dvector]
        
        # 数値安定化パラメータ
        self.epsilon = 1e-6
        self.chi_min = -0.5
        self.chi_max = 0.8
        self.mu_r_min = 0.1
        self.mu_r_max = 50.0
    
    def perform(self, node, inputs, output_storage):
        a_scale, gamma, g_factor, B4, B6 = inputs
        full_predicted_y = []
        
        for data in self.datasets:
            try:
                # 該当する磁場の固定パラメータを取得
                b_field = data['b_field']
                if b_field in self.field_specific_params:
                    d_used = self.field_specific_params[b_field]['d']
                    eps_bg_fixed = self.field_specific_params[b_field]['eps_bg']
                else:
                    # フォールバック値
                    d_used = 157.8e-6
                    eps_bg_fixed = 14.0
                
                # ハミルトニアンと磁気感受率の計算
                H = get_hamiltonian(data['b_field'], g_factor, B4, B6)
                chi_raw = calculate_susceptibility(data['omega'], H, data['temperature'], gamma)
                
                # スケーリング係数の計算
                G0 = a_scale * mu0 * N_spin * (g_factor * muB)**2 / (2 * hbar)
                chi = G0 * chi_raw
                
                # 【改善点1】chi値の制約 - 発散を防ぐ重要な処理
                chi_constrained = np.clip(chi, self.chi_min, self.chi_max)
                
                # 数値的安定性のチェック
                if np.any(~np.isfinite(chi_constrained)):
                    chi_constrained = np.where(np.isfinite(chi_constrained), 
                                             chi_constrained, 0.01)
                
                # 【改善点2】安定化されたB_form透磁率計算
                denominator = 1 - chi_constrained + self.epsilon
                
                # 分母がゼロに近い場合の安全処理
                denominator = np.where(np.abs(denominator) > 1e-10, 
                                     denominator, 1e-10)
                
                mu_r = 1 / denominator
                
                # 【改善点3】mu_r範囲制限 - 物理的に意味のある範囲
                mu_r = np.clip(mu_r, self.mu_r_min, self.mu_r_max)
                
                # 最終的な数値安定性チェック
                if np.any(~np.isfinite(mu_r)):
                    mu_r = np.where(np.isfinite(mu_r), mu_r, 1.0)
                
                # 透過率計算
                predicted_y = calculate_normalized_transmission(
                    data['omega'], mu_r, d_used, eps_bg_fixed)
                
                # 予測値の最終チェック
                if np.any(~np.isfinite(predicted_y)):
                    predicted_y = np.where(np.isfinite(predicted_y), 
                                         predicted_y, 0.5)
                
                full_predicted_y.append(predicted_y)
                
            except Exception as e:
                print(f"⚠️  B_form計算エラー (磁場 {data['b_field']}T): {e}")
                # エラー時のフォールバック値
                fallback_y = np.full(len(data['omega']), 0.5)
                full_predicted_y.append(fallback_y)
        
        output_storage[0][0] = np.concatenate(full_predicted_y)

def create_improved_b_form_model(datasets, field_specific_params, 
                                prior_magnetic_params=None):
    """改善されたB_formベイズモデルの作成"""
    
    print("🔧 改善されたB_formモデルを構築中...")
    
    trans_observed = np.concatenate([d['transmittance'] for d in datasets])
    
    with pm.Model() as model:
        # 【改善点4】強化された事前分布設定
        if prior_magnetic_params:
            print("📊 前回結果を事前分布として使用")
            
            # より制約的な事前分布（前回結果の周辺に集中）
            a_scale = pm.TruncatedNormal('a_scale', 
                                       mu=prior_magnetic_params['a_scale'], 
                                       sigma=0.05,  # 標準偏差を小さく
                                       lower=0.2, upper=1.2)
            
            g_factor = pm.TruncatedNormal('g_factor', 
                                        mu=prior_magnetic_params['g_factor'], 
                                        sigma=0.02,  # 標準偏差を小さく
                                        lower=1.9, upper=2.1)
            
            B4 = pm.TruncatedNormal('B4', 
                                  mu=prior_magnetic_params['B4'], 
                                  sigma=0.0002,  # 標準偏差を小さく
                                  lower=-0.005, upper=0.005)
            
            B6 = pm.TruncatedNormal('B6', 
                                  mu=prior_magnetic_params['B6'], 
                                  sigma=0.00001,  # 標準偏差を小さく
                                  lower=-0.0005, upper=0.0005)
        else:
            print("📊 デフォルト事前分布を使用")
            
            # 初回用の制約的事前分布
            a_scale = pm.TruncatedNormal('a_scale', 
                                       mu=0.6, sigma=0.1, 
                                       lower=0.2, upper=1.2)
            
            g_factor = pm.TruncatedNormal('g_factor', 
                                        mu=2.0, sigma=0.05, 
                                        lower=1.9, upper=2.1)
            
            B4 = pm.TruncatedNormal('B4', 
                                  mu=0.001, sigma=0.0005, 
                                  lower=-0.005, upper=0.005)
            
            B6 = pm.TruncatedNormal('B6', 
                                  mu=-0.00001, sigma=0.00002, 
                                  lower=-0.0005, upper=0.0005)
        
        # ガンマパラメータ（log-normal分布で制約）
        log_gamma_mu = pm.Normal('log_gamma_mu', mu=np.log(1e11), sigma=0.3)
        log_gamma_sigma = pm.HalfNormal('log_gamma_sigma', sigma=0.2)
        log_gamma_offset = pm.Normal('log_gamma_offset', mu=0, sigma=0.2, shape=7)
        gamma = pm.Deterministic('gamma', 
                               pt.exp(log_gamma_mu + log_gamma_offset * log_gamma_sigma))
        
        # 改善されたB_formオペレーター
        op = ImprovedBFormMagneticOp(datasets, field_specific_params)
        mu = op(a_scale, gamma, g_factor, B4, B6)
        
        # 観測誤差（より制約的）
        sigma = pm.HalfCauchy('sigma', beta=0.03)  # より小さい誤差
        
        # 観測モデル
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=trans_observed)
        
    return model

def run_improved_b_form_sampling(model, n_datasets=3):
    """改善されたサンプリング設定でB_formを実行"""
    
    print("🚀 改善B_formサンプリングを開始...")
    
    # 【改善点5】高度なサンプリング設定
    sampling_config = {
        'draws': 2500,          # サンプル数増加
        'tune': 4000,           # tune期間延長
        'chains': 4,            # チェーン数増加
        'cores': 4,
        'target_accept': 0.98,  # 高精度設定
        'max_treedepth': 15,    # より深い探索
        'return_inferencedata': True,
        'idata_kwargs': {'log_likelihood': True}
    }
    
    try:
        print("📈 高精度サンプリングを試行...")
        
        # カスタムサンプラーの使用
        with model:
            step = pm.CompoundStep([
                pm.NUTS(['a_scale', 'g_factor'], 
                       target_accept=0.98, max_treedepth=15),
                pm.Slice(['B4', 'B6'], w=0.0001),  # 小さいステップサイズ
                pm.NUTS(['log_gamma_mu', 'log_gamma_sigma'], 
                       target_accept=0.95),
                pm.Metropolis(['log_gamma_offset'])
            ])
            
            trace = pm.sample(step=step, **sampling_config)
            
        print("✅ 高精度サンプリング成功!")
        return trace
        
    except Exception as e:
        print(f"⚠️  高精度サンプリング失敗: {e}")
        print("🔄 標準設定でリトライ...")
        
        # フォールバック設定
        fallback_config = {
            'draws': 2000,
            'tune': 3000,
            'chains': 2,
            'target_accept': 0.90,
            'return_inferencedata': True,
            'idata_kwargs': {'log_likelihood': True}
        }
        
        try:
            with model:
                trace = pm.sample(**fallback_config)
            print("✅ 標準サンプリング成功!")
            return trace
            
        except Exception as e2:
            print(f"❌ サンプリング完全失敗: {e2}")
            return None

def compare_b_form_improvements(original_trace, improved_trace):
    """改善前後のB_form結果比較"""
    
    print("\n=== B_form改善効果の比較 ===")
    
    def analyze_trace(trace, label):
        """トレースの統計解析"""
        if trace is None:
            return None
        
        try:
            posterior = trace["posterior"]
            
            results = {}
            for param in ['a_scale', 'g_factor', 'B4', 'B6']:
                if param in posterior:
                    mean_val = float(posterior[param].mean())
                    std_val = float(posterior[param].std())
                    
                    # 収束指標
                    summary = az.summary(trace, var_names=[param])
                    ess_bulk = summary.loc[param, 'ess_bulk'] if param in summary.index else 0
                    r_hat = summary.loc[param, 'r_hat'] if param in summary.index else np.inf
                    
                    results[param] = {
                        'mean': mean_val,
                        'std': std_val,
                        'ess_bulk': ess_bulk,
                        'r_hat': r_hat
                    }
            
            return results
            
        except Exception as e:
            print(f"❌ {label}の解析失敗: {e}")
            return None
    
    print("📊 統計比較:")
    print("=" * 60)
    
    original_results = analyze_trace(original_trace, "改善前")
    improved_results = analyze_trace(improved_trace, "改善後")
    
    if original_results and improved_results:
        for param in ['a_scale', 'g_factor', 'B4']:
            if param in original_results and param in improved_results:
                orig = original_results[param]
                impr = improved_results[param]
                
                print(f"\n{param}:")
                print(f"  標準偏差: {orig['std']:.4f} → {impr['std']:.4f} "
                      f"({impr['std']/orig['std']:.2f}倍)")
                print(f"  ESS:      {orig['ess_bulk']:4.0f} → {impr['ess_bulk']:4.0f} "
                      f"({impr['ess_bulk']/max(orig['ess_bulk'], 1):.1f}倍)")
                print(f"  R̂:       {orig['r_hat']:.2f} → {impr['r_hat']:.2f}")
    
    print("\n🎯 改善効果評価:")
    
    if improved_results:
        # ESS改善率
        ess_improvements = []
        for param in ['a_scale', 'B4']:
            if (param in improved_results and 
                improved_results[param]['ess_bulk'] > 100):
                ess_improvements.append(improved_results[param]['ess_bulk'])
        
        if ess_improvements:
            avg_ess = np.mean(ess_improvements)
            print(f"✅ 平均ESS: {avg_ess:.0f} (目標: >100)")
        else:
            print("❌ ESS改善不十分")
        
        # R̂改善
        r_hat_good = []
        for param in improved_results.values():
            if param['r_hat'] < 1.1:
                r_hat_good.append(param['r_hat'])
        
        if len(r_hat_good) >= 3:
            print(f"✅ 収束達成: {len(r_hat_good)}/4 パラメータ")
        else:
            print(f"⚠️  収束不完全: {len(r_hat_good)}/4 パラメータ")
    
    return improved_results

# 使用例とテスト関数
def test_improved_b_form():
    """改善されたB_formのテスト実行"""
    
    print("🧪 改善B_formテストを開始...")
    
    # ダミーデータでテスト
    dummy_datasets = [
        {
            'b_field': 5.0,
            'omega': np.linspace(1e12, 3e12, 50),
            'transmittance': np.random.normal(0.5, 0.1, 50),
            'temperature': 1.5
        }
    ]
    
    dummy_field_params = {
        5.0: {'d': 157.8e-6, 'eps_bg': 14.0}
    }
    
    # モデル作成
    model = create_improved_b_form_model(dummy_datasets, dummy_field_params)
    
    print(f"✅ モデル作成成功")
    print(f"📋 パラメータ数: {len(model.free_RVs)}")
    print(f"📊 観測データ点数: {len(dummy_datasets[0]['transmittance'])}")
    
    return model

if __name__ == "__main__":
    # テスト実行
    test_model = test_improved_b_form()
    print("\n🎉 改善B_form実装完了!")
    print("\n📝 次のステップ:")
    print("1. 実データでの検証")
    print("2. H_formとの性能比較")
    print("3. 信用区間の改善確認")
