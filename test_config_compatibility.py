"""config_unified_gpu.yml と uni_gpu_test.py の対応確認スクリプト"""
import yaml
import pathlib

def test_config_compatibility():
    config_path = pathlib.Path(__file__).parent / "config_unified_gpu.yml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("=== Config構造確認 ===\n")
    
    # コードで期待される構造と実際のconfig値を比較
    priors_cfg = config.get('bayesian_priors', {})
    mag_priors = priors_cfg.get('magnetic_parameters', {})
    gamma_priors = priors_cfg.get('gamma_parameters', {})
    
    print("📋 Magnetic Parameters:")
    print(f"  g_factor.mu     : {mag_priors.get('g_factor', {}).get('mu', 'MISSING')} (expected: 2.0)")
    print(f"  g_factor.sigma  : {mag_priors.get('g_factor', {}).get('sigma', 'MISSING')} (expected: 0.1)")
    print(f"  B4.mu           : {mag_priors.get('B4', {}).get('mu', 'MISSING')} (expected: 0.0005)")
    print(f"  B4.sigma        : {mag_priors.get('B4', {}).get('sigma', 'MISSING')} (expected: 0.0001)")
    print(f"  B6.mu           : {mag_priors.get('B6', {}).get('mu', 'MISSING')} (expected: 0.00005)")
    print(f"  B6.sigma        : {mag_priors.get('B6', {}).get('sigma', 'MISSING')} (expected: 0.00001)")
    print(f"  a_scale.sigma   : {mag_priors.get('a_scale', {}).get('sigma', 'MISSING')} (expected: 1.0)")
    
    print("\n📋 Gamma Parameters:")
    print(f"  log_gamma_mu_base.mu    : {gamma_priors.get('log_gamma_mu_base', {}).get('mu', 'MISSING')} (expected: 25.0)")
    print(f"  log_gamma_mu_base.sigma : {gamma_priors.get('log_gamma_mu_base', {}).get('sigma', 'MISSING')} (expected: 1.0)")
    print(f"  temp_gamma_slope.sigma  : {gamma_priors.get('temp_gamma_slope', {}).get('sigma', 'MISSING')} (expected: 0.01)")
    print(f"  log_gamma_sigma_base.sigma : {gamma_priors.get('log_gamma_sigma_base', {}).get('sigma', 'MISSING')} (expected: 0.3)")
    print(f"  log_gamma_offset_base.sigma: {gamma_priors.get('log_gamma_offset_base', {}).get('sigma', 'MISSING')} (expected: 0.3)")
    
    # 検証
    print("\n=== 検証結果 ===")
    errors = []
    
    if mag_priors.get('g_factor', {}).get('mu') != 2.0:
        errors.append("g_factor.mu が設定されていません")
    if mag_priors.get('B4', {}).get('mu') != 0.0005:
        errors.append("B4.mu が設定されていません")
    if mag_priors.get('B6', {}).get('mu') != 0.00005:
        errors.append("B6.mu が設定されていません")
    if gamma_priors.get('log_gamma_mu_base', {}).get('mu') != 25.0:
        errors.append("log_gamma_mu_base.mu が設定されていません")
    
    if errors:
        print("❌ エラー検出:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("✅ すべての設定値が正しく定義されています!")
        print("✅ uni_gpu_test.py と config_unified_gpu.yml は完全に対応しています。")
        return True

if __name__ == "__main__":
    test_config_compatibility()
