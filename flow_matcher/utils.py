"""
Flow matcher utilities for creating flow matchers.
"""
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    MA_ExactOT,
)


def create_flow_matcher(method, sigma, ma_method="downsample_2x"):
    """Create flow matcher based on method.
    
    Args:
        method: Flow matching method ('cfm', 'otcfm', 'sbcfm', 'ma_otcfm', 'ma_tcfm')
        sigma: Sigma parameter for flow matching
        ma_method: Method for model-aware transformation (only for ma_otcfm/ma_tcfm)
        
    Returns:
        Flow matcher instance
    """
    if method == 'cfm':
        return ConditionalFlowMatcher(sigma=sigma)
    elif method == 'otcfm':
        return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif method == 'sbcfm':
        # SB-CFM needs positive sigma
        if sigma <= 0:
            sigma = 0.5
        return SchrodingerBridgeConditionalFlowMatcher(sigma=sigma, ot_method='exact')
    elif method in ['ma_otcfm', 'ma_tcfm']:
        return MA_ExactOT(sigma=sigma, ma_method=ma_method)
    else:
        raise ValueError(f"Unknown method: {method}. Must be one of ['cfm', 'otcfm', 'sbcfm', 'ma_otcfm', 'ma_tcfm']")
