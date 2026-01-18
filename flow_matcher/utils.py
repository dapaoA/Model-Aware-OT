"""
Flow matcher utilities for creating flow matchers.
"""
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    MA_ExactOT,
)


def create_flow_matcher(method, sigma, ma_method="downsample_2x", time_sampler="uniform"):
    """Create flow matcher based on method.
    
    Args:
        method: Flow matching method ('cfm', 'otcfm', 'sbcfm', 'ma_otcfm', 'ma_tcfm', 'ma3_tcfm', 'ma_incepcfm')
        sigma: Sigma parameter for flow matching
        ma_method: Method for model-aware transformation (only for ma_otcfm/ma_tcfm/ma3_tcfm)
                  If method is 'ma3_tcfm', automatically uses 'downsample_3x'
                  If method is 'ma_incepcfm', automatically uses 'inception'
                  Supported: 'downsample_2x', 'downsample_3x', 'low_pass', 'inception', 'dct_4x4'
        time_sampler: Time sampling method ('uniform', 'logit_normal', 'transport_logit_normal')
        
    Returns:
        Flow matcher instance
    """
    if method == 'cfm':
        return ConditionalFlowMatcher(sigma=sigma, time_sampler=time_sampler)
    elif method == 'otcfm':
        return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma, time_sampler=time_sampler)
    elif method == 'sbcfm':
        # SB-CFM needs positive sigma
        if sigma <= 0:
            sigma = 0.5
        return SchrodingerBridgeConditionalFlowMatcher(sigma=sigma, ot_method='exact')
    elif method == 'ma3_tcfm':
        # ma3_tcfm always uses 3x downsampling
        return MA_ExactOT(sigma=sigma, ma_method='downsample_3x', time_sampler=time_sampler)
    elif method == 'ma_incepcfm':
        # ma_incepcfm always uses Inception feature extraction
        return MA_ExactOT(sigma=sigma, ma_method='inception', time_sampler=time_sampler)
    elif method in ['ma_otcfm', 'ma_tcfm']:
        # ma_tcfm defaults to 2x downsampling if not specified
        if ma_method == "downsample_2x" or ma_method is None:
            ma_method = "downsample_2x"
        return MA_ExactOT(sigma=sigma, ma_method=ma_method, time_sampler=time_sampler)
    else:
        raise ValueError(f"Unknown method: {method}. Must be one of ['cfm', 'otcfm', 'sbcfm', 'ma_otcfm', 'ma_tcfm', 'ma3_tcfm', 'ma_incepcfm']")
