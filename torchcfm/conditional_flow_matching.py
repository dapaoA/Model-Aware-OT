"""Implements Conditional Flow Matcher Losses."""

# Author: Alex Tong
#         Kilian Fatras
#         +++
# License: MIT License

import math
import warnings
from typing import Union, Callable

import torch
import torch.fft
import numpy as np
from scipy.fft import dctn

from .optimal_transport import OTPlanSampler


def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class ConditionalFlowMatcher:
    """Base class for conditional flow matching methods. This class implements the independent
    conditional flow matching methods from [1] and serves as a parent class for all other flow
    matching methods.

    It implements:
    - Drawing data from gaussian probability path N(t * x1 + (1 - t) * x0, sigma) function
    - conditional flow matching ut(x1|x0) = x1 - x0
    - score function $\nabla log p_t(x|x0, x1)$
    """

    def __init__(self, sigma: Union[float, int] = 0.0, time_sampler: str = "uniform"):
        r"""Initialize the ConditionalFlowMatcher class.

        It requires the hyper-parameter $\sigma$.
                Parameters
                ----------
                sigma : Union[float, int]
                time_sampler : str
                    Time sampling method: 'uniform', 'logit_normal', or 'transport_logit_normal'
        """
        self.sigma = sigma
        self.time_sampler = time_sampler

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1 + (1 - t) * x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t, xt
        return x1 - x0

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_time(self, batch_size, device, x0=None, x1=None):
        """
        Sample time values based on time_sampler method.
        
        Parameters
        ----------
        batch_size : int
            Batch size
        device : torch.device
            Device to place tensors on
        x0 : Tensor, optional
            Source batch (needed for transport_logit_normal)
        x1 : Tensor, optional
            Target batch (needed for transport_logit_normal)
            
        Returns
        -------
        t : FloatTensor, shape (batch_size,)
            Sampled time values in [0, 1]
        """
        if self.time_sampler == "uniform":
            # Uniform sampling in [0, 1]
            return torch.rand(batch_size, device=device)
        
        elif self.time_sampler == "logit_normal":
            # Logit-normal distribution: samples from N(0, 1), then apply sigmoid
            # This gives higher probability around 0.5, with 0 and 1 having probability 0
            z = torch.randn(batch_size, device=device)
            # Use sigmoid to map to [0, 1] with concentration around 0.5
            t = torch.sigmoid(z)
            return t
        
        elif self.time_sampler == "transport_logit_normal":
            # Transport logit-normal: assign t based on pairing distance
            # Closer pairs get t closer to 0.5, farther pairs get t closer to 0 or 1
            if x0 is None or x1 is None:
                raise ValueError("x0 and x1 are required for transport_logit_normal time sampling")
            
            # Compute pairwise distances (flatten spatial dimensions)
            x0_flat = x0.reshape(x0.shape[0], -1)
            x1_flat = x1.reshape(x1.shape[0], -1)
            distances = torch.norm(x0_flat - x1_flat, dim=1)  # (batch_size,)
            
            # Sample t values using logit_normal (centered around 0.5)
            z = torch.randn(batch_size, device=device)
            t_sampled = torch.sigmoid(z)  # (batch_size,)
            
            # Sort t values by distance from 0.5 (closer to 0.5 first)
            # Compute |t - 0.5| for each sampled t
            t_dist_from_center = torch.abs(t_sampled - 0.5)
            
            # Get indices that sort t by distance from 0.5 (ascending: closest to 0.5 first)
            _, t_sorted_indices = torch.sort(t_dist_from_center)
            
            # Get indices that sort distances (ascending: closest pairs first)
            _, dist_sorted_indices = torch.sort(distances)
            
            # Assign t values: closest pairs get t closest to 0.5
            t = torch.zeros_like(t_sampled)
            t[dist_sorted_indices] = t_sampled[t_sorted_indices]
            
            return t
        
        else:
            raise ValueError(f"Unknown time_sampler: {self.time_sampler}. Must be 'uniform', 'logit_normal', or 'transport_logit_normal'")

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn according to time_sampler
        return_noise : bool
            return the noise sample epsilon


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) eps: Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        if t is None:
            t = self.sample_time(x0.shape[0], x0.device, x0, x1)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def compute_lambda(self, t):
        """Compute the lambda function, see Eq.(23) [3].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        lambda : score weighting function

        References
        ----------
        [4] Simulation-free Schrodinger bridges via score and flow matching, Preprint, Tong et al.
        """
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma**2 + 1e-8)


class ExactOptimalTransportConditionalFlowMatcher(ConditionalFlowMatcher):
    """Child class for optimal transport conditional flow matching method.

    This class implements the OT-CFM methods from [1] and inherits the ConditionalFlowMatcher
    parent class.

    It overrides the sample_location_and_conditional_flow.
    """

    def __init__(self, sigma: Union[float, int] = 0.0, time_sampler: str = "uniform"):
        r"""Initialize the ConditionalFlowMatcher class.

        It requires the hyper-parameter $\sigma$.
                Parameters
                ----------
                sigma : Union[float, int]
                time_sampler : str
                    Time sampling method: 'uniform', 'logit_normal', or 'transport_logit_normal'
                ot_sampler: exact OT method to draw couplings (x0, x1) (see Eq.(17) [1]).
        """
        super().__init__(sigma, time_sampler=time_sampler)
        self.ot_sampler = OTPlanSampler(method="exact")

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn according to time_sampler
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        # First do OT pairing
        x0_paired, x1_paired = self.ot_sampler.sample_plan(x0, x1)
        
        # For transport_logit_normal, we need to compute t based on paired distances
        # For other time samplers, t will be computed in parent class
        if t is None and self.time_sampler == "transport_logit_normal":
            t = self.sample_time(x0_paired.shape[0], x0_paired.device, x0_paired, x1_paired)
        
        return super().sample_location_and_conditional_flow(x0_paired, x1_paired, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        y0 : Tensor, shape (bs) (default: None)
            represents the source label minibatch
        y1 : Tensor, shape (bs) (default: None)
            represents the target label minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


class TargetConditionalFlowMatcher(ConditionalFlowMatcher):
    """Lipman et al.

    2023 style target OT conditional flow matching. This class inherits the ConditionalFlowMatcher
    and override the compute_mu_t, compute_sigma_t and compute_conditional_flow functions in order
    to compute [2]'s flow matching.

    [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
    """

    def compute_mu_t(self, x0, x1, t):
        """Compute the mean of the probability path tx1, see (Eq.20) [2].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1

        References
        ----------
        [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        del x0
        t = pad_t_like_x(t, x1)
        return t * x1

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t x1, 1 - (1 - sigma) t), see (Eq.20) [2].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma 1 - (1 - sigma) t

        References
        ----------
        [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        return 1 - (1 - self.sigma) * t

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = (x1 - (1 - sigma) t)/(1 - (1 - sigma)t), see Eq.(21) [2].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = (x1 - (1 - sigma) t)/(1 - (1 - sigma)t)

        References
        ----------
        [1] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        del x0
        t = pad_t_like_x(t, x1)
        return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)


class SchrodingerBridgeConditionalFlowMatcher(ConditionalFlowMatcher):
    """Child class for Schrödinger bridge conditional flow matching method.

    This class implements the SB-CFM methods from [1] and inherits the ConditionalFlowMatcher
    parent class.

    It overrides the compute_sigma_t, compute_conditional_flow and
    sample_location_and_conditional_flow functions.
    """

    def __init__(self, sigma: Union[float, int] = 1.0, ot_method="exact", time_sampler: str = "uniform"):
        r"""Initialize the SchrodingerBridgeConditionalFlowMatcher class.

        It requires the hyper- parameter $\sigma$ and the entropic OT map.

        Parameters
        ----------
        sigma : Union[float, int]
        ot_method : str
            OT method to draw couplings (x0, x1) (see Eq.(17) [1]).
            we use exact as the default as we found this to perform better
            (more accurate and faster) in practice for reasonable batch sizes.
            We note that as batchsize --> infinity the correct choice is the
            sinkhorn method theoretically.
        time_sampler : str
            Time sampling method: 'uniform', 'logit_normal', or 'transport_logit_normal'
        """
        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}.")
        elif sigma < 1e-3:
            warnings.warn("Small sigma values may lead to numerical instability.")
        super().__init__(sigma, time_sampler=time_sampler)
        self.ot_method = ot_method
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=2 * self.sigma**2)

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sqrt(t * (1 - t))*sigma^2),
        see (Eq.20) [1].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        return self.sigma * torch.sqrt(t * (1 - t))

    def compute_conditional_flow(self, x0, x1, t, xt):
        """Compute the conditional vector field.

        ut(x1|x0) = (1 - 2 * t) / (2 * t * (1 - t)) * (xt - mu_t) + x1 - x0,
        see Eq.(21) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field
        ut(x1|x0) = (1 - 2 * t) / (2 * t * (1 - t)) * (xt - mu_t) + x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models
        with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x0)
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + x1 - x0
        return ut

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sqrt(t * (1 - t))*sigma^2 ))
        and the conditional vector field ut(x1|x0) = (1 - 2 * t) / (2 * t * (1 - t)) * (xt - mu_t) + x1 - x0,
        (see Eq.(15) [1]) with respect to the minibatch entropic OT plan.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise: bool
            return the noise sample epsilon


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch entropic OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        y0 : Tensor, shape (bs) (default: None)
            represents the source label minibatch
        y1 : Tensor, shape (bs) (default: None)
            represents the target label minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


class VariancePreservingConditionalFlowMatcher(ConditionalFlowMatcher):
    """Albergo et al.

    2023 trigonometric interpolants class. This class inherits the ConditionalFlowMatcher and
    override the compute_mu_t and compute_conditional_flow functions in order to compute [3]'s
    trigonometric interpolants.

    [3] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, Albergo et al.
    """

    def compute_mu_t(self, x0, x1, t):
        r"""Compute the mean of the probability path (Eq.5) from [3].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: cos(pi t/2)x0 + sin(pi t/2)x1

        References
        ----------
        [3] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, Albergo et al.
        """
        t = pad_t_like_x(t, x0)
        return torch.cos(math.pi / 2 * t) * x0 + torch.sin(math.pi / 2 * t) * x1

    def compute_conditional_flow(self, x0, x1, t, xt):
        r"""Compute the conditional vector field similar to [3].

        ut(x1|x0) = pi/2 (cos(pi*t/2) x1 - sin(pi*t/2) x0),
        see Eq.(21) [3].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field
        ut(x1|x0) = pi/2 (cos(pi*t/2) x1 - sin(\pi*t/2) x0)

        References
        ----------
        [3] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, Albergo et al.
        """
        del xt
        t = pad_t_like_x(t, x0)
        return math.pi / 2 * (torch.cos(math.pi / 2 * t) * x1 - torch.sin(math.pi / 2 * t) * x0)


class MA_ExactOT(ExactOptimalTransportConditionalFlowMatcher):
    """Model-Aware Exact Optimal Transport Conditional Flow Matcher.
    
    This class extends ExactOptimalTransportConditionalFlowMatcher by applying
    a transformation M to x0 and x1 before computing the OT plan. The transformation
    M can be configured via different methods (e.g., low-pass filtering).
    
    It overrides the sample_location_and_conditional_flow method to use M(x0) and M(x1)
    for OT plan computation, but still uses original x0 and x1 for the actual flow matching.
    """

    def __init__(self, sigma: Union[float, int] = 0.0, ma_method: str = "downsample_2x", time_sampler: str = "uniform"):
        r"""Initialize the MA_ExactOT class.

        Parameters
        ----------
        sigma : Union[float, int]
            Hyper-parameter for the flow matcher.
        ma_method : str
            Method for the transformation M. Currently supports:
            - "downsample_2x": 2x downsampling (default for ma_tcfm)
            - "downsample_3x": 3x downsampling (for ma3_tcfm)
            - "low_pass": Low-pass filtering using FFT (filters out 80% high-frequency region)
            - "inception": Pre-trained WideResNet-40 feature extractor for CIFAR-10 (trained with energy-constrained learning)
            - "dct_4x4": DCT 4x4 low-frequency coefficients extraction using zigzag scanning
            - "dct_8x8": DCT 8x8 low-frequency coefficients extraction using zigzag scanning
        time_sampler : str
            Time sampling method: 'uniform', 'logit_normal', or 'transport_logit_normal'
        """
        super().__init__(sigma, time_sampler=time_sampler)
        self.ma_method = ma_method
        self.M = self._get_transformation(ma_method)
        # Initialize Inception model if needed
        if ma_method == "inception":
            self._init_inception_model()

    def _get_transformation(self, method: str) -> Callable:
        """Get the transformation function M based on method name."""
        if method == "downsample_2x":
            return lambda x: self._downsample_nx(x, factor=2)
        elif method == "downsample_3x":
            return lambda x: self._downsample_nx(x, factor=3)
        elif method == "low_pass":
            return self._low_pass_filter
        elif method == "inception":
            return self._inception_feature_extractor
        elif method == "dct_4x4":
            return self._dct_4x4_extractor
        elif method == "dct_8x8":
            return self._dct_8x8_extractor
        else:
            raise ValueError(f"Unknown MA method: {method}. Supported methods: ['downsample_2x', 'downsample_3x', 'low_pass', 'inception', 'dct_4x4', 'dct_8x8']")

    def _low_pass_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply low-pass filter using FFT.
        
        Filters out 80% of the high-frequency region, keeping only the center 20%
        low-frequency region.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W) for images or (B, D) for 2D data
            
        Returns
        -------
        torch.Tensor
            Low-pass filtered tensor with same shape as input
        """
        if x.dim() == 4:
            # Image data: (B, C, H, W)
            return self._low_pass_filter_image(x)
        elif x.dim() == 2:
            # 2D data: (B, D)
            return self._low_pass_filter_2d(x)
        else:
            # For other dimensions, return as is
            return x

    def _low_pass_filter_image(self, x: torch.Tensor) -> torch.Tensor:
        """Apply low-pass filter to image data using 2D FFT.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W)
            
        Returns
        -------
        torch.Tensor
            Low-pass filtered image with same shape
        """
        B, C, H, W = x.shape
        
        # Compute FFT (shifted to center low frequencies)
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
        
        # Create circular mask for low-pass filter
        # Keep center 20% of frequencies (sqrt(0.2) ≈ 0.447 radius)
        center_h, center_w = H // 2, W // 2
        radius = min(H, W) * 0.447 / 2  # 20% area = sqrt(0.2) radius
        
        # Create coordinate grids
        y_coords = torch.arange(H, device=x.device, dtype=x.dtype).view(-1, 1) - center_h
        x_coords = torch.arange(W, device=x.device, dtype=x.dtype).view(1, -1) - center_w
        dist_sq = y_coords ** 2 + x_coords ** 2
        
        # Create mask (1 inside radius, 0 outside)
        mask = (dist_sq <= radius ** 2).float()
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Apply mask
        x_fft_filtered = x_fft_shifted * mask
        
        # Shift back and compute IFFT
        x_fft_ishifted = torch.fft.ifftshift(x_fft_filtered, dim=(-2, -1))
        x_filtered = torch.fft.ifft2(x_fft_ishifted, norm='ortho')
        
        # Take real part (should be real anyway, but FFT may have small imaginary parts)
        return x_filtered.real

    def _low_pass_filter_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Apply low-pass filter to 2D data.
        
        For 2D data, we treat each sample as a 1D signal and apply FFT filtering.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, D)
            
        Returns
        -------
        torch.Tensor
            Low-pass filtered data with same shape
        """
        B, D = x.shape
        
        # Compute FFT
        x_fft = torch.fft.fft(x, norm='ortho')
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=-1)
        
        # Create mask: keep center 20% of frequencies
        center = D // 2
        keep_width = int(D * 0.447)  # sqrt(0.2) ≈ 0.447
        start_idx = center - keep_width // 2
        end_idx = center + keep_width // 2
        
        mask = torch.zeros_like(x_fft_shifted)
        mask[:, start_idx:end_idx] = 1.0
        
        # Apply mask
        x_fft_filtered = x_fft_shifted * mask
        
        # Shift back and compute IFFT
        x_fft_ishifted = torch.fft.ifftshift(x_fft_filtered, dim=-1)
        x_filtered = torch.fft.ifft(x_fft_ishifted, norm='ortho')
        
        # Take real part
        return x_filtered.real

    def _downsample_nx(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """Apply Nx downsampling to the input.
        
        For image data, downsamples by a factor of N in both spatial dimensions.
        For 2D data, returns as is (downsampling not applicable).
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W) for images or (B, D) for 2D data
        factor : int
            Downsampling factor (e.g., 2 for 2x, 3 for 3x)
            
        Returns
        -------
        torch.Tensor
            Downsampled tensor. For images: (B, C, H//factor, W//factor), for 2D: same as input
        """
        if x.dim() == 4:
            # Image data: (B, C, H, W)
            return self._downsample_nx_image(x, factor)
        else:
            # For 2D or other dimensions, return as is
            return x

    def _downsample_nx_image(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """Apply Nx downsampling to image data.
        
        Uses average pooling with kernel size N and stride N for downsampling.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W)
        factor : int
            Downsampling factor (e.g., 2 for 2x, 3 for 3x)
            
        Returns
        -------
        torch.Tensor
            Downsampled image of shape (B, C, H//factor, W//factor)
        """
        import torch.nn.functional as F
        
        # Use average pooling with kernel size factor and stride factor
        # This performs Nx downsampling: each output pixel is the average of a NxN region
        x_downsampled = F.avg_pool2d(x, kernel_size=factor, stride=factor, padding=0)
        
        return x_downsampled
    
    def _downsample_2x(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2x downsampling to the input (backward compatibility).
        
        This method is kept for backward compatibility. It calls _downsample_nx with factor=2.
        """
        return self._downsample_nx(x, factor=2)

    def _init_inception_model(self):
        """Initialize pre-trained WideResNet-40 model for feature extraction.
        
        Using WideResNet-40 trained on CIFAR-10 with energy-constrained learning loss.
        This model is specifically trained for CIFAR-10 (32x32 images) and provides
        good feature representations for the dataset.
        
        The model is loaded in eval mode and its parameters are frozen.
        """
        try:
            from pytorch_ood.model import WideResNet
            
            # Load pre-trained WideResNet-40 with energy-constrained learning on CIFAR-10
            # pretrained="er-cifar10-tune" means energy-constrained learning fine-tuned on CIFAR-10
            self._feature_model = WideResNet(num_classes=10, pretrained="er-cifar10-tune")
            self._feature_model.eval()
            
            # Note: WideResNet.transform_for() returns transforms for PIL Images
            # Since we already have tensors, we'll use CIFAR-10 normalization directly
            # CIFAR-10 normalization: mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]
            self._preprocess = None  # We'll handle normalization manually
            
            # Store device for later use (initialize as None, will be set on first use)
            self._inception_device = None
            self._features = None  # Will store features from forward hook
            
        except ImportError:
            raise ImportError("pytorch_ood is required for WideResNet feature extraction. Install with: pip install pytorch-ood")
        except Exception as e:
            raise RuntimeError(f"Failed to load WideResNet-40 model: {e}")

    def _inception_feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using pre-trained WideResNet-40 model.
        
        For CIFAR-10 images (32x32), this method:
        1. Applies the model's preprocessing (normalization)
        2. Extracts features from the model before the final classification layer
        3. Returns flattened feature vectors
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W) for images
            Expected: CIFAR-10 images (B, 3, 32, 32) with values in [-1, 1]
            
        Returns
        -------
        torch.Tensor
            Feature tensor of shape (B, D) where D is the flattened feature dimension
        """
        import torch.nn.functional as F
        
        if x.dim() != 4:
            # For non-image data, return as is
            return x
        
        # Ensure feature model is on the same device as input
        if self._inception_device != x.device:
            self._feature_model = self._feature_model.to(x.device)
            self._inception_device = x.device
        
        # Normalize from [-1, 1] to [0, 1] first
        x_normalized = (x + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        
        # Apply CIFAR-10 normalization directly (since we have tensors, not PIL Images)
        # CIFAR-10 stats: mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2471, 0.2435, 0.2616], device=x.device).view(1, 3, 1, 1)
        x_preprocessed = (x_normalized - mean) / std
        
        # Extract features before the final classification layer
        with torch.no_grad():
            # Forward through the model to get features
            # We need to get features before the final fc layer
            # WideResNet typically has: conv layers -> bn -> relu -> avgpool -> fc
            # We'll extract features after avgpool but before fc
            
            # Method 1: Use forward hook to capture features
            if hasattr(self._feature_model, 'fc'):
                # Register hook to capture features before fc layer
                self._features = None
                
                def hook_fn(module, input, output):
                    self._features = input[0]  # Input to fc layer (after avgpool)
                
                handle = self._feature_model.fc.register_forward_hook(hook_fn)
                
                # Forward pass (this will trigger the hook)
                _ = self._feature_model(x_preprocessed)
                
                # Remove hook
                handle.remove()
                
                if self._features is not None:
                    features = self._features
                else:
                    # Fallback: manually extract features
                    # Forward through all layers except fc
                    features = x_preprocessed
                    for name, module in self._feature_model.named_children():
                        if name != 'fc':
                            features = module(features)
                    # Apply avgpool if not already applied
                    if features.dim() == 4:
                        features = F.adaptive_avg_pool2d(features, output_size=(1, 1))
                        features = features.view(features.size(0), -1)
            else:
                # If no fc layer, just forward through the model
                features = self._feature_model(x_preprocessed)
                # If output is 4D, apply global avgpool
                if features.dim() == 4:
                    features = F.adaptive_avg_pool2d(features, output_size=(1, 1))
                    features = features.view(features.size(0), -1)
        
        # Ensure features are flattened
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        
        return features

    @staticmethod
    def _get_zigzag_indices(h, w):
        """Get zigzag scan indices (cached).
        
        Args:
            h, w: height and width of the matrix
        
        Returns:
            indices: list of (h_idx, w_idx) tuples in zigzag order
        """
        cache_key = (h, w)
        if not hasattr(MA_ExactOT._get_zigzag_indices, '_cache'):
            MA_ExactOT._get_zigzag_indices._cache = {}
        
        if cache_key not in MA_ExactOT._get_zigzag_indices._cache:
            total_coeffs = h * w
            indices = []
            i, j = 0, 0
            direction = 1  # 1: up-right, -1: down-left
            
            while len(indices) < total_coeffs and (i < h and j < w):
                indices.append((i, j))
                
                if direction == 1:  # Moving up-right
                    if i == 0 or j == w - 1:
                        if j == w - 1:
                            i += 1
                        else:
                            j += 1
                        direction = -1
                    else:
                        i -= 1
                        j += 1
                else:  # Moving down-left
                    if j == 0 or i == h - 1:
                        if i == h - 1:
                            j += 1
                        else:
                            i += 1
                        direction = 1
                    else:
                        i += 1
                        j -= 1
            
            MA_ExactOT._get_zigzag_indices._cache[cache_key] = indices[:total_coeffs]
        
        return MA_ExactOT._get_zigzag_indices._cache[cache_key]

    def _dct_4x4_extractor(self, x: torch.Tensor) -> torch.Tensor:
        """Extract DCT 4x4 low-frequency coefficients using zigzag scanning.
        
        Applies 2D DCT to each channel, extracts first 16 coefficients (4x4) 
        using zigzag scanning, and returns flattened features.
        
        This method is batch-efficient: processes all samples in a batch at once
        per channel.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W) for images
            
        Returns
        -------
        torch.Tensor
            Feature tensor of shape (B, C * 16) with DCT 4x4 coefficients
        """
        if x.dim() != 4:
            # For non-image data, return as is
            return x
        
        B, C, H, W = x.shape
        
        # Get zigzag indices for first 16 coefficients (4x4)
        zigzag_indices = self._get_zigzag_indices(H, W)
        low_freq_indices = zigzag_indices[:16]  # First 16 coefficients (4x4)
        
        # Convert to numpy arrays for indexing
        h_indices = np.array([idx[0] for idx in low_freq_indices])
        w_indices = np.array([idx[1] for idx in low_freq_indices])
        
        # Process each channel separately (but batch all samples together)
        x_np = x.cpu().numpy()  # Move to CPU for scipy DCT
        dct_features_list = []
        
        for c in range(C):
            # Extract channel for all samples: (B, H, W)
            x_channel = x_np[:, c, :, :]
            
            # Apply 2D DCT to all samples in batch for this channel
            # dctn can handle 3D arrays: last 2 dims are spatial
            dct_2d = dctn(x_channel, norm='ortho', axes=[1, 2])  # (B, H, W)
            
            # Extract low-frequency coefficients using zigzag indices
            # Use advanced indexing: for each sample, get coefficients at zigzag positions
            dct_low = dct_2d[:, h_indices, w_indices]  # (B, 16)
            
            dct_features_list.append(torch.from_numpy(dct_low).to(x.device))
        
        # Concatenate all channels: (B, C * 16)
        dct_features = torch.cat(dct_features_list, dim=1)
        
        return dct_features

    def _dct_8x8_extractor(self, x: torch.Tensor) -> torch.Tensor:
        """Extract DCT 8x8 low-frequency coefficients using zigzag scanning.
        
        Applies 2D DCT to each channel, extracts first 64 coefficients (8x8) 
        using zigzag scanning, and returns flattened features.
        
        This method is batch-efficient: processes all samples in a batch at once
        per channel.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W) for images
            
        Returns
        -------
        torch.Tensor
            Feature tensor of shape (B, C * 64) with DCT 8x8 coefficients
        """
        if x.dim() != 4:
            # For non-image data, return as is
            return x
        
        B, C, H, W = x.shape
        
        # Get zigzag indices for first 64 coefficients (8x8)
        zigzag_indices = self._get_zigzag_indices(H, W)
        low_freq_indices = zigzag_indices[:64]  # First 64 coefficients (8x8)
        
        # Convert to numpy arrays for indexing
        h_indices = np.array([idx[0] for idx in low_freq_indices])
        w_indices = np.array([idx[1] for idx in low_freq_indices])
        
        # Process each channel separately (but batch all samples together)
        x_np = x.cpu().numpy()  # Move to CPU for scipy DCT
        dct_features_list = []
        
        for c in range(C):
            # Extract channel for all samples: (B, H, W)
            x_channel = x_np[:, c, :, :]
            
            # Apply 2D DCT to all samples in batch for this channel
            # dctn can handle 3D arrays: last 2 dims are spatial
            dct_2d = dctn(x_channel, norm='ortho', axes=[1, 2])  # (B, H, W)
            
            # Extract low-frequency coefficients using zigzag indices
            # Use advanced indexing: for each sample, get coefficients at zigzag positions
            dct_low = dct_2d[:, h_indices, w_indices]  # (B, 64)
            
            dct_features_list.append(torch.from_numpy(dct_low).to(x.device))
        
        # Concatenate all channels: (B, C * 64)
        dct_features = torch.cat(dct_features_list, dim=1)
        
        return dct_features

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        r"""
        Compute the sample xt and conditional vector field using model-aware OT.
        
        The OT plan is computed using M(x0) and M(x1), but the actual flow matching
        uses the original x0 and x1.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon
        """
        # Apply transformation M to compute OT plan
        x0_transformed = self.M(x0)
        x1_transformed = self.M(x1)
        
        # Get OT plan using transformed coordinates
        pi = self.ot_sampler.get_map(x0_transformed, x1_transformed)
        
        # Sample indices from OT plan
        import numpy as np
        i, j = self.ot_sampler.sample_map(pi, x0.shape[0], replace=True)
        
        # Convert to torch tensors if needed
        if isinstance(i, np.ndarray):
            i = torch.from_numpy(i).to(x0.device)
        if isinstance(j, np.ndarray):
            j = torch.from_numpy(j).to(x1.device)
        
        # Reorder original x0 and x1 according to OT matching
        x0_matched = x0[i]
        x1_matched = x1[j]
        
        # Use matched original samples for flow matching
        return super(ExactOptimalTransportConditionalFlowMatcher, self).sample_location_and_conditional_flow(
            x0_matched, x1_matched, t, return_noise
        )


class Bary_ExactOT(ExactOptimalTransportConditionalFlowMatcher):
    """
    Barycenter-based Optimal Transport Conditional Flow Matching.
    
    After computing OT pairing between x0 and x1, this method:
    1. For each x1, finds all x0 that match to it according to OT plan
    2. Computes the weighted average (barycenter) of these x0
    3. The weights are normalized such that sum(w_i^2) = 1 to preserve 
       noise statistics (mean=0, variance=1)
    4. Uses the barycenter x0 instead of original x0 for flow matching
    """
    
    def __init__(self, sigma: Union[float, int] = 0.0, time_sampler: str = "uniform"):
        r"""Initialize the Bary_ExactOT class.
        
        Parameters
        ----------
        sigma : Union[float, int]
            Noise parameter
        time_sampler : str
            Time sampling method: 'uniform', 'logit_normal', or 'transport_logit_normal'
        """
        super().__init__(sigma, time_sampler=time_sampler)
        
    def compute_x0_barycenter(self, x0, x1):
        """
        Compute barycenter of x0 based on OT plan with x1.
        
        For each x1_j, find all x0_i that match to it according to OT plan,
        then compute weighted average of these x0_i.
        Weights are normalized so that sum(w_i^2) = 1 to preserve noise statistics.
        
        Parameters
        ----------
        x0 : Tensor, shape (N, *dim)
            Source samples (noise)
        x1 : Tensor, shape (M, *dim)
            Target samples (images)
            
        Returns
        -------
        x0_bary : Tensor, shape (M, *dim)
            Barycenter x0 for each x1
        """
        import numpy as np
        import scipy.optimize
        
        # Get OT plan
        pi = self.ot_sampler.get_map(x0, x1)  # Shape: (N, M)
        
        # For exact OT, we can use Hungarian algorithm to get deterministic pairing
        # But for barycenter, we want to use the full plan weights
        # Convert to torch tensor
        if isinstance(pi, np.ndarray):
            pi = torch.from_numpy(pi).to(x0.device).float()
        
        # For each x1_j, compute weighted average of x0_i
        # Weight w_ij = pi[i, j] / sqrt(sum_k pi[k, j]^2) to ensure sum(w_ij^2) = 1
        x0_bary = []
        
        for j in range(x1.shape[0]):
            # Get OT plan weights for x1_j: pi[:, j]
            weights = pi[:, j]  # Shape: (N,)
            
            # Normalize weights so that sum(w_i^2) = 1
            weight_sum_sq = torch.sum(weights ** 2)
            if weight_sum_sq > 1e-8:
                weights_normalized = weights / torch.sqrt(weight_sum_sq)
            else:
                # Fallback: uniform weights
                weights_normalized = torch.ones_like(weights) / torch.sqrt(torch.tensor(float(weights.shape[0]), device=weights.device))
            
            # Compute weighted average: x0_bary[j] = sum_i(w_i * x0[i])
            # Expand weights to match x0 dimensions
            weights_expanded = weights_normalized.view(-1, *([1] * (x0.dim() - 1)))  # (N, 1, ..., 1)
            x0_bary_j = torch.sum(weights_expanded * x0, dim=0)  # (*dim,)
            x0_bary.append(x0_bary_j)
        
        x0_bary = torch.stack(x0_bary, dim=0)  # (M, *dim)
        
        return x0_bary
    
    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        r"""
        Compute the sample xt and conditional vector field using barycenter-based OT.
        
        First compute OT pairing, then compute barycenter of x0 based on OT plan,
        and use the barycenter x0 for flow matching.
        
        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn according to time_sampler
        return_noise : bool
            return the noise sample epsilon
            
        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon
        """
        # Compute barycenter of x0 based on OT plan with x1
        x0_bary = self.compute_x0_barycenter(x0, x1)  # Shape: (M, *dim) where M = x1.shape[0]
        
        # Now use barycenter x0 with x1 for flow matching
        # x0_bary and x1 should have the same batch size (M)
        if x0_bary.shape[0] != x1.shape[0]:
            raise ValueError(f"Barycenter x0 batch size ({x0_bary.shape[0]}) != x1 batch size ({x1.shape[0]})")
        
        # For transport_logit_normal, we need to compute t based on paired distances
        # For other time samplers, t will be computed in parent class
        if t is None and self.time_sampler == "transport_logit_normal":
            t = self.sample_time(x0_bary.shape[0], x0_bary.device, x0_bary, x1)
        
        return super(ExactOptimalTransportConditionalFlowMatcher, self).sample_location_and_conditional_flow(
            x0_bary, x1, t, return_noise
        )


class MAC_ExactOT(ExactOptimalTransportConditionalFlowMatcher):
    """
    Model-Aware Conditional Flow Matching using model predictions to compute pairing loss.
    
    This class pairs samples by minimizing the L2 loss between model predictions 
    v(x0, 0) and v(x1, 1) with the theoretical direction (x1 - x0).
    
    The pairing loss for a pair (x0_i, x1_j) is:
        loss = ||v(x0_i, 0) - (x1_j - x0_i)||^2 + ||v(x1_j, 1) - (x1_j - x0_i)||^2
    
    We use Hungarian algorithm to find the pairing that minimizes the total loss.
    """
    
    def __init__(self, sigma: Union[float, int] = 0.0, time_sampler: str = "uniform", model: torch.nn.Module = None):
        r"""Initialize the MAC_ExactOT class.
        
        Parameters
        ----------
        sigma : Union[float, int]
            Noise parameter
        time_sampler : str
            Time sampling method: 'uniform', 'logit_normal', or 'transport_logit_normal'
        model : torch.nn.Module, optional
            The flow model to use for computing pairing losses.
            If None, must be provided in sample_location_and_conditional_flow.
        """
        super().__init__(sigma, time_sampler=time_sampler)
        self.pairing_model = model
        
    def compute_pairing_loss_matrix(self, x0, x1, model):
        """
        Compute the pairwise loss matrix for all (x0_i, x1_j) pairs.
        
        For each pair (x0_i, x1_j), compute:
            loss = ||v(x0_i, 0) - (x1_j - x0_i)||^2 + ||v(x1_j, 1) - (x1_j - x0_i)||^2
        
        Parameters
        ----------
        x0 : Tensor, shape (N, *dim)
            Source samples
        x1 : Tensor, shape (M, *dim)
            Target samples
        model : torch.nn.Module
            The flow model to use for predictions
            
        Returns
        -------
        loss_matrix : Tensor, shape (N, M)
            Loss matrix where loss_matrix[i, j] is the loss for pairing (x0_i, x1_j)
        """
        model.eval()
        with torch.no_grad():
            N = x0.shape[0]
            M = x1.shape[0]
            
            # Theoretical direction: x1_j - x0_i for each pair (i, j)
            # Shape: (N, M, *dim)
            x0_expanded = x0.unsqueeze(1).expand(-1, M, *[-1] * (x0.dim() - 1))
            x1_expanded = x1.unsqueeze(0).expand(N, -1, *[-1] * (x1.dim() - 1))
            theoretical_direction = x1_expanded - x0_expanded  # (N, M, *dim)
            
            # Compute v(x0_i, 0) for all x0_i
            # Create time tensor t0 = 0 for all x0
            t0 = torch.zeros(N, device=x0.device)
            if x0.dim() == 4:  # Image data: model(x, t)
                v_x0_0 = model(x0, t0)  # (N, *dim)
            else:  # 2D data: model(torch.cat([x, t], dim=-1))
                v_x0_0 = model(torch.cat([x0, t0.unsqueeze(-1)], dim=-1))
            
            # Compute v(x1_j, 1) for all x1_j
            # Create time tensor t1 = 1 for all x1
            t1 = torch.ones(M, device=x1.device)
            if x1.dim() == 4:  # Image data: model(x, t)
                v_x1_1 = model(x1, t1)  # (M, *dim)
            else:  # 2D data: model(torch.cat([x, t], dim=-1))
                v_x1_1 = model(torch.cat([x1, t1.unsqueeze(-1)], dim=-1))
            
            # Expand predictions to match theoretical_direction shape
            # v_x0_0: (N, *dim) -> (N, 1, *dim) -> (N, M, *dim)
            # v_x1_1: (M, *dim) -> (1, M, *dim) -> (N, M, *dim)
            v_x0_0_expanded = v_x0_0.unsqueeze(1).expand(-1, M, *[-1] * (v_x0_0.dim() - 1))
            v_x1_1_expanded = v_x1_1.unsqueeze(0).expand(N, -1, *[-1] * (v_x1_1.dim() - 1))
            
            # Compute losses for each pair
            # Flatten spatial dimensions for L2 loss
            v_x0_0_flat = v_x0_0_expanded.reshape(N, M, -1)
            v_x1_1_flat = v_x1_1_expanded.reshape(N, M, -1)
            theoretical_direction_flat = theoretical_direction.reshape(N, M, -1)
            
            # Loss for v(x0_i, 0)
            loss_v0 = torch.sum((v_x0_0_flat - theoretical_direction_flat) ** 2, dim=-1)  # (N, M)
            
            # Loss for v(x1_j, 1)
            loss_v1 = torch.sum((v_x1_1_flat - theoretical_direction_flat) ** 2, dim=-1)  # (N, M)
            
            # Total loss for each pair
            loss_matrix = loss_v0 + loss_v1  # (N, M)
            
        return loss_matrix
    
    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False, model=None):
        r"""
        Compute the sample xt and conditional vector field using model-aware conditional pairing.
        
        The pairing is computed by minimizing the L2 loss between model predictions
        v(x0, 0) and v(x1, 1) with the theoretical direction (x1 - x0).
        
        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn according to time_sampler
        return_noise : bool
            return the noise sample epsilon
        model : torch.nn.Module, optional
            The flow model to use for computing pairing losses.
            If None, uses self.pairing_model.
            
        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon
        """
        # Use provided model or self.pairing_model
        pairing_model = model if model is not None else self.pairing_model
        
        if pairing_model is None:
            raise ValueError("Model must be provided either in __init__ or as a parameter to sample_location_and_conditional_flow")
        
        # Compute pairing loss matrix
        loss_matrix = self.compute_pairing_loss_matrix(x0, x1, pairing_model)
        
        # Use Hungarian algorithm to find optimal pairing
        import numpy as np
        import scipy.optimize
        
        # Hungarian algorithm finds minimum cost assignment
        # If N != M, we need to handle it (for now, assume N == M)
        if x0.shape[0] != x1.shape[0]:
            raise ValueError("MAC_ExactOT currently requires x0 and x1 to have the same batch size")
        
        # Solve assignment problem
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(loss_matrix.cpu().numpy())
        
        # Convert to torch tensors
        if isinstance(row_ind, np.ndarray):
            row_ind = torch.from_numpy(row_ind).to(x0.device)
        if isinstance(col_ind, np.ndarray):
            col_ind = torch.from_numpy(col_ind).to(x1.device)
        
        # Reorder x0 and x1 according to optimal pairing
        x0_paired = x0[row_ind]
        x1_paired = x1[col_ind]
        
        # For transport_logit_normal, we need to compute t based on paired distances
        # For other time samplers, t will be computed in parent class
        if t is None and self.time_sampler == "transport_logit_normal":
            t = self.sample_time(x0_paired.shape[0], x0_paired.device, x0_paired, x1_paired)
        
        return super(ExactOptimalTransportConditionalFlowMatcher, self).sample_location_and_conditional_flow(
            x0_paired, x1_paired, t, return_noise
        )
