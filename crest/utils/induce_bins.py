import numpy as np

def induce_bins(
    quantile_values : np.ndarray,
    quantile_probs  : np.ndarray,
    n_bins          : int = 10,
    alpha           : float = 0.25,
    space           : str = 'linear',
    drop_zero_width : bool = True,
) -> list[tuple[float, float]]:
    """
    Build `n_bins` bins from feature quantiles [q0, q1, ..., q100]
    using a continuous interpolation parameter `alpha` between:

        alpha = 0 -> max-min (value-space) uniform partitioning
        alpha = 1 -> quantile-space partitioning (underlying distribution) 

    Notes
    -----
    Let q[i] and q[i+1] define the i-th percentile bin, and let

        width[i] = q[i+1] - q[i]

    be that percentile bin's width in value-space.

    If percentile bins are sampled uniformly, the induced distribution matches
    the original data distribution (up to the percentile resolution). If instead
    percentile bins are sampled in proportion to their value widths, the induced
    distribution is uniform in value-space.

    This function uses the continuous family

        p_i(alpha) ~ width[i] ** (1 - alpha)

    where p_i(alpha) is the probability mass assigned to percentile bin i.

    Therefore:
        alpha = 0: p_i = width[i]  # uniform in value-space
        alpha = 1: p_i = 1         # uniform in quantile-space
        0<alpha<1: a continuous hybrid between those two extremes

    Parameters
    ----------
    quantile_values : array-like of shape (k,)
        Monotone nondecreasing quantile values.
    quantile_probs : array-like of shape (k,)
        Corresponding cumulative probabilities. Must be nondecreasing, 
        start at 0, and end at 1 - but does not need equal spacing.
    n_bins : int
        Number of output bins.
    alpha : float
        Interpolation parameter in [0, 1]. alpha=0 results in a uniform induced
        distribution, alpha=1 matches the underlying data distribution, and 
        intermediate values induce a continuous hybrid distribution.
    drop_zero_width : bool
        If True, drop bins whose left/right edges are equal (which will reduce
        the number of returned bins when quantiles repeat).

    Returns
    -------
    list[tuple[float, float]]
        A list of `(left_edge, right_edge)` bin tuples; sampling these bins
        uniformly is approximately equivalent to sampling directly from the 
        alpha-induced distribution.

    """

    q = np.array(quantile_values).flatten()
    p = np.array(quantile_probs).flatten()

    if q.shape != p.shape:
        raise ValueError('Quantile arrays must have the same shape')
    if np.any(~np.isfinite(q + p)):
        raise ValueError('Quantile arrays must contain only finite values')
    if np.any(np.diff(q + p) < 0):
        raise ValueError('Quantile arrays must be nondecreasing')
    if np.any(p < 0):
        raise ValueError('Quantile probabilities must be in the range [0,1]')
    if p[0] != 0 or p[-1] != 1:
        raise ValueError('Quantile probabilities must be in the range [0,1]'+
                        ' and have [0,1] as endpoints.')

    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError('n_bins must be a positive integer')
    if not (0. <= alpha <= 1.):
        raise ValueError('alpha must lie in [0,1]') 

    # Only calculate edges if quantiles aren't from a constant feature
    if not np.allclose(q, q[0]):
        edges = _alpha_edges(q, p, n_bins, alpha, space)
    else: edges = np.full(n_bins+1, q[0])

    # Return list of (left edge, right edge) bins
    bins = list(zip(edges[:-1], edges[1:]))
    if drop_zero_width:
        bins = [lr for lr in bins if not np.isclose(*lr)]
    return bins


def _alpha_edges(q: np.ndarray, p: np.ndarray, n_bins: int, alpha: float, space: str):
    """
    Invert the induced distribution under the assumption that each base 
    interval [q[i], q[i+1]] is uniform internally.
    """

    # Calculate the induced distribution
    pdf = _alpha_pdf(q, p, alpha)
    cdf = np.concatenate([[0.], np.cumsum(pdf)])
    cdf[-1] = 1.
    
    # Target bins span the [0, 1] fraction range
    if space == 'linear':
        targets = np.linspace(0., 1., n_bins + 1)
    elif space == 'log': 
        targets = np.logspace(-3., 0., n_bins + 1)
    else:
        raise Exception(f'Unknown {space=}; options=[linear, log]')
        
    # Find interval i such that cdf[i] <= target < cdf[i+1]
    i = np.searchsorted(cdf, targets, side="right") - 1
    i = np.clip(i, 0, len(pdf) - 1)

    sum_mass = cdf[i]
    bin_mass = np.maximum(pdf[i], 1e-6)
    q_l = q[i]
    q_r = q[i + 1]

    # Interpolate inside bin wherever there exists any probability mass
    valid = (bin_mass > 0) & (q_r > q_l)
    fracs = np.where(valid, (targets - sum_mass) / bin_mass, 0.)
    return q_l + np.clip(fracs, 0., 1.) * (q_r - q_l)


def _alpha_pdf(q: np.ndarray, p: np.ndarray, alpha: float):
    """ Normalized induced probability mass on each quantile interval """

    # Quantile widths
    dq = np.diff(q)
    dp = np.diff(p)

    def _pdf(alpha_n):
        """ m_i ~ dq[i]^(1-alpha_ * dp[i]^alpha """
        pdf = dq**(1-alpha_n) * dp**alpha_n
        return pdf / pdf.sum()

    # Linearize alpha such that it controls a percent between alpha 0 and 1
    alpha_0 = _pdf(0.)
    alpha_1 = _pdf(1.)
    d_alpha = np.abs(alpha_1 - alpha_0).sum()

    if d_alpha > 0:
        target = alpha * d_alpha
        lo, hi = 0., 1.

        # Perform a binary search to find alpha with the lowest error
        for _ in range(100):
            alpha = (hi + lo) * 0.5
            delta = np.abs(_pdf(alpha) - alpha_0).sum()

            if abs(delta - target) <= 1e-8:
                break

            if delta < target: lo = alpha
            else:              hi = alpha
    return _pdf(alpha)
