import inspect
import numpy as np
import pandas as pd

import numpyro
from numpyro import infer
from numpyro import distributions as dist

from .. import prior


def makemodel_transformed(mylogl, transform=prior.makelogtransform_uniform, priordict={}):
    logx = transform(mylogl, priordict=priordict)

    parlen = sum(int(par[par.index('(')+1:par.index(')')]) if '(' in par else 1 for par in logx.params)

    def numpyro_model():
        pars = numpyro.sample('pars', dist.Normal(0, 10).expand([parlen]))
        logl = logx.logL(pars)
        numpyro.deterministic('log_likelihood', logl)
        numpyro.factor('logl', logl + logx.logprior(pars))
    numpyro_model.to_df = lambda chain: logx.to_df(chain['pars'])

    return numpyro_model


def makemodel(mylogl, priordict={}):
    def numpyro_model():
        logl = mylogl({par: numpyro.sample(par, dist.Uniform(*prior.getprior_uniform(par, priordict)))
                       for par in mylogl.params})
        numpyro.deterministic('log_likelihood', logl)
        numpyro.factor('logl', logl)
    numpyro_model.to_df = lambda chain: pd.DataFrame(chain)

    return numpyro_model


def makemodel_laplace(mylogl, u_init, u_sigma, transform=prior.makelogtransform_uniform, priordict={}):
    """Like makemodel_transformed but with Laplace whitening in tanh-transformed parameter space.

    The numpyro model samples v ~ N(0,1) and maps u = u_init + u_sigma * v before
    evaluating the log-posterior.  For common/unwhitened parameters set u_init=0,
    u_sigma=10 (reproducing the standard N(0,10) sampling used by makemodel_transformed).
    For Laplace-whitened parameters set u_init to the per-pulsar MAP in tanh-space and
    u_sigma to the corresponding posterior std.  Use laplace_per_pulsar() to compute
    these arrays automatically from a list of per-pulsar log-likelihood functions.

    Parameters
    ----------
    mylogl : callable with .params
        Full log-likelihood function.
    u_init : array-like, shape (parlen,)
        Centre of the whitened distribution in tanh-space  (0 for unoptimised params).
    u_sigma : array-like, shape (parlen,)
        Scale of the whitened distribution in tanh-space  (10 for unoptimised params).
    """
    import numpy as _np
    logx = transform(mylogl, priordict=priordict)

    u_init_arr  = np.array(u_init,  dtype=float)
    u_sigma_arr = np.array(u_sigma, dtype=float)
    u_init_jnp  = prior.jnp.array(u_init_arr)
    u_sigma_jnp = prior.jnp.array(u_sigma_arr)
    parlen = len(u_init_arr)

    def numpyro_model():
        v    = numpyro.sample('pars', dist.Normal(0, 1).expand([parlen]))
        pars = u_init_jnp + u_sigma_jnp * v
        logl = logx.logL(pars)
        numpyro.deterministic('log_likelihood', logl)
        numpyro.factor('logl', logl + logx.logprior(pars))

    # chain['pars'] contains v; un-whiten to get u before passing to logx.to_df
    numpyro_model.to_df = (
        lambda chain, _u0=u_init_arr, _us=u_sigma_arr:
            logx.to_df(_np.array(chain['pars']) * _us + _u0)
    )

    return numpyro_model


def laplace_per_pulsar(logL_func, per_pulsar_logls, priordict={}, verbose=True):
    """Run per-pulsar Laplace optimisation and return whitening arrays for makemodel_laplace.

    For each single-pulsar log-likelihood in *per_pulsar_logls*, this runs a BFGS
    optimisation in the tanh-transformed parameter space and extracts the MAP estimate
    and diagonal posterior standard deviations from the BFGS inverse-Hessian approximation.
    These are used to centre and scale NUTS, dramatically reducing warmup for well-
    constrained per-pulsar noise parameters.

    Common parameters that do not appear in any per-pulsar model (e.g. CURN amplitude
    and spectral index) are left with u_init=0, u_sigma=10, which reproduces the
    standard N(0,10) prior used by makemodel_transformed.

    Parameters
    ----------
    logL_func : callable with .params
        Full PTA log-likelihood.  Defines the parameter ordering used by the
        returned arrays.
    per_pulsar_logls : list of callables with .params
        Per-pulsar log-likelihood functions built from single-pulsar noise models
        (i.e. without the CURN).  Each function's .params must be a subset of
        logL_func.params.
    priordict : dict
        Forwarded to makelogtransform_uniform for both the full and per-pulsar models.
    verbose : bool
        Print per-pulsar optimisation progress.

    Returns
    -------
    u_init : jnp.array, shape (parlen,)
    u_sigma : jnp.array, shape (parlen,)
    """
    import jax
    import jax.numpy as jnp_local

    # --- helpers ---------------------------------------------------------------
    def _make_slices(params):
        """Map each param name to its slice (or scalar index) in the u-vector."""
        slices, offset = {}, 0
        for par in params:
            if '(' in par:
                l = int(par[par.index('(')+1:par.index(')')])
                slices[par] = slice(offset, offset + l)
                offset += l
            else:
                slices[par] = offset
                offset += 1
        return slices, offset

    # Param ordering and total parlen for the full model
    full_slices, parlen = _make_slices(logL_func.params)

    u_init  = np.zeros(parlen)
    u_sigma = np.full(parlen, 10.0)  # default: reproduce N(0,10)

    for idx, psr_logL in enumerate(per_pulsar_logls):
        psr_name = psr_logL.params[0].split('_')[0] if psr_logL.params else f'pulsar_{idx}'
        if verbose:
            print(f"  Laplace [{idx+1}/{len(per_pulsar_logls)}] {psr_name} "
                  f"({len(psr_logL.params)} params) ...", flush=True)

        logx_psr = prior.makelogtransform_uniform(psr_logL, priordict=priordict)
        psr_slices, psr_parlen = _make_slices(psr_logL.params)

        u0 = jnp_local.zeros(psr_parlen)
        neg_logpost = lambda u: -(logx_psr.logL(u) + logx_psr.logprior(u))

        try:
            result = jax.scipy.optimize.minimize(neg_logpost, u0, method='BFGS',
                                                 options={'maxiter': 2000})
            if verbose:
                status = 'converged' if result.success else f'status={result.status}'
                print(f"    {status}, logpost={-float(result.fun):.2f}", flush=True)

            u_hat = np.array(result.x)
            # Compute exact Hessian at the mode rather than using the BFGS
            # inverse-Hessian approximation, which can be inaccurate when the
            # optimiser converges quickly or directions are poorly explored.
            H = np.array(jax.hessian(neg_logpost)(result.x))
            cov = np.linalg.inv(H)
            sigma_psr = np.sqrt(np.clip(np.diag(cov), 1e-6, None))
            sigma_psr = np.clip(sigma_psr, 0.02, 10.0)  # guard against ill-conditioning

        except Exception as e:
            if verbose:
                print(f"    optimisation failed ({e}), using defaults", flush=True)
            u_hat     = np.zeros(psr_parlen)
            sigma_psr = np.full(psr_parlen, 10.0)

        # Map per-pulsar param slices into the full-model u vectors
        for par in psr_logL.params:
            if par not in full_slices:
                continue
            ps = psr_slices[par]
            fs = full_slices[par]
            u_init[fs]  = u_hat[ps]
            u_sigma[fs] = sigma_psr[ps]

    return jnp_local.array(u_init), jnp_local.array(u_sigma)


def makesampler_nuts(numpyro_model, num_warmup=512, num_samples=1024, num_chains=1, **kwargs):
    nuts_keys = set(inspect.getfullargspec(infer.NUTS).args)
    mcmc_keys = set(inspect.getfullargspec(infer.MCMC).kwonlyargs)

    nutsargs = dict(max_tree_depth=8, dense_mass=False,
                    forward_mode_differentiation=False, target_accept_prob=0.8)
    nutsargs.update({k: v for k, v in kwargs.items() if k in nuts_keys})

    mcmcargs = dict(num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                    chain_method='vectorized', progress_bar=True)
    mcmcargs.update({k: v for k, v in kwargs.items() if k in mcmc_keys})

    sampler = infer.MCMC(infer.NUTS(numpyro_model, **nutsargs), **mcmcargs)

    def _to_df():
        samples = sampler.get_samples()

        df = numpyro_model.to_df(samples)

        if 'log_likelihood' in samples:
            df = df.drop(columns=['log_likelihood'], errors='ignore')
            df['logl'] = np.asarray(samples['log_likelihood'])

        return df

    def _make_plots(save_name=None, diagnostics=False):
        import matplotlib.pyplot as plt
        import corner
        import re

        df = sampler.to_df()
        reserved = [r'^logl$', r'^(.*_)?alpha_scaling\[\d+\]$'] # don't plot likelihood or outlier parameters
        labels = [c for c in df.columns if not any(re.match(r, c) for r in reserved)]
        data = df[labels].values

        fig = corner.corner(
            data,
            labels=labels,
            show_titles=True,
            title_fmt=".2f",
            title_kwargs={"fontsize": 10},
            label_kwargs={"fontsize": 9},
            plot_datapoints=True,
            hist_kwargs={"color": "C0"},
            contour_kwargs={"colors": ["C0"]},
        )
        plt.tight_layout()
        if save_name:
            plt.savefig(f"{save_name}_corner.png")
        plt.close()

    sampler.to_df = _to_df
    sampler.make_plots = _make_plots

    return sampler
