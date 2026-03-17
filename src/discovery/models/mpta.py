import numpy as np
import jax.numpy as jnp
import re

from .. import matrix
from .. import signals
from .. import prior
from .. import solar
from .. import likelihood
from .. import deterministic
from .. import const

def write_ml_json(df, savename):
    import json
    ml_idx = df['logl'].idxmax()
    ml_params = df.loc[ml_idx].to_dict()
    with open(savename, 'w') as f:
        json.dump(ml_params, f, indent=2)
    return

def powerlaw_bkgrnd(f, df): # fixed amplitude and slope for a GWB at A = 2*10**-15
    return ((2 * 10**(-15))**2) / 12.0 / jnp.pi**2 * const.fyr ** (4.33 - 3.0) * f ** (-4.33) * df

def update_priordict_standard_mpta():
    # Update the standard prior dictionary with PTA-specific parameters
    prior.priordict_standard.update({
        # White noise parameters
        '(.*_)?efac':               [0.5, 2],
        '(.*_)?log10_tnequad':      [-10, -5],
        '(.*_)?log10_ecorr':        [-10, -5],
        # Per-pulsar GW background parameters
        '(.*_)?bkgrnd_log10_A':     [-18, -11],
        '(.*_)?gw_log10_A':     [-18, -11],
        # GP parameters
        '(.*_)?red_noise_log10_A.*':  [-18, -11],
        '(.*_)?red_noise_gamma.*':    [0, 7],
        '(.*_)?red_noise2_log10_A.*':  [-18, -11],
        '(.*_)?red_noise2_gamma.*':    [0, 7],
        '(.*_)?dm_gp_log10_A':      [-18, -11],
        '(.*_)?dm_gp_gamma':        [0, 7],
        '(.*_)?chrom_gp_log10_A':   [-18, -11],
        '(.*_)?chrom_gp_gamma':     [0, 7],
        '(.*_)?chrom_gp_alpha':     [2.5, 14],
        '(.*_)?sw_gp_log10_A':      [-10, -2],
        '(.*_)?sw_gp_gamma':        [0, 4],
        '(.*_)?band_gp_log10_A':    [-18, -11],
        '(.*_)?band_gp_gamma':      [0, 7],
        '(.*_)?band_low_gp_log10_A':    [-18, -11],
        '(.*_)?band_low_gp_gamma':      [0, 7],
        '(.*_)?band_low_gp_fcutoff':    [856, 1712], # MeerKAT L-band
        '(.*_)?band_gp_flow':       [856, 1712], # MeerKAT L-band
        '(.*_)?band_gp_fhigh':      [856, 1712], # MeerKAT L-band
        '(.*_)?bandalpha_gp_log10_A':    [-18, -11],
        '(.*_)?bandalpha_gp_gamma':      [0, 7],
        '(.*_)?bandalpha_gp_alpha':      [0, 10],
        '(.*_)?bandalpha_gp_fcutoff':    [856, 1712], # MeerKAT L-band
        '(.*_)?bandalpha_gp_fhigh':    [856, 1712], # MeerKAT L-band
        '(.*_)?bandalpha_gp_flow':    [856, 1712], # MeerKAT L-band
        # common noise parameters
        'curn_log10_A':             [-18, -11],
        'curn_gamma':               [0, 7],
        # deterministic parameters
        '(.*_)?chrom_exp_t0': [58525, 60900], # MPTA 6-yr range
        '(.*_)?chrom_exp_log10_Amp': [-10, -4],
        '(.*_)?chrom_exp_log10_tau': [0, 4],
        '(.*_)?chrom_exp_sign_param': [-1, 1],
        '(.*_)?chrom_exp_alpha': [0, 7],
        '(.*_)?chrom_1yr_log10_Amp': [-10, -4],
        '(.*_)?chrom_1yr_phase': [0, 2 * np.pi],
        '(.*_)?chrom_1yr_alpha': [0, 7],
        '(.*_)?chrom_gauss_t0': [58525, 60900], # MPTA 6-yr range
        '(.*_)?chrom_gauss_log10_Amp': [-10, -4],
        '(.*_)?chrom_gauss_log10_sigma': [0, 4],
        '(.*_)?chrom_gauss_sign_param': [-1, 1],
        '(.*_)?chrom_gauss_alpha': [0, 7],
        r'(.*_)?timingmodel_coefficients\(\d+\)': [-20.0, 20.0],
        r'(.*_)?alpha_scaling\(\d+\)': [0.0, 100.0],
        r'(.*_)?h3': [0.0, 10**-5],
        r'(.*_)?stig': [0.0, 1.0]
    })
    return

update_priordict_standard_mpta() # Ensure priordict_standard is updated on import, but also update when a model is created to catch any changes during likelihood/prior initialisation

def _get_F_ncols(gp):
    """Get the number of columns in a GP's F matrix, whether static or callable."""
    if callable(gp.F):
        return sum(s.stop - s.start for s in gp.index.values())
    else:
        return gp.F.shape[1]


def gps2commongp(gps):
    """Merge per-pulsar compound GPs into a single vectorised VariableGP.

    Works for both static F (numpy arrays) and callable F (parameter-dependent,
    e.g. chromatic or band-noise GPs). When any GP has callable F, the per-pulsar
    F entries are wrapped so they are all padded to the same column count.

    Handles both 1D diagonal priors (fourier basis) and 2D dense priors (fftcov basis).
    """
    priors = [gp.Phi.getN for gp in gps]
    pmax = len(gps)
    ns = [_get_F_ncols(gp) for gp in gps]
    nmax = max(ns)

    # Detect whether priors are 1D (diagonal) or 2D (dense covariance)
    is_2d = isinstance(gps[0].Phi, matrix.NoiseMatrix2D_var)

    if is_2d:
        # 2D priors: each p(params) returns (n_i, n_i), pad to (nmax, nmax)
        def prior(params):
            yp = matrix.jnp.full((pmax, nmax, nmax), 1e-40) * matrix.jnp.eye(nmax)
            for i, p in enumerate(priors):
                yp = yp.at[i, :ns[i], :ns[i]].set(p(params))
            return yp
        prior.params = sorted(set([par for p in priors for par in p.params]))
    else:
        # 1D priors: each p(params) returns (n_i,), pad to (nmax,)
        def prior(params):
            yp = matrix.jnp.full((pmax, nmax), 1e-40)
            for i, p in enumerate(priors):
                yp = yp.at[i, :ns[i]].set(p(params))
            return yp
        prior.params = sorted(set([par for p in priors for par in p.params]))

    has_callable = any(callable(gp.F) for gp in gps)

    if has_callable:
        # Build per-pulsar F entries, wrapping callables so they zero-pad to nmax cols
        Fs = []
        F_params_all = []
        for gp, n in zip(gps, ns):
            if callable(gp.F):
                # Capture gp.F and n in a closure
                def _make_padded(Ffunc, ncols):
                    pad_width = nmax - ncols
                    def padded_F(params):
                        return jnp.pad(Ffunc(params), [(0, 0), (0, pad_width)])
                    padded_F.params = Ffunc.params
                    return padded_F
                Fs.append(_make_padded(gp.F, n))
                F_params_all.extend(gp.F.params)
            else:
                Fs.append(np.pad(gp.F, [(0, 0), (0, nmax - n)]))
    else:
        Fs = [np.pad(gp.F, [(0, 0), (0, nmax - gp.F.shape[1])]) for gp in gps]

    if is_2d:
        result = matrix.VariableGP(matrix.VectorNoiseMatrix2D_var(prior), Fs)
        # Store per-pulsar priors and column counts so VectorWoodburyKernel_varFP can
        # process pulsars sequentially, avoiding the (npsr, nmax, nmax) batched tensor.
        result._per_pulsar_priors = priors
        result._per_pulsar_ns = ns
    else:
        result = matrix.VariableGP(matrix.VectorNoiseMatrix1D_var(prior), Fs)
    result._has_callable_F = has_callable
    return result


def make_psr_gps_fourier(psr, max_cadence_days=14, Tspan=None, background=True, red=True, red2=False, dm=True, chrom=True, sw=True, band=False, band_low=False, band_alpha=False):
    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))

    return (([signals.makegp_fourier(psr, matrix.partial(signals.powerlaw, gamma=13/3.), components=psr_components, name='gw')] if background else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, name='red_noise')] if red else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, name='red_noise2')] if red2 else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_dm, name='dm_gp')] if dm else [])+ \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_chrom, name='chrom_gp')] if chrom else [])+ \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=solar.fourierbasis_solar_dm, name='sw_gp')] if sw else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_band_range, name='band_gp')] if band else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_band, name='band_low_gp')] if band_low else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_band_range_alpha, name='bandalpha_gp')] if band_alpha else []))


def make_psr_gps_fftint(psr, max_cadence_days=14, Tspan=None, background=True, red=True, red2=False, dm=True, chrom=True, sw=True, band=False, band_low=False, band_alpha=False):
    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))
    psr_knots = 2 * psr_components + 1

    return (([signals.makegp_fftcov(psr, matrix.partial(signals.powerlaw, gamma=13/3.), components=psr_knots, name='gw')] if background else []) + \
            ([signals.makegp_fftcov(psr, signals.powerlaw, components=psr_knots, name='red_noise')] if red else []) + \
            ([signals.makegp_fftcov(psr, signals.powerlaw, components=psr_knots, name='red_noise2')] if red2 else []) + \
            ([signals.makegp_fftcov_dm(psr, signals.powerlaw, components=psr_knots, name='dm_gp')] if dm else [])+ \
            ([signals.makegp_fftcov_chrom(psr, signals.powerlaw, components=psr_knots, name='chrom_gp')] if chrom else [])+ \
            ([signals.makegp_fftcov_solar(psr, signals.powerlaw, components=psr_knots, name='sw_gp')] if sw else []) + \
            ([signals.makegp_fftcov_band_range(psr, signals.powerlaw, components=psr_knots, name='band_gp')] if band else []) + \
            ([signals.makegp_fftcov_band(psr, signals.powerlaw, components=psr_knots, name='band_low_gp')] if band_low else []) + \
            ([signals.makegp_fftcov_band_range_alpha(psr, signals.powerlaw, components=psr_knots, name='bandalpha_gp')] if band_alpha else []))


def make_common_gps_fftint(psrs, common_knots=61, max_cadence_days=14, Tspan=None,
                           background=True, red=True, red2=False, dm=True, chrom=True,
                           sw=True, band=False, band_low=False, band_alpha=False):
    """Merge per-pulsar GPs into a single vectorised GP for use in ArrayLikelihood.

    Works for all GP types including those with callable F (chromatic, band, etc.).
    """
    if Tspan is None:
        Tspan = signals.getspan(psrs)
    return gps2commongp([matrix.CompoundGP(
        make_psr_gps_fftint(psr, max_cadence_days=max_cadence_days, Tspan=Tspan,
                            background=background, red=red, red2=red2, dm=dm, chrom=chrom,
                            sw=sw, band=band, band_low=band_low, band_alpha=band_alpha))
        for psr in psrs])


def make_common_gps_fourier(psrs, max_cadence_days=14, Tspan=None,
                            background=True, red=True, red2=False, dm=True, chrom=True,
                            sw=True, band=False, band_low=False, band_alpha=False):
    """Fourier-basis variant of make_common_gps_fftint."""
    if Tspan is None:
        Tspan = signals.getspan(psrs)
    return gps2commongp([matrix.CompoundGP(
        make_psr_gps_fourier(psr, max_cadence_days=max_cadence_days, Tspan=Tspan,
                             background=background, red=red, red2=red2, dm=dm, chrom=chrom,
                             sw=sw, band=band, band_low=band_low, band_alpha=band_alpha))
        for psr in psrs])

def single_pulsar_noise(psr, fftint=True, max_cadence_days=14, Tspan=None, noisedict={}, tm_variable=False, timing_inds=None, outliers=False, global_ecorr=False,
                        background=True, red=True, red2=False, dm=True, chrom=True, sw=True, band=False, band_low=False, band_alpha=False, # GP models
                        chrom_annual=False, chrom_exponential=False, chrom_gaussian=False, shapiro=False, extra_gps=None): # Deterministic chromatic models
    # Set up per-backend white noise
    measurement_noise = signals.makenoise_measurement(psr, tnequad=True, noisedict=noisedict, outliers=outliers)
    # Set up timing model
    tm = signals.makegp_timing(psr, svd=True, variable=tm_variable, timing_inds=timing_inds)
    if not isinstance(tm, list): # ensure the timing model is unpacked if returning a list
        tm = [tm]
    # Set up model components
    model_components = [psr.residuals]
    model_components += tm
    model_components += [measurement_noise]
    model_components += [signals.makegp_ecorr(psr, noisedict=noisedict)]
    if global_ecorr: # add an additional global ECORR term
        model_components += [signals.makegp_ecorr_simple(psr, noisedict=noisedict)]
    # Add deterministic chromatic components
    if chrom_annual:
        model_components += [signals.makedelay(psr, deterministic.chromatic_annual(psr), name='chrom_1yr')]
    if chrom_exponential:
        model_components += [signals.makedelay(psr, deterministic.chromatic_exponential(psr), name='chrom_exp')]
    if chrom_gaussian:
        model_components += [signals.makedelay(psr, deterministic.chromatic_gaussian(psr), name='chrom_gauss')]
    if shapiro:
        tasc = 59000.033444485320853 * 86400 # Example value for J2241-5236
        pb = 1 / 7.9452845656629659959e-05 # Example value for J2241-5236
        binphase = (2 * np.pi / pb) * (psr.toas - tasc)
        print("warning: using example values for tasc and pb in shapiro delay model")
        model_components += [signals.makedelay(psr, deterministic.orthometric_shapiro(psr, binphase), name='shapiro')]
    # Add GP components
    if fftint:
        model_components += make_psr_gps_fftint(psr, max_cadence_days=max_cadence_days,Tspan=Tspan, background=background, red=red, red2=red2, dm=dm, chrom=chrom, sw=sw, band=band, band_low=band_low, band_alpha=band_alpha)
    else:
        model_components += make_psr_gps_fourier(psr, max_cadence_days=max_cadence_days, Tspan=Tspan, background=background, red=red, red2=red2, dm=dm, chrom=chrom, sw=sw, band=band, band_low=band_low, band_alpha=band_alpha)
    if extra_gps is not None:
        model_components += extra_gps

    comp_params = []
    for comp in model_components:
        if hasattr(comp, 'params'):
            comp_params.extend(comp.params)

    m = likelihood.PulsarLikelihood(model_components)
    m.all_params.extend(comp_params)
    m.logL.params = sorted(set(m.all_params))

    return m

def common_noise(psrs, chain_dfs, fftInt=False, max_cadence_days=14, name="gw_crn"):
    # Accepts a list of pulsars and their corresponding chain dataframes and constructs an ArrayLikelihood
    def has_param(df, param_string="red_noise"):
        return any(f"{param_string}" in col for col in list(df.columns))

    Tspan = signals.getspan(psrs)
    common_components = int(Tspan / (max_cadence_days * 86400))
    common_knots = 2 * common_components + 1

    psls = []
    for psr, df in zip(psrs, chain_dfs):
        if not any(psr.name in col for col in df.columns):
            raise ValueError("Chain data frames do not match pulsar names")
        # Get max-likelihood parameters for this pulsar
        ml_idx = df['logl'].idxmax()
        noisedict = {col: df.loc[ml_idx, col] for col in df.columns if col.startswith(psr.name)}

        if not fftInt:
            curn = signals.makegp_fourier(psr, signals.powerlaw, common_components, Tspan, common=['curn_log10_A', 'curn_gamma'], name='curn')
        else:
            curn = signals.makegp_fftcov(psr, signals.powerlaw, common_knots, Tspan, common=['curn_log10_A', 'curn_gamma'], name='curn')
        extra_gps = curn if isinstance(curn, list) else [curn]


        # background = False, as we are including a common red noise process
        m = single_pulsar_noise(psr, fftint=fftInt, max_cadence_days=max_cadence_days, Tspan=None, background=False, noisedict=noisedict, global_ecorr=has_param(df, f"{psr.name}_ecorr"),
                                red=has_param(df, "red_noise"), dm=has_param(df, "dm_gp"), chrom=has_param(df, "chrom_gp"), sw=has_param(df, "sw_gp"),
                                band=has_param(df, "band_gp"), band_low=has_param(df, "band_low_gp"), band_alpha=has_param(df, "bandalpha_gp"),
                                chrom_annual=has_param(df, "chrom_1yr"), chrom_exponential=has_param(df, "chrom_exp"), chrom_gaussian=has_param(df, "chrom_gauss"),
                                extra_gps=extra_gps)

        print("Including pulsar", psr.name, "with model parameters:\n", m.logL.params)
        psls.append(m)

    return likelihood.GlobalLikelihood(psls)
    # return likelihood.ArrayLikelihood(psls)


def common_noise_array(psrs, chain_dfs, fftInt=False, max_cadence_days=14, name="gw_crn"):
    """Like common_noise, but uses ArrayLikelihood + gps2commongp for faster vectorised evaluation.

    All per-pulsar GPs (including those with callable F such as chromatic/band) are merged
    into a single vectorised GP with a common basis, which enables batched Woodbury algebra.
    """
    def has_param(df, param_string="red_noise"):
        return any(f"{param_string}" in col for col in list(df.columns))

    Tspan = signals.getspan(psrs)
    common_components = int(Tspan / (max_cadence_days * 86400))
    common_knots = 2 * common_components + 1

    pslmodels = []
    per_psr_gps = []
    for psr, df in zip(psrs, chain_dfs):
        if not any(psr.name in col for col in df.columns):
            raise ValueError("Chain data frames do not match pulsar names")
        ml_idx = df['logl'].idxmax()
        noisedict = {col: df.loc[ml_idx, col] for col in df.columns if col.startswith(psr.name)}

        # White noise + timing model (goes into PulsarLikelihood)
        measurement_noise = signals.makenoise_measurement(psr, tnequad=True, noisedict=noisedict)
        tm = signals.makegp_timing(psr, svd=True)
        if not isinstance(tm, list):
            tm = [tm]
        base_components = [psr.residuals] + tm + [measurement_noise]
        base_components += [signals.makegp_ecorr(psr, noisedict=noisedict)]
        if has_param(df, f"{psr.name}_ecorr"):
            base_components += [signals.makegp_ecorr_simple(psr, noisedict=noisedict)]

        psl = likelihood.PulsarLikelihood(base_components)
        pslmodels.append(psl)

        # Per-pulsar noise GPs + CURN (go into commongp)
        gp_kwargs = dict(
            max_cadence_days=max_cadence_days, Tspan=Tspan, background=False,
            red=has_param(df, "red_noise"),
            dm=has_param(df, "dm_gp"),
            chrom=has_param(df, "chrom_gp"),
            sw=has_param(df, "sw_gp"),
            band=has_param(df, "band_gp"),
            band_low=has_param(df, "band_low_gp"),
            band_alpha=has_param(df, "bandalpha_gp"),
        )
        if fftInt:
            gps = make_psr_gps_fftint(psr, **gp_kwargs)
            curn = signals.makegp_fftcov(psr, signals.powerlaw, common_knots, Tspan,
                                         common=['curn_log10_A', 'curn_gamma'], name='curn')
        else:
            gps = make_psr_gps_fourier(psr, **gp_kwargs)
            curn = signals.makegp_fourier(psr, signals.powerlaw, common_components, Tspan,
                                          common=['curn_log10_A', 'curn_gamma'], name='curn')
        curn_list = curn if isinstance(curn, list) else [curn]
        per_psr_gps.append(matrix.CompoundGP(gps + curn_list))

        print("Including pulsar", psr.name, "in ArrayLikelihood")

    commongp = gps2commongp(per_psr_gps)
    return likelihood.ArrayLikelihood(pslmodels, commongp=commongp)
