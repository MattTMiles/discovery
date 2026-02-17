import functools

import numpy as np
import jax
import jax.numpy as jnp

from . import matrix
from . import const

from . import solar


def fpc_fast(pos, gwtheta, gwphi):
    x, y, z = pos

    sin_phi = jnp.sin(gwphi)
    cos_phi = jnp.cos(gwphi)
    sin_theta = jnp.sin(gwtheta)
    cos_theta = jnp.cos(gwtheta)

    m_dot_pos = sin_phi * x - cos_phi * y
    n_dot_pos = -cos_theta * cos_phi * x - cos_theta * sin_phi * y + sin_theta * z
    omhat_dot_pos = -sin_theta * cos_phi * x - sin_theta * sin_phi * y - cos_theta * z

    denom = 1.0 + omhat_dot_pos

    fplus = 0.5 * (m_dot_pos**2 - n_dot_pos**2) / denom
    fcross = (m_dot_pos * n_dot_pos) / denom

    return fplus, fcross

def fpcmu_fast(pos, gwtheta, gwphi):
    x, y, z = pos

    sin_phi = jnp.sin(gwphi)
    cos_phi = jnp.cos(gwphi)
    sin_theta = jnp.sin(gwtheta)
    cos_theta = jnp.cos(gwtheta)

    m_dot_pos = sin_phi * x - cos_phi * y
    n_dot_pos = -cos_theta * cos_phi * x - cos_theta * sin_phi * y + sin_theta * z
    omhat_dot_pos = -sin_theta * cos_phi * x - sin_theta * sin_phi * y - cos_theta * z

    denom = 1.0 + omhat_dot_pos

    fplus = 0.5 * (m_dot_pos**2 - n_dot_pos**2) / denom
    fcross = (m_dot_pos * n_dot_pos) / denom

    return fplus, fcross, -omhat_dot_pos


def makedelay_binary(pulsarterm=True):
    def delay_binary(toas, pos, log10_h, log10_f0, ra, sindec, cosinc, psi, phi_earth, phi_psr):
        """BBH residuals from Ellis et. al 2012, 2013"""

        h0 = 10**log10_h
        f0 = 10**log10_f0

        dec, inc = jnp.arcsin(sindec), jnp.arccos(cosinc)

        # calculate antenna pattern (note: pos is pulsar sky position unit vector)
        fplus, fcross = fpc_fast(pos, 0.5 * jnp.pi - dec, ra)  # careful with dec -> gwtheta conversion

        if pulsarterm:
            phi_avg = 0.5 * (phi_earth + phi_psr)
        else:
            phi_avg = phi_earth

        tref = 86400.0 * 51544.5  # MJD J2000 in seconds

        phase = phi_avg + 2.0 * jnp.pi * f0 * (toas - tref)
        cphase, sphase = jnp.cos(phase), jnp.sin(phase)

        # fix this for no pulsarterm

        if pulsarterm:
            phi_diff = 0.5 * (phi_earth - phi_psr)
            sin_diff = jnp.sin(phi_diff)

            delta_sin =  2.0 * cphase * sin_diff
            delta_cos = -2.0 * sphase * sin_diff
        else:
            delta_sin = sphase
            delta_cos = cphase

        At = -1.0 * (1.0 + jnp.cos(inc)**2) * delta_sin
        Bt =  2.0 * jnp.cos(inc) * delta_cos

        alpha = h0 / (2 * jnp.pi * f0)

        # calculate rplus and rcross
        rplus  = alpha * (-At * jnp.cos(2 * psi) + Bt * jnp.sin(2 * psi))
        rcross = alpha * ( At * jnp.sin(2 * psi) + Bt * jnp.cos(2 * psi))

        # calculate residuals
        res = -fplus * rplus - fcross * rcross

        return res

    if not pulsarterm:
        delay_binary = functools.partial(delay_binary, phi_psr=jnp.nan)

    return delay_binary


def makedelay_binary_phases(pulsarterm=True):
    """Factory for computing cphase and sphase vectors from binary parameters."""
    def delay_binary_phases(toas, log10_f0):
        """Compute cosine and sine phase vectors.

        Returns:
            cphase: cosine phase vector (toas,)
            sphase: sine phase vector (toas,)
        """
        f0 = 10**log10_f0
        tref = 86400.0 * 51544.5  # MJD J2000 in seconds

        # Compute base phase vectors from frequency evolution only
        phase_base = 2.0 * jnp.pi * f0 * (toas - tref)
        cphase = jnp.cos(phase_base)
        sphase = jnp.sin(phase_base)

        return jnp.array([cphase, sphase])

    return delay_binary_phases


def makedelay_binary_coefficients(pulsarterm=True):
    """Factory for computing coefficients to multiply phase vectors."""
    def delay_binary_coefficients(pos, log10_h0, log10_f0, ra, sindec, cosinc, psi, phi_earth, phi_psr):
        """Compute antenna pattern factors and phase coefficients.

        Returns:
            coeffs: dictionary with keys:
                - 'fplus', 'fcross': antenna pattern factors
                - 'rplus_coeff_c', 'rplus_coeff_s': coefficients for cphase/sphase in rplus
                - 'rcross_coeff_c', 'rcross_coeff_s': coefficients for cphase/sphase in rcross

            Full response is reconstructed as:
            res = -fplus * (rplus_coeff_c * cphase + rplus_coeff_s * sphase)
                  -fcross * (rcross_coeff_c * cphase + rcross_coeff_s * sphase)
        """
        h0 = 10**log10_h0
        f0 = 10**log10_f0

        dec, inc = jnp.arcsin(sindec), jnp.arccos(cosinc)

        # calculate antenna pattern (note: pos is pulsar sky position unit vector)
        fplus, fcross = fpc_fast(pos, 0.5 * jnp.pi - dec, ra)  # careful with dec -> gwtheta conversion

        # Calculate coefficients that multiply cphase and sphase
        # Apply addition theorem: cos(phi_avg + phase_base) = cos(phi_avg)*cos(phase_base) - sin(phi_avg)*sin(phase_base)
        # Original cphase_orig = cos(phi_avg + phase_base)
        # Original sphase_orig = sin(phi_avg + phase_base)
        if pulsarterm:
            phi_avg = 0.5 * (phi_earth + phi_psr)
            phi_diff = 0.5 * (phi_earth - phi_psr)

            cos_avg = jnp.cos(phi_avg)
            sin_avg = jnp.sin(phi_avg)
            sin_diff = jnp.sin(phi_diff)

            # cphase_orig = cos_avg * cphase - sin_avg * sphase
            # sphase_orig = sin_avg * cphase + cos_avg * sphase
            # delta_sin =  2.0 * cphase_orig * sin_diff
            # delta_cos = -2.0 * sphase_orig * sin_diff

            c_coeff_sin = 2.0 * cos_avg * sin_diff    # coefficient for cphase in delta_sin
            s_coeff_sin = -2.0 * sin_avg * sin_diff   # coefficient for sphase in delta_sin
            c_coeff_cos = -2.0 * sin_avg * sin_diff   # coefficient for cphase in delta_cos
            s_coeff_cos = -2.0 * cos_avg * sin_diff   # coefficient for sphase in delta_cos
        else:
            # cphase_orig = cos(phi_earth + phase_base) = cos(phi_earth)*cphase - sin(phi_earth)*sphase
            # sphase_orig = sin(phi_earth + phase_base) = sin(phi_earth)*cphase + cos(phi_earth)*sphase
            # delta_sin = sphase_orig
            # delta_cos = cphase_orig
            cos_earth = jnp.cos(phi_earth)
            sin_earth = jnp.sin(phi_earth)

            c_coeff_sin = sin_earth    # coefficient for cphase in delta_sin
            s_coeff_sin = cos_earth    # coefficient for sphase in delta_sin
            c_coeff_cos = cos_earth    # coefficient for cphase in delta_cos
            s_coeff_cos = -sin_earth   # coefficient for sphase in delta_cos

        # At = -1.0 * (1.0 + cos(inc)^2) * delta_sin
        # Bt = 2.0 * cos(inc) * delta_cos
        cos_inc = jnp.cos(inc)
        At_coeff_c = -1.0 * (1.0 + cos_inc**2) * c_coeff_sin
        At_coeff_s = -1.0 * (1.0 + cos_inc**2) * s_coeff_sin
        Bt_coeff_c = 2.0 * cos_inc * c_coeff_cos
        Bt_coeff_s = 2.0 * cos_inc * s_coeff_cos

        alpha = h0 / (2 * jnp.pi * f0)
        cos_2psi = jnp.cos(2 * psi)
        sin_2psi = jnp.sin(2 * psi)

        # rplus = alpha * (-At * cos(2*psi) + Bt * sin(2*psi))
        # rcross = alpha * (At * sin(2*psi) + Bt * cos(2*psi))

        # Coefficient for cphase in rplus: alpha * (-At_coeff_c * cos(2*psi) + Bt_coeff_c * sin(2*psi))
        rplus_coeff_c = alpha * (-At_coeff_c * cos_2psi + Bt_coeff_c * sin_2psi)
        # Coefficient for sphase in rplus: alpha * (-At_coeff_s * cos(2*psi) + Bt_coeff_s * sin(2*psi))
        rplus_coeff_s = alpha * (-At_coeff_s * cos_2psi + Bt_coeff_s * sin_2psi)

        # Coefficient for cphase in rcross: alpha * (At_coeff_c * sin(2*psi) + Bt_coeff_c * cos(2*psi))
        rcross_coeff_c = alpha * (At_coeff_c * sin_2psi + Bt_coeff_c * cos_2psi)
        # Coefficient for sphase in rcross: alpha * (At_coeff_s * sin(2*psi) + Bt_coeff_s * cos(2*psi))
        rcross_coeff_s = alpha * (At_coeff_s * sin_2psi + Bt_coeff_s * cos_2psi)

        Ac = -fplus * rplus_coeff_c - fcross * rcross_coeff_c
        As = -fplus * rplus_coeff_s - fcross * rcross_coeff_s

        return jnp.array([Ac, As])

    if not pulsarterm:
        delay_binary_coefficients = functools.partial(delay_binary_coefficients, phi_psr=jnp.nan)

    return delay_binary_coefficients



def cos2comp(f, df, A, f0, phi, t0):
    """Project signal A * cos(2pi f t + phi) onto Fourier basis
    cos(2pi k t/T), sin(2pi k t/T) for t in [t0, t0+T]."""

    T = 1.0 / df[0]

    Delta_omega = 2.0 * jnp.pi * (f0 - f[::2])
    Sigma_omega = 2.0 * jnp.pi * (f0 + f[::2])

    phase_Delta_start = phi + Delta_omega * t0
    phase_Delta_end   = phi + Delta_omega * (t0 + T)

    phase_Sigma_start = phi + Sigma_omega * t0
    phase_Sigma_end   = phi + Sigma_omega * (t0 + T)

    ck = (A / T) * (
        (jnp.sin(phase_Delta_end) - jnp.sin(phase_Delta_start)) / Delta_omega +
        (jnp.sin(phase_Sigma_end) - jnp.sin(phase_Sigma_start)) / Sigma_omega
    )

    sk = (A / T) * (
        (jnp.cos(phase_Delta_end) - jnp.cos(phase_Delta_start)) / Delta_omega -
        (jnp.cos(phase_Sigma_end) - jnp.cos(phase_Sigma_start)) / Sigma_omega
    )

    return jnp.stack((sk, ck), axis=1).reshape(-1)


def makefourier_binary(pulsarterm=True):
    def fourier_binary(f, df, mintoa, pos, log10_h, log10_f0, ra, sindec, cos_inc, psi, phi_earth, phi_psr):
        """BBH residuals from Ellis et. al 2012, 2013"""

        h0 = 10**log10_h
        f0 = 10**log10_f0

        dec, inc = jnp.arcsin(sindec), jnp.arccos(cos_inc)

        # calculate antenna pattern (note: pos is pulsar sky position unit vector)
        fplus, fcross = fpc_fast(pos, 0.5 * jnp.pi - dec, ra)  # careful with dec -> gwtheta conversion

        if pulsarterm:
            phi_avg  = 0.5 * (phi_earth + phi_psr)
        else:
            phi_avg = phi_earth

        tref = 86400.0 * 51544.5  # MJD J2000 in seconds

        cphase = cos2comp(f, df, 1.0, f0, phi_avg - 2.0 * jnp.pi * f0 * tref, mintoa)
        sphase = cos2comp(f, df, 1.0, f0, phi_avg - 2.0 * jnp.pi * f0 * tref - 0.5*jnp.pi, mintoa)

        # fix this for no pulsarterm

        if pulsarterm:
            phi_diff = 0.5 * (phi_earth - phi_psr)
            sin_diff = jnp.sin(phi_diff)

            delta_sin =  2.0 * cphase * sin_diff
            delta_cos = -2.0 * sphase * sin_diff
        else:
            delta_sin = sphase
            delta_cos = cphase

        At = -1.0 * (1.0 + jnp.cos(inc)**2) * delta_sin
        Bt =  2.0 * jnp.cos(inc) * delta_cos

        alpha = h0 / (2 * jnp.pi * f0)

        # calculate rplus and rcross
        rplus  = alpha * (-At * jnp.cos(2 * psi) + Bt * jnp.sin(2 * psi))
        rcross = alpha * ( At * jnp.sin(2 * psi) + Bt * jnp.cos(2 * psi))

        # calculate residuals
        res = -fplus * rplus - fcross * rcross

        return res

    if not pulsarterm:
        fourier_binary = functools.partial(fourier_binary, phi_psr=jnp.nan)

    return fourier_binary


def makefourier_binary_pdist(pulsarterm=True):
    def fourier_binary_pdist(f, df, mintoa, pos, log10_h, log10_f0, ra, sindec, cos_inc, psi, phi_earth, p_dist):
        h0 = 10**log10_h
        f0 = 10**log10_f0

        pos = jnp.array(pos)
        
        dec, inc = jnp.arcsin(sindec), jnp.arccos(cos_inc)

        # calculate antenna pattern
        fplus, fcross = fpc_fast(pos, 0.5 * jnp.pi - dec, ra)

        c = 2.99792458e8 
        omega_hat = jnp.array([ -jnp.cos(dec) * jnp.cos(ra), 
                                -jnp.cos(dec) * jnp.sin(ra),
                                -jnp.sin(dec)
                              ])

        # Convert pulsar distance from kpc to meters to match c [m/s]
        p_dist_m = p_dist * 3.0856775814913673e19  # 1 kpc in meters
        phi_psr = (p_dist_m / c) * 2.0 * jnp.pi * f0  * (1.0 + jnp.dot(omega_hat, pos))

        if pulsarterm:
            phi_avg = 0.5 * (phi_earth + phi_psr)
        else:
            phi_avg = phi_earth

        tref = 86400.0 * 51544.5  # MJD J2000 in seconds

        cphase = cos2comp(f, df, 1.0, f0, phi_avg - 2.0 * jnp.pi * f0 * tref, mintoa)
        sphase = cos2comp(f, df, 1.0, f0, phi_avg - 2.0 * jnp.pi * f0 * tref - 0.5 * jnp.pi, mintoa)

        if pulsarterm:
            phi_diff = 0.5 * (phi_earth - phi_psr)
            sin_diff = jnp.sin(phi_diff)

            delta_sin =  2.0 * cphase * sin_diff
            delta_cos = -2.0 * sphase * sin_diff
        else:
            delta_sin = sphase
            delta_cos = cphase

        At = -1.0 * (1.0 + jnp.cos(inc)**2) * delta_sin
        Bt =  2.0 * jnp.cos(inc) * delta_cos

        alpha = h0 / (2 * jnp.pi * f0)

        rplus  = alpha * (-At * jnp.cos(2 * psi) + Bt * jnp.sin(2 * psi))
        rcross = alpha * ( At * jnp.sin(2 * psi) + Bt * jnp.cos(2 * psi))

        res = -fplus * rplus - fcross * rcross

        return res

    if not pulsarterm:
        fourier_binary_pdist = functools.partial(fourier_binary_pdist, p_dist=jnp.nan)

    return fourier_binary_pdist

def makefourier_binary_pdist_twoD(pulsarterm=True):
    def fourier_binary_pdist_twoD(f, df, mintoa, pos, log10_h, log10_f0, ra, sindec, cos_inc, psi, phi_earth, 
                             log10_h_2, log10_f0_2, ra_2, sindec_2, cos_inc_2, psi_2, phi_earth_2, p_dist):

        h0 = 10**log10_h
        f0 = 10**log10_f0

        h0_2 = 10**log10_h_2
        f0_2 = 10**log10_f0_2

        pos = jnp.array(pos)

        dec, inc = jnp.arcsin(sindec), jnp.arccos(cos_inc)
        dec_2, inc_2 = jnp.arcsin(sindec_2), jnp.arccos(cos_inc_2)

        # calculate antenna pattern
        fplus, fcross = fpc_fast(pos, 0.5 * jnp.pi - dec, ra)
        fplus_2, fcross_2 = fpc_fast(pos, 0.5 * jnp.pi - dec_2, ra_2)

        c = 2.99792458e8 
        omega_hat = jnp.array([ -jnp.cos(dec) * jnp.cos(ra), 
                                -jnp.cos(dec) * jnp.sin(ra),
                                -jnp.sin(dec)
                              ])
        omega_hat_2 = jnp.array([ -jnp.cos(dec_2) * jnp.cos(ra_2), 
                                -jnp.cos(dec_2) * jnp.sin(ra_2),
                                -jnp.sin(dec_2)
                              ])

        # Convert pulsar distance from kpc to meters to match c [m/s]
        p_dist_m = p_dist * 3.0856775814913673e19  # 1 kpc in meters
        phi_psr = (p_dist_m / c) * 2.0 * jnp.pi * f0  * (1.0 + jnp.dot(omega_hat, pos))

        phi_psr_2 = (p_dist_m / c) * 2.0 * jnp.pi * f0_2  * (1.0 + jnp.dot(omega_hat_2, pos))


        if pulsarterm:
            phi_avg = 0.5 * (phi_earth + phi_psr)
            phi_avg_2 = 0.5 * (phi_earth_2 + phi_psr_2)
        else:
            phi_avg = phi_earth
            phi_avg_2 = phi_earth_2

        tref = 86400.0 * 51544.5  # MJD J2000 in seconds

        cphase = cos2comp(f, df, 1.0, f0, phi_avg - 2.0 * jnp.pi * f0 * tref, mintoa)
        sphase = cos2comp(f, df, 1.0, f0, phi_avg - 2.0 * jnp.pi * f0 * tref - 0.5 * jnp.pi, mintoa)

        cphase_2 = cos2comp(f, df, 1.0, f0_2, phi_avg_2 - 2.0 * jnp.pi * f0_2 * tref, mintoa)
        sphase_2 = cos2comp(f, df, 1.0, f0_2, phi_avg_2 - 2.0 * jnp.pi * f0_2 * tref - 0.5 * jnp.pi, mintoa)

        if pulsarterm:
            phi_diff = 0.5 * (phi_earth - phi_psr)
            sin_diff = jnp.sin(phi_diff)

            delta_sin =  2.0 * cphase * sin_diff
            delta_cos = -2.0 * sphase * sin_diff

            phi_diff_2 = 0.5 * (phi_earth_2 - phi_psr_2)
            sin_diff_2 = jnp.sin(phi_diff_2)

            delta_sin_2 =  2.0 * cphase_2 * sin_diff_2
            delta_cos_2 = -2.0 * sphase_2 * sin_diff_2
        else:
            delta_sin = sphase
            delta_cos = cphase

            delta_sin_2 = sphase_2
            delta_cos_2 = cphase_2

        At = -1.0 * (1.0 + jnp.cos(inc)**2) * delta_sin
        Bt =  2.0 * jnp.cos(inc) * delta_cos

        At_2 = -1.0 * (1.0 + jnp.cos(inc_2)**2) * delta_sin_2
        Bt_2 =  2.0 * jnp.cos(inc_2) * delta_cos_2

        alpha = h0 / (2 * jnp.pi * f0)

        alpha_2 = h0_2 / (2 * jnp.pi * f0_2)

        rplus  = alpha * (-At * jnp.cos(2 * psi) + Bt * jnp.sin(2 * psi))
        rcross = alpha * ( At * jnp.sin(2 * psi) + Bt * jnp.cos(2 * psi))


        rplus_2  = alpha_2 * (-At_2 * jnp.cos(2 * psi_2) + Bt_2 * jnp.sin(2 * psi_2))
        rcross_2 = alpha_2 * ( At_2 * jnp.sin(2 * psi_2) + Bt_2 * jnp.cos(2 * psi_2))

        res = -fplus * rplus - fcross * rcross
        res2 = -fplus_2 * rplus_2 - fcross_2 * rcross_2

        return res + res2

    if not pulsarterm:
        fourier_binary_pdist_twoD = functools.partial(fourier_binary_pdist_twoD, p_dist=jnp.nan)

    return fourier_binary_pdist_twoD

def make_phase_connected_binary(pulsarterm=True, evolve=True):
    @functools.partial(jax.jit, static_argnames=("psr_term", "evolve"))
    def phase_connected_binary(
        toas,
        pos,
        cos_gwtheta,
        gwphi,
        cos_inc,
        log10_mc,
        log10_fgw,
        log10_h,
        phase0,
        psi,
        p_dist,
        psr_term=pulsarterm,
        evolve=evolve,
        p_phase=None,
        log10_dist=None,
    ):

        toas = jnp.asarray(toas)
        pos = jnp.asarray(pos)
        # pdist = jnp.asarray(pdist)

        mc = (10.0 ** log10_mc) * const.Tsun
        w0 = jnp.pi * (10.0 ** log10_fgw)
        gwtheta = jnp.arccos(cos_gwtheta)
        inc = jnp.arccos(cos_inc)
        phase0_orb = 0.5 * phase0  # convert GW phase to orbital phase

        # Determine distance or strain
        if (log10_h is None) == (log10_dist is None):
            raise ValueError("Provide exactly one of log10_dist or log10_h")
        if log10_h is None:
            dist = (10.0 ** log10_dist) * const.Mpc / const.c
        else:
            dist = 2.0 * mc ** (5.0 / 3.0) * w0 ** (2.0 / 3.0) / (10.0 ** log10_h)

        fplus, fcross, cos_mu = fpcmu_fast(pos, gwtheta, gwphi)
        tref = 86400.0 * 51544.5  # MJD J2000 in seconds 
        toas_rel = toas - tref
        ## This won't work with how I'm sampling it. I don't have a Gaussian prior on p_dist yet
        #parallax_coeff = const.kpc / const.c * (pdist[0] + pdist[1] * p_dist)
        parallax_coeff = const.kpc / const.c * p_dist
        tp = toas_rel - parallax_coeff * (1.0 - cos_mu)
        tp = jnp.where(psr_term, tp, toas_rel)

        # Frequency/phase evolution
        def evolve_phase(t, p_phase):
            term = 1.0 - (256.0 / 5.0) * mc ** (5.0 / 3.0) * w0 ** (8.0 / 3.0) * t
            omega = w0 * jnp.power(term, -3.0 / 8.0)
            if p_phase is None:
                phase = phase0_orb + (1.0 / (32.0 * mc ** (5.0 / 3.0))) * (
                    w0 ** (-5.0 / 3.0) - omega ** (-5.0 / 3.0)
                )
            else:
                phase = phase0_orb + p_phase + (1.0 / (32.0 * mc ** (5.0 / 3.0))) * (
                    w0 ** (-5.0 / 3.0) - omega ** (-5.0 / 3.0)
                )
            return omega, phase
        
        omega, phase = evolve_phase(toas_rel, p_phase=None) if evolve else (w0, w0 * toas_rel + phase0_orb)
        omega_p, phase_p = evolve_phase(tp, p_phase) if evolve else (w0, w0 * tp + phase0_orb)

        At = -0.5 * jnp.sin(2.0 * phase) * (3.0 + jnp.cos(2.0 * inc))
        Bt = 2.0 * jnp.cos(2.0 * phase) * jnp.cos(inc)
        At_p = -0.5 * jnp.sin(2.0 * phase_p) * (3.0 + jnp.cos(2.0 * inc))
        Bt_p = 2.0 * jnp.cos(2.0 * phase_p) * jnp.cos(inc)

        alpha = mc ** (5.0 / 3.0) / (dist * omega ** (1.0 / 3.0))
        alpha_p = mc ** (5.0 / 3.0) / (dist * omega_p ** (1.0 / 3.0))

        rplus = alpha * (-At * jnp.cos(2.0 * psi) + Bt * jnp.sin(2.0 * psi))
        rcross = alpha * (At * jnp.sin(2.0 * psi) + Bt * jnp.cos(2.0 * psi))
        rplus_p = alpha_p * (-At_p * jnp.cos(2.0 * psi) + Bt_p * jnp.sin(2.0 * psi))
        rcross_p = alpha_p * (At_p * jnp.sin(2.0 * psi) + Bt_p * jnp.cos(2.0 * psi))


        return jnp.where(
            psr_term,
            fplus * (rplus_p - rplus) + fcross * (rcross_p - rcross),
            -fplus * rplus - fcross * rcross,
        )
    
    if pulsarterm:
        if evolve:
            return functools.partial(phase_connected_binary, psr_term=True, evolve=evolve)
        return functools.partial(phase_connected_binary, psr_term=True, evolve=False)
    else:
        if evolve:
            return functools.partial(phase_connected_binary, psr_term=False, p_dist=0.0, evolve=evolve)
        return functools.partial(phase_connected_binary, psr_term=False, p_dist=0.0, evolve=False)

def make_phase_unconnected_binary(pulsarterm=True):
    @functools.partial(jax.jit, static_argnames=("psr_term", "evolve"))
    def phase_unconnected_binary(
        toas,
        pos,
        cos_gwtheta,
        gwphi,
        cos_inc,
        log10_mc,
        log10_fgw,
        log10_h,
        phase0,
        psi,
        p_dist,
        p_phase,
        psr_term=pulsarterm,
        evolve=True,
        log10_dist=None,
    ):

        toas = jnp.asarray(toas)
        pos = jnp.asarray(pos)
        # pdist = jnp.asarray(pdist)

        mc = (10.0 ** log10_mc) * const.Tsun
        w0 = jnp.pi * (10.0 ** log10_fgw)
        gwtheta = jnp.arccos(cos_gwtheta)
        inc = jnp.arccos(cos_inc)
        phase0_orb = 0.5 * phase0  # convert GW phase to orbital phase

        # Determine distance or strain
        if (log10_h is None) == (log10_dist is None):
            raise ValueError("Provide exactly one of log10_dist or log10_h")
        if log10_h is None:
            dist = (10.0 ** log10_dist) * const.Mpc / const.c
        else:
            dist = 2.0 * mc ** (5.0 / 3.0) * w0 ** (2.0 / 3.0) / (10.0 ** log10_h)

        fplus, fcross, cos_mu = fpcmu_fast(pos, gwtheta, gwphi)
        tref = 86400.0 * 51544.5  # MJD J2000 in seconds 
        toas_rel = toas - tref
        ## This won't work with how I'm sampling it. I don't have a Gaussian prior on p_dist yet
        #parallax_coeff = const.kpc / const.c * (pdist[0] + pdist[1] * p_dist)
        parallax_coeff = const.kpc / const.c * p_dist
        tp = toas_rel - parallax_coeff * (1.0 - cos_mu)
        tp = jnp.where(psr_term, tp, toas_rel)

        # Frequency/phase evolution
        def evolve_phase(t, p_phase):
            term = 1.0 - (256.0 / 5.0) * mc ** (5.0 / 3.0) * w0 ** (8.0 / 3.0) * t
            omega = w0 * jnp.power(term, -3.0 / 8.0)
            if p_phase is None:
                phase = phase0_orb + (1.0 / (32.0 * mc ** (5.0 / 3.0))) * (
                    w0 ** (-5.0 / 3.0) - omega ** (-5.0 / 3.0)
                )
            else:
                phase = phase0_orb + p_phase + (1.0 / (32.0 * mc ** (5.0 / 3.0))) * (
                    w0 ** (-5.0 / 3.0) - omega ** (-5.0 / 3.0)
                )
            return omega, phase

        omega, phase = evolve_phase(toas_rel, p_phase=None) if evolve else (w0, w0 * toas_rel + phase0_orb)
        omega_p, phase_p = evolve_phase(tp, p_phase) if evolve else (w0, w0 * tp + phase0_orb)

        At = -0.5 * jnp.sin(2.0 * phase) * (3.0 + jnp.cos(2.0 * inc))
        Bt = 2.0 * jnp.cos(2.0 * phase) * jnp.cos(inc)
        At_p = -0.5 * jnp.sin(2.0 * phase_p) * (3.0 + jnp.cos(2.0 * inc))
        Bt_p = 2.0 * jnp.cos(2.0 * phase_p) * jnp.cos(inc)

        alpha = mc ** (5.0 / 3.0) / (dist * omega ** (1.0 / 3.0))
        alpha_p = mc ** (5.0 / 3.0) / (dist * omega_p ** (1.0 / 3.0))

        rplus = alpha * (-At * jnp.cos(2.0 * psi) + Bt * jnp.sin(2.0 * psi))
        rcross = alpha * (At * jnp.sin(2.0 * psi) + Bt * jnp.cos(2.0 * psi))
        rplus_p = alpha_p * (-At_p * jnp.cos(2.0 * psi) + Bt_p * jnp.sin(2.0 * psi))
        rcross_p = alpha_p * (At_p * jnp.sin(2.0 * psi) + Bt_p * jnp.cos(2.0 * psi))


        return jnp.where(
            psr_term,
            fplus * (rplus_p - rplus) + fcross * (rcross_p - rcross),
            -fplus * rplus - fcross * rcross,
        )
    
    if pulsarterm:
        return functools.partial(phase_unconnected_binary, psr_term=True)
    else:
        return functools.partial(phase_unconnected_binary, psr_term=False, p_dist=0.0)


def chromatic_exponential(psr, fref=1400.0):
    """Chromatic exponential delay model."""
    toas, fnorm = matrix.jnparray(psr.toas / const.day), matrix.jnparray(fref / psr.freqs)

    def delay(t0, log10_Amp, log10_tau, sign_param, alpha):
        return jnp.sign(sign_param) * 10**log10_Amp * jnp.exp(- (toas - t0) / 10**log10_tau) * fnorm**alpha * jnp.heaviside(toas - t0, 1.0)

    return delay


def chromatic_annual(psr, fref=1400.0):
    """Chromatic annual delay model."""
    toas, fnorm = matrix.jnparray(psr.toas), matrix.jnparray(fref / psr.freqs)

    def delay(log10_Amp, phase, alpha):
        return 10**log10_Amp * jnp.sin(2*jnp.pi * const.fyr * toas + phase) * fnorm**alpha

    return delay


def chromatic_gaussian(psr, fref=1400.0):
    """Chromatic Gaussian delay model."""
    toas, fnorm = matrix.jnparray(psr.toas / const.day), matrix.jnparray(fref / psr.freqs)

    def delay(t0, log10_Amp, log10_sigma, sign_param, alpha):
        return jnp.sign(sign_param) * 10**log10_Amp * jnp.exp(-(toas - t0)**2 / (2 * (10**log10_sigma)**2)) * fnorm**alpha

    return delay


def sw_deterministic(psr):
    """Deterministic solar wind delay model."""
    theta, R_earth, _, _ = solar.theta_impact(psr)
    def delay(n_earth):
        
        dm_sol_wind = solar.dm_solar(n_earth, theta, R_earth)
        
        return (dm_sol_wind) * 4.148808e3 / psr.freqs**2

    return delay

def orthometric_shapiro(psr, binphase):
    """Orthometric Shapiro delay model from Freire & Wex (2010)."""
    toas, binphase = matrix.jnparray(psr.toas / const.day), matrix.jnparray(binphase)
    if not np.shape(binphase) == np.shape(toas):
        raise ValueError("Input binphase must have the same shape as toas")

    def delay(h3, stig):
        return -(2.0 * h3 / stig**3) * jnp.log(1 + stig**2 - 2 * stig * jnp.sin(binphase))

    return delay
