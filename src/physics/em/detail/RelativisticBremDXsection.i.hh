//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RelativisticBremDXsection.i.hh
//---------------------------------------------------------------------------//
#include "RelativisticBremDXsection.hh"

#include <cmath>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION RelativisticBremDXsection::RelativisticBremDXsection(
    const RelativisticBremNativeRef& shared,
    const ParticleTrackView&         particle,
    const MaterialView&              material,
    const ElementComponentId&        elcomp_id)
    : shared_(shared)
    , elem_data_(shared.elem_data[material.element_id(elcomp_id)])
    , total_energy_(particle.energy().value() + shared.electron_mass.value())
{
    real_type density_factor = material.electron_density()
                               * this->migdal_constant();
    density_corr_ = density_factor * ipow<2>(total_energy_);

    lpm_energy_ = material.radiation_length() * this->lpm_constant();

    real_type lpm_threshold = lpm_energy_ * std::sqrt(density_factor);
    enable_lpm_ = (shared_.enable_lpm && (total_energy_ > lpm_threshold));
}

//---------------------------------------------------------------------------//
/*!
 * Compute the relativistic differential cross section per atom at the given
 * bremsstrahlung photon energy in MeV.
 */
CELER_FUNCTION
real_type RelativisticBremDXsection::operator()(real_type energy)
{
    CELER_EXPECT(energy > 0);
    return (enable_lpm_) ? this->dxsec_per_atom_lpm(energy)
                         : this->dxsec_per_atom(energy);
}

//---------------------------------------------------------------------------//
/*!
 * Compute the differential cross section without the LPM effect.
 */
CELER_FUNCTION
real_type RelativisticBremDXsection::dxsec_per_atom(real_type gamma_energy)
{
    real_type dxsec{0};

    real_type y     = gamma_energy / total_energy_;
    real_type onemy = 1 - y;
    real_type term0 = onemy + R(0.75) * ipow<2>(y);

    if (elem_data_.iz < 5)
    {
        // The Dirac-Fock model
        dxsec = term0 * elem_data_.factor1 + onemy * elem_data_.factor2;
    }
    else
    {
        // Tsai's analytical approximation.
        real_type invz    = 1 / static_cast<real_type>(elem_data_.iz);
        real_type term1   = y / (total_energy_ - gamma_energy);
        real_type gamma   = term1 * elem_data_.gamma_factor;
        real_type epsilon = term1 * elem_data_.epsilon_factor;

        // Evaluate the screening functions
        auto sfunc = compute_screen_functions(gamma, epsilon);

        dxsec
            = term0
                  * ((R(0.25) * sfunc.phi1 - elem_data_.fz)
                     + (R(0.25) * sfunc.psi1 - 2 * elem_data_.logz / 3) * invz)
              + R(0.125) * onemy * (sfunc.phi2 + sfunc.psi2 * invz);
    }

    return max(dxsec, R(0));
}

//---------------------------------------------------------------------------//
/*!
 * Compute the differential cross section with the LPM effect.
 */
CELER_FUNCTION
real_type RelativisticBremDXsection::dxsec_per_atom_lpm(real_type gamma_energy)
{
    real_type y     = gamma_energy / total_energy_;
    real_type onemy = 1 - y;
    real_type y2    = R(0.25) * ipow<2>(y);

    // Evaluate LPM functions
    auto lpm = compute_lpm_functions(gamma_energy);

    real_type term  = lpm.xis * (y2 * lpm.gs + (onemy + 2 * y2) * lpm.phis);
    real_type dxsec = term * elem_data_.factor1 + onemy * elem_data_.factor2;

    return max(dxsec, R(0));
}

//---------------------------------------------------------------------------//
/*!
 * Compute screen_functions: Tsai's analytical approximations of coherent and
 * incoherent screening function to the numerical screening functions computed
 * by using the Thomas-Fermi model: Y.-S.Tsai, Rev. Mod. Phys. 49 (1977) 421.
 */
auto RelativisticBremDXsection::compute_screen_functions(real_type gam,
                                                         real_type eps)
    -> ScreenFunctions
{
    ScreenFunctions func;
    real_type       gam2 = ipow<2>(gam);
    real_type       eps2 = ipow<2>(eps);

    func.phi1 = R(16.863) - 2 * std::log(1 + R(0.311877) * gam2)
                + R(2.4) * std::exp(R(-0.9) * gam)
                + R(1.6) * std::exp(R(-1.5) * gam);
    func.phi2 = 2 / (3 + R(19.5) * gam + 18 * gam2);

    func.psi1 = R(24.34) - 2 * std::log(1 + R(13.111641) * eps2)
                + R(2.8) * std::exp(R(-8) * eps)
                + R(1.2) * std::exp(R(-29.2) * eps);
    func.psi2 = 2 / (3 + 120 * eps + 1200 * eps2);

    return func;
}

//---------------------------------------------------------------------------//
/*!
 * Compute the LPM suppression functions.
 */
auto RelativisticBremDXsection::compute_lpm_functions(real_type egamma)
    -> LPMFunctions
{
    LPMFunctions func;

    // Calcuate s
    real_type y = egamma / total_energy_;

    real_type sprime
        = std::sqrt(R(0.125) * y * lpm_energy_ / (total_energy_ - egamma));

    real_type xi_sprime{2};

    if (sprime > 1)
    {
        xi_sprime = 1;
    }
    else if (sprime > constants::sqrt_two * elem_data_.s1)
    {
        real_type inv_logs2 = elem_data_.inv_logs2;
        real_type h_sprime  = std::log(sprime) * inv_logs2;
        xi_sprime           = 1 + h_sprime
                    - R(0.08) * (1 - h_sprime) * h_sprime * (2 - h_sprime)
                          * inv_logs2;
    }

    real_type s = sprime / std::sqrt(xi_sprime);

    // Include a dielectric suppression effect into s according to Migdal
    real_type shat = s * (1 + density_corr_ / ipow<2>(egamma));

    // Calculate xi(s)

    func.xis = 2;
    if (shat > 1)
    {
        func.xis = 1;
    }
    else if (shat > elem_data_.s1)
    {
        func.xis = 1 + std::log(shat) * elem_data_.inv_logs1;
    }

    // Evaluate G(s) and phi(s): G4eBremsstrahlungRelModel::GetLPMFunctions
    if (shat < shared_.limit_s_lpm())
    {
        real_type val = shat * shared_.inv_delta_lpm();

        CELER_ASSERT(val >= 0);
        size_type ilow = static_cast<size_type>(val);
        val -= ilow;
        CELER_ASSERT(ilow + 1 < shared_.lpm_table.size());

        auto linterp = [val](real_type lo, real_type hi) {
            return lo + (hi - lo) * val;
        };

        auto get_table = [ilow, this](size_type offset) {
            return this->shared_.lpm_table[ItemIdT{ilow + offset}];
        };

        func.gs   = linterp(get_table(0).gs, get_table(1).gs);
        func.phis = linterp(get_table(0).phis, get_table(1).phis);
    }
    else
    {
        real_type ss = ipow<4>(shat);
        func.phis    = 1 - R(0.01190476) / ss;
        func.gs      = 1 - R(0.0230655) / ss;
    }

    // Make sure suppression is smaller than 1: Migdal's approximation on xi
    if (func.xis * func.phis > 1 || shat > R(0.57))
    {
        func.xis = 1 / func.phis;
    }

    return func;
}
//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
