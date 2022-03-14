//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LPMCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "base/Algorithms.hh"
#include "base/Constants.hh"
#include "base/Macros.hh"
#include "base/Quantity.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"
#include "physics/em/LPMData.hh"
#include "physics/material/MaterialView.hh"
#include "physics/material/Types.hh"

#include "PhysicsConstants.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the Landau-Pomeranchuk-Migdal (LPM) suppression functions.
 *
 * The LPM effect is the suppression of low-energy photon production due to
 * electron multiple scattering. At high energies and in high density
 * materials, the cross sections for pair production and bremsstrahlung are
 * reduced. The differential cross sections accounting for the LPM effect are
 * expressed in terms of the LPM suppression functions \f$ xi(s) \f$, \f$ G(s)
 * \f$, and \f$ \phi(s) \f$.
 *
 * See section 10.2.2 of the Geant4 Physics Reference Manual (Release 10.7).
 */
class LPMCalculator
{
  public:
    //!@{
    //! Type aliases
    using LPMDataRef = LPMData<Ownership::const_reference, MemSpace::native>;
    //!@}

    //! LPM suppression functions
    struct LPMFunctions
    {
        real_type xi;
        real_type g;
        real_type phi;
    };

  public:
    // Construct with LPM data, material data, and photon energy
    inline CELER_FUNCTION LPMCalculator(const LPMDataRef&   shared,
                                        const MaterialView& material,
                                        const ElementView&  element,
                                        bool dielectric_suppression,
                                        units::MevEnergy gamma_energy);

    // Compute the LPM supression functions
    inline CELER_FUNCTION LPMFunctions operator()(real_type epsilon);

  private:
    //// DATA ////

    // Shared LPM data
    const LPMDataRef& shared_;
    // Current element
    const ElementView& element_;
    // Electron density of the current material [1/cm^3]
    const real_type electron_density_;
    // Characteristic energy for the LPM effect for this material [MeV]
    const real_type lpm_energy_;
    // Include a dielectric suppression effect
    const bool dielectric_suppression_;
    // Photon energy [MeV]
    const real_type gamma_energy_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with LPM data, material data, and photon energy.
 */
CELER_FUNCTION
LPMCalculator::LPMCalculator(const LPMDataRef&   shared,
                             const MaterialView& material,
                             const ElementView&  element,
                             bool                dielectric_suppression,
                             units::MevEnergy    gamma_energy)
    : shared_(shared)
    , element_(element)
    , electron_density_(material.electron_density())
    , lpm_energy_(material.radiation_length()
                  * value_as<MevPerCm>(lpm_constant()))
    , dielectric_suppression_(dielectric_suppression)
    , gamma_energy_(gamma_energy.value())
{
    CELER_EXPECT(gamma_energy_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Compute the LPM suppression functions.
 *
 * Here \f$ \epsilon \f$ is the ratio of the  electron (or positron) energy to
 * the photon energy, \f$ \epsilon = E / k \f$.
 *
 * See section 10.2.2 of the Geant4 Physics Reference Manual and
 * ComputeLPMfunctions and GetLPMFunctions in G4eBremsstrahlungRelModel and
 * G4PairProductionRelModel.
 */
CELER_FUNCTION auto LPMCalculator::operator()(real_type epsilon) -> LPMFunctions
{
    LPMFunctions result;

    // Suppression variable \f$ s' \f$. For bremsstrahlung \f$ s' =
    // \sqrt{\frac{E_\textrm{LPM} k}{8 E (E - k)}} \f$, and for pair production
    // \f$ s' = \sqrt{\frac{E_\textrm{LPM} k}{8 E (k - E)}} \f$, where \f$ k \$
    // is the photon energy and \f$ E \f$ is the electon (or positron) energy
    const real_type s_prime = std::sqrt(
        lpm_energy_ / (8 * epsilon * gamma_energy_ * std::fabs(epsilon - 1)));

    // TODO: In the Geant4 relativistic pair production model the denominator
    // is 184 instead of 184.15 -- why? Will it matter?
    const real_type s1 = ipow<2>(element_.cbrt_z() / real_type(184.15));

    // Calculate \f$ \xi(s') \f$ and \f$ s = \frac{s'}{\sqrt{\xi(s')}} \f$
    result.xi = 2;
    if (s_prime > 1)
    {
        result.xi = 1;
    }
    else if (s_prime > constants::sqrt_two * s1)
    {
        const real_type log_s1 = std::log(constants::sqrt_two * s1);
        const real_type h      = std::log(s_prime) / log_s1;
        result.xi = 1 + h - real_type(0.08) * (1 - h) * h * (2 - h) / log_s1;
    }
    real_type s = s_prime / std::sqrt(result.xi);

    if (dielectric_suppression_)
    {
        // Include a dielectric suppression effect in \f$ s \f$ according to
        // Migdal by multiplying \f$ s \f$ by \f$ 1 + \frac{k_p^2}{k^2} \f$,
        // where the characteristic photon energy scale \f$ k_p \f$ is defined
        // in terms of the plasma frequency of the medium \f$ \omega_p \f$: \f$
        // k_p = \hbar \omega_p \frac{E}{m_e c^2} \f$
        const real_type k_p = electron_density_ * migdal_constant()
                              * ipow<2>(epsilon * gamma_energy_);
        s *= (1 + k_p / ipow<2>(gamma_energy_));

        // Recalculate \f$ \xi \$ from the modified suppression variable
        result.xi = 2;
        if (s > 1)
        {
            result.xi = 1;
        }
        else if (s > s1)
        {
            result.xi = 1 + std::log(s) / std::log(s1);
        }
    }

    // Calculate \f$ G(s) \f$ and \f$ \phi(s) \f$
    if (s < shared_.s_limit())
    {
        real_type val = s * shared_.inv_delta();
        CELER_ASSERT(val >= 0);

        const size_type ilow = static_cast<size_type>(val);
        CELER_ASSERT(ilow + 1 < shared_.lpm_table.size());

        val -= ilow;

        auto linterp = [val](real_type lo, real_type hi) {
            return lo + (hi - lo) * val;
        };

        auto get_table = [ilow, this](size_type offset) {
            return this->shared_.lpm_table[ItemId<MigdalData>{ilow + offset}];
        };

        result.g   = linterp(get_table(0).g, get_table(1).g);
        result.phi = linterp(get_table(0).phi, get_table(1).phi);
    }
    else
    {
        const real_type s4 = ipow<4>(s);
        result.g           = 1 - real_type(0.0230655) / s4;
        result.phi         = 1 - real_type(0.01190476) / s4;
    }

    // Make sure suppression is less than 1 (due to Migdal's approximation on
    // \f$ xi \f$)
    if (result.xi * result.phi > 1 || s > real_type(0.57))
    {
        result.xi = 1 / result.phi;
    }

    return result;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
