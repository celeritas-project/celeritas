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
    //! LPM suppression functions
    struct LPMFunctions
    {
        real_type xi;
        real_type g;
        real_type phi;
    };

  public:
    // Construct with material data and photon energy
    inline CELER_FUNCTION LPMCalculator(const MaterialView& material,
                                        const ElementView&  element,
                                        bool dielectric_suppression,
                                        units::MevEnergy gamma_energy);

    // Compute the LPM supression functions
    inline CELER_FUNCTION LPMFunctions operator()(real_type epsilon);

  private:
    //// DATA ////

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

    //// HELPER FUNCTIONS ////

    inline CELER_FUNCTION LPMFunctions compute_g_phi(real_type s) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with LPM data, material data, and photon energy.
 */
CELER_FUNCTION
LPMCalculator::LPMCalculator(const MaterialView& material,
                             const ElementView&  element,
                             bool                dielectric_suppression,
                             units::MevEnergy    gamma_energy)
    : element_(element)
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
    real_type xi = 2;
    if (s_prime > 1)
    {
        xi = 1;
    }
    else if (s_prime > constants::sqrt_two * s1)
    {
        const real_type log_s1 = std::log(constants::sqrt_two * s1);
        const real_type h      = std::log(s_prime) / log_s1;
        xi = 1 + h - real_type(0.08) * (1 - h) * h * (2 - h) / log_s1;
    }
    real_type s = s_prime / std::sqrt(xi);

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
        xi = 2;
        if (s > 1)
        {
            xi = 1;
        }
        else if (s > s1)
        {
            xi = 1 + std::log(s) / std::log(s1);
        }
    }

    // Calculate \f$ G(s) \f$ and \f$ \phi(s) \f$
    LPMFunctions result = this->compute_g_phi(s);

    // Make sure suppression is less than 1 (due to Migdal's approximation on
    // \f$ xi \f$)
    if (xi * result.phi > 1 || s > real_type(0.57))
    {
        xi = 1 / result.phi;
    }
    result.xi = xi;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Compute the LPM suppression functions \f$ G(s) \f$ and \f$ \phi(s) \f$.
 *
 * The functions are calculated using a piecewise approximation with simple
 * analytic functions.
 *
 * See section 10.2.2 of the Geant4 Physics Reference Manual and
 * ComputeLPMGsPhis in G4eBremsstrahlungRelModel and G4PairProductionRelModel.
 * Note that in Geant4 these are precomputed and tabulated at initialization.
 */
auto LPMCalculator::compute_g_phi(real_type s) const -> LPMFunctions
{
    using R = real_type;

    LPMFunctions result;

    if (s < R(0.01))
    {
        result.phi = 6 * s * (1 - constants::pi * s);
        result.g   = 12 * s - 2 * result.phi;
    }
    else
    {
        real_type s2 = ipow<2>(s);
        real_type s3 = s * s2;
        real_type s4 = ipow<2>(s2);

        // use Stanev approximation: for \psi(s) and compute G(s)
        if (s < R(0.415827))
        {
            result.phi
                = 1
                  - std::exp(-6 * s * (1 + s * (3 - constants::pi))
                             + s3 / (R(0.623) + R(0.796) * s + R(0.658) * s2));
            real_type psi = 1
                            - std::exp(-4 * s
                                       - 8 * s2
                                             / (1 + R(3.936) * s + R(4.97) * s2
                                                - R(0.05) * s3 + R(7.5) * s4));
            result.g = 3 * psi - 2 * result.phi;
        }
        else if (s < R(1.55))
        {
            result.phi
                = 1
                  - std::exp(-6 * s * (1 + s * (3 - constants::pi))
                             + s3 / (R(0.623) + R(0.796) * s + R(0.658) * s2));
            result.g = std::tanh(R(-0.160723) + R(3.755030) * s
                                 - R(1.798138) * s2 + R(0.672827) * s3
                                 - R(0.120772) * s4);
        }
        else
        {
            result.phi = 1 - R(0.01190476) / s4;
            if (s < R(1.9156))
            {
                result.g = std::tanh(R(-0.160723) + R(3.755030) * s
                                     - R(1.798138) * s2 + R(0.672827) * s3
                                     - R(0.120772) * s4);
            }
            else
            {
                result.g = 1 - R(0.0230655) / s4;
            }
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
