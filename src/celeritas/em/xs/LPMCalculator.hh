//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/LPMCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"
#include "celeritas/grid/PolyEvaluator.hh"
#include "celeritas/mat/MaterialView.hh"

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

    inline CELER_FUNCTION real_type calc_phi(real_type s) const;
    inline CELER_FUNCTION real_type calc_g(real_type s, real_type phi) const;
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
 * G4PairProductionRelModel. Also see T. Stanev, Ch. Vankov, Development of
 * ultrahigh-energy electromagnetic cascades in water and lead including the
 * Landau-Pomeranchuk-Migdal effect, Phys. Rev. D, 25 (1982), p. 1291.
 */
CELER_FUNCTION auto LPMCalculator::operator()(real_type epsilon) -> LPMFunctions
{
    // Suppression variable \f$ s' \f$. For bremsstrahlung \f$ s' =
    // \sqrt{\frac{E_\textrm{LPM} k}{8 E (E - k)}} \f$, and for pair production
    // \f$ s' = \sqrt{\frac{E_\textrm{LPM} k}{8 E (k - E)}} \f$, where \f$ k \$
    // is the photon energy and \f$ E \f$ is the electon (or positron) energy
    const real_type s_prime = std::sqrt(
        lpm_energy_ / (8 * epsilon * gamma_energy_ * std::fabs(epsilon - 1)));

    const real_type s1 = ipow<2>(element_.cbrt_z() / real_type(184.15));

    // Calculate \f$ \xi(s') \f$ and \f$ s = \frac{s'}{\sqrt{\xi(s')}} \f$ (Eq.
    // 21 in Stanev)
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
        const real_type k_p_sq = electron_density_ * migdal_constant()
                                 * ipow<2>(epsilon * gamma_energy_);
        s *= (1 + k_p_sq / ipow<2>(gamma_energy_));

        // Recalculate \f$ \xi \$ from the modified suppression variable (Eq.
        // 16 in Stanev)
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

    // Make sure suppression is less than 1 (due to Migdal's approximation on
    // \f$ xi \f$)
    real_type phi = this->calc_phi(s);
    if (xi * phi > 1 || s > real_type(0.57))
    {
        xi = 1 / phi;
    }
    LPMFunctions result;
    result.phi = phi;
    result.xi  = xi;
    result.g   = this->calc_g(s, phi);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Compute the LPM suppression function \f$ \phi(s) \f$.
 *
 * The functions are calculated using a piecewise approximation with simple
 * analytic functions. For s > 1.55 use the Stanev approximation.
 *
 * See section 10.2.2 of the Geant4 Physics Reference Manual and
 * ComputeLPMGsPhis in G4eBremsstrahlungRelModel and G4PairProductionRelModel.
 * Note that in Geant4 these are precomputed and tabulated at initialization.
 */
CELER_FUNCTION real_type LPMCalculator::calc_phi(real_type s) const
{
    using PolyLin  = PolyEvaluator<real_type, 1>;
    using PolyQuad = PolyEvaluator<real_type, 2>;

    if (s < real_type(0.01))
    {
        return s * PolyLin(6, -6 * constants::pi)(s);
    }
    else if (s < real_type(1.55))
    {
        real_type a = PolyQuad{0.623, 0.796, 0.658}(s);
        real_type b = PolyQuad{-6, -6 * (3 - constants::pi), 1 / a}(s);
        return 1 - std::exp(s * b);
    }
    else
    {
        return 1 - real_type(0.01190476) / ipow<4>(s);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Compute the LPM suppression function \f$ g(s) \f$.
 */
CELER_FUNCTION real_type LPMCalculator::calc_g(real_type s, real_type phi) const
{
    using PolyLin   = PolyEvaluator<real_type, 1>;
    using PolyQuart = PolyEvaluator<real_type, 4>;

    if (s < real_type(0.01))
    {
        return PolyLin(-2 * phi, 12)(s);
    }
    else if (s < real_type(0.415827))
    {
        real_type a   = PolyQuart{1, 3.936, 4.97, -0.05, 7.5}(s);
        real_type b   = PolyLin{-4, -8 / a}(s);
        real_type psi = 1 - std::exp(s * b);
        return 3 * psi - 2 * phi;
    }
    else if (s < real_type(1.9156))
    {
        return std::tanh(
            PolyQuart{-0.160723, 3.755030, -1.798138, 0.672827, -0.120772}(s));
    }
    else
    {
        return 1 - real_type(0.0230655) / ipow<4>(s);
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
