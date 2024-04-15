//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/WentzelTransportXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "celeritas/mat/MaterialView.hh"

#include "WentzelHelper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the transport cross section for the Wentzel OK and VI model.
 *
 * \note This performs the same calculation as the Geant4 method
 * G4WentzelOKandVIxSection::ComputeTransportCrossSectionPerAtom.
 */
class WentzelTransportXsCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using XsUnits = units::Native;  // [len^2]
    using Mass = units::MevMass;
    using MomentumSq = units::MevMomentumSq;
    //!@}

  public:
    // Construct with particle and precalculatad Wentzel data
    inline CELER_FUNCTION
    WentzelTransportXsCalculator(ParticleTrackView const& particle,
                                 WentzelHelper const& helper);

    // Calculate the transport cross section for the given angle [len^2]
    inline CELER_FUNCTION real_type operator()(real_type costheta_max) const;

  private:
    //// DATA ////

    AtomicNumber z_;
    real_type screening_coeff_;
    real_type costheta_max_elec_;
    real_type beta_sq_;
    real_type kin_factor_;

    //// HELPER FUNCTIONS ////

    // Calculate xs contribution from scattering off electrons or nucleus
    real_type calc_xs_contribution(real_type costheta_max) const;

    //! Limit on (1 - \c costheta_max) / \c screening_coeff
    static CELER_CONSTEXPR_FUNCTION real_type limit() { return 0.1; }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with particle and precalculatad Wentzel data.
 *
 * \c beta_sq should be calculated from the incident particle energy and mass.
 * \c screening_coeff and \c costheta_max_elec are calculated using the Wentzel
 * OK and VI model in \c WentzelHelper and depend on properties of the incident
 * particle, the energy cutoff in the current material, and the target element.
 */
CELER_FUNCTION
WentzelTransportXsCalculator::WentzelTransportXsCalculator(
    ParticleTrackView const& particle, WentzelHelper const& helper)
    : z_(helper.atomic_number())
    , screening_coeff_(2 * helper.screening_coefficient())
    , costheta_max_elec_(helper.costheta_max_electron())
    , beta_sq_(particle.beta_sq())
    , kin_factor_(helper.kin_factor())
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the transport cross section for the given angle [len^2].
 */
CELER_FUNCTION real_type
WentzelTransportXsCalculator::operator()(real_type costheta_max) const
{
    CELER_EXPECT(costheta_max <= 1);

    // Sum xs contributions from scattering off electrons and nucleus
    real_type xs_nuc = this->calc_xs_contribution(costheta_max);
    real_type xs_elec = costheta_max_elec_ > costheta_max
                            ? this->calc_xs_contribution(costheta_max_elec_)
                            : xs_nuc;
    real_type result = kin_factor_ * (xs_elec + z_.get() * xs_nuc);

    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate contribution to xs from scattering off electrons or nucleus.
 */
CELER_FUNCTION real_type WentzelTransportXsCalculator::calc_xs_contribution(
    real_type costheta_max) const
{
    real_type result;
    real_type const spin = real_type(0.5);
    real_type x = (1 - costheta_max) / screening_coeff_;
    if (x < WentzelTransportXsCalculator::limit())
    {
        real_type x_sq = ipow<2>(x);
        result = real_type(0.5) * x_sq
                 * ((1 - real_type(4) / 3 * x + real_type(1.5) * x_sq)
                    - screening_coeff_ * spin * beta_sq_ * x
                          * (real_type(2) / 3 - x));
    }
    else
    {
        real_type x_1 = x / (1 + x);
        real_type log_x = std::log(1 + x);
        result = log_x - x_1
                 - screening_coeff_ * spin * beta_sq_ * (x + x_1 - 2 * log_x);
    }
    return clamp_to_nonneg(result);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
