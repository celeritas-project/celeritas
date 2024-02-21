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
    using Charge = units::ElementaryCharge;
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
    real_type xs_factor_;

    //// HELPER FUNCTIONS ////

    // Calculate the multiplicative factor for the transport cross section
    real_type calc_xs_factor(ParticleTrackView const&) const;

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
    , screening_coeff_(helper.screening_coefficient())
    , costheta_max_elec_(helper.costheta_max_electron())
    , beta_sq_(particle.beta_sq())
    , xs_factor_(this->calc_xs_factor(particle))
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
    real_type result = xs_factor_ * (xs_elec + z_.get() * xs_nuc);

    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the multiplicative factor for the transport cross section.
 *
 * This calculates the factor
 * \f[
   f = \frac{2 \pi m_e^2 r_e^2 Z q^2}{\beta^2 p^2},
 * \f]
 * where \f$ m_e, r_e, Z, q, \beta \f$, and \f$ p \f$ are the electron mass,
 * classical electron radius, atomic number of the target atom, charge,
 * relativistic speed, and momentum of the incident particle, respectively.
 */
CELER_FUNCTION real_type WentzelTransportXsCalculator::calc_xs_factor(
    ParticleTrackView const& particle) const
{
    real_type constexpr twopi_mrsq
        = 2 * constants::pi
          * ipow<2>(native_value_to<Mass>(constants::electron_mass).value()
                    * constants::r_electron);

    return twopi_mrsq * z_.get() * ipow<2>(value_as<Charge>(particle.charge()))
           / (particle.beta_sq()
              * value_as<MomentumSq>(particle.momentum_sq()));
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
