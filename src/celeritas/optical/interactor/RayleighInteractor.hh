//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/interactor/RayleighInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArraySoftUnit.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/random/distribution/BernoulliDistribution.hh"
#include "corecel/random/distribution/IsotropicDistribution.hh"
#include "corecel/random/distribution/RejectionSampler.hh"
#include "celeritas/optical/Interaction.hh"
#include "celeritas/optical/ParticleTrackView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Sample optical Rayleigh scattering.
 *
 * Optical Rayleigh scattering is the elastic scattering of optical photons
 * in a material. The scattered polarization is guaranteed to be in the
 * same plane as the original polarization and new direction.
 */
class RayleighInteractor
{
  public:
    // Construct interactor from an optical track
    inline CELER_FUNCTION RayleighInteractor(ParticleTrackView const& particle,
                                             Real3 const& direction);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    Real3 const& inc_dir_;  //!< Direction of incident photon
    Real3 const& inc_pol_;  //!< Polarization of incident photon
    IsotropicDistribution<real_type> sample_direction_{};
    RejectionSampler<real_type> reject_angle_{1};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct the interactor for the given optical track.
 */
CELER_FUNCTION
RayleighInteractor::RayleighInteractor(ParticleTrackView const& particle,
                                       Real3 const& direction)
    : inc_dir_(direction), inc_pol_(particle.polarization())
{
    CELER_EXPECT(is_soft_unit_vector(inc_dir_));
    CELER_EXPECT(is_soft_unit_vector(inc_pol_));
    CELER_EXPECT(is_soft_orthogonal(inc_dir_, inc_pol_));
}

//---------------------------------------------------------------------------//
/*!
 * Sample a single optical Rayleigh interaction.
 */
template<class Engine>
CELER_FUNCTION Interaction RayleighInteractor::operator()(Engine& rng)
{
    Real3 new_dir, new_pol;
    do
    {
        do
        {
            new_dir = sample_direction_(rng);

            // Project polarization onto plane perpendicular to new direction
            new_pol = make_unit_vector(make_orthogonal(inc_pol_, new_dir));

            // Reject rare case of polarization and new direction being
            // coincident leading to loss of orthogonality
        } while (CELER_UNLIKELY(!is_soft_orthogonal(new_pol, new_dir)));

        if (!BernoulliDistribution{0.5}(rng))
        {
            // Flip direction with 50% probability: there are two polarizations
            // perpendicular to the new direction and the original polarization
            new_pol = -new_pol;
        }
        // Accept with the probability of the scattered polarization overlap
        // squared
    } while (reject_angle_(ipow<2>(dot_product(new_pol, inc_pol_)), rng));

    CELER_ENSURE(is_soft_unit_vector(new_dir));
    CELER_ENSURE(is_soft_unit_vector(new_pol));
    Interaction result;
    result.direction = new_dir;
    result.polarization = new_pol;
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
