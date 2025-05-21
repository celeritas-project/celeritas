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
#include "corecel/math/SoftEqual.hh"
#include "corecel/random/distribution/BernoulliDistribution.hh"
#include "corecel/random/distribution/RejectionSampler.hh"
#include "geocel/random/IsotropicDistribution.hh"
#include "celeritas/optical/Interaction.hh"
#include "celeritas/optical/ParticleTrackView.hh"
#include "celeritas/phys/InteractionUtils.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Sample optical Rayleigh scattering.
 *
 * Optical Rayleigh scattering is the elastic scattering of optical photons
 * in a material. The scattered polarization is guarenteed to be in the
 * same plane as the original polarization and new direction if the latter
 * vectors are not parallel. Otherwise it will just be perpendicular to
 * the new direction.
 */
class RayleighInteractor
{
  public:
    // Construct interactor from an optical track
    inline CELER_FUNCTION RayleighInteractor(ParticleTrackView const& particle,
                                             Real3 const& direction);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng) const;

  private:
    Real3 const& inc_dir_;  //!< Direction of incident photon
    Real3 const& inc_pol_;  //!< Polarization of incident photon
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
    CELER_EXPECT(soft_zero(dot_product(inc_dir_, inc_pol_)));
}

//---------------------------------------------------------------------------//
/*!
 * Sample the scattered direction and polarization for a single optical
 * Rayliegh interaction.
 */
template<class Engine>
CELER_FUNCTION Interaction RayleighInteractor::operator()(Engine& rng) const
{
    Interaction result;
    Real3& new_dir = result.direction;
    Real3& new_pol = result.polarization;

    IsotropicDistribution sample_direction{};
    do
    {
        new_dir = sample_direction(rng);

        auto projected_pol = dot_product(new_dir, inc_pol_);

        if (soft_zero(projected_pol))
        {
            // If new direction is parallel to incident polarization, then
            // randomly sample azimuthal direction
            new_pol = ExitingDirectionSampler{0, new_dir}(rng);
        }
        else
        {
            // Project polarization onto plane perpendicular to new direction
            new_pol = inc_pol_ - projected_pol * new_dir;
            // Enforce orthogonality
            axpy(-dot_product(new_pol, new_dir), new_dir, &new_pol);
            new_pol = make_unit_vector(
                BernoulliDistribution{0.5}(rng) ? new_pol : -new_pol);
        }

        // Accept with the probability of the scattered polarization overlap
        // squared
    } while (RejectionSampler{ipow<2>(
        clamp<real_type>(dot_product(new_pol, inc_pol_), -1, 1))}(rng));

    CELER_ENSURE(is_soft_unit_vector(new_dir));
    CELER_ENSURE(is_soft_unit_vector(new_pol));
    CELER_ENSURE(soft_zero(dot_product(new_pol, new_dir)));

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
