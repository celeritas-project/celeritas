//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/UnifiedReflectionSampler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <numeric>

#include "corecel/Macros.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/random/distribution/Selector.hh"
#include "celeritas/optical/surface/SurfacePhysicsUtils.hh"

#include "LambertianDistribution.hh"
#include "SurfaceInteraction.hh"
#include "UnifiedReflectionData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Calculator for UNIFIED reflection model.
 *
 * The model specifies 4 different reflection modes:
 *  1. Specular spike: geometric reflection about global normal
 *  2. Specular lobe: geometric reflection about facet normal
 *  3. Back-scattering: reversed photon direction and polarization
 *  4. Diffuse Lambertian: reflection following Lambert's cosine law
 *
 * Only one reflection mode is selected based on the provided list of
 * probabilities.
 */
class UnifiedReflectionSampler
{
  public:
    // Construct from mode probabilities, photon, and surface data
    explicit inline CELER_FUNCTION
    UnifiedReflectionSampler(UnifiedModeProbs const& probs,
                             PhotonPhasor const& inc_photon,
                             Real3 const& global_normal,
                             Real3 const& facet_normal);

    // Sample reflection mode and calculate reflected phasor
    template<class Engine>
    inline CELER_FUNCTION PhotonPhasor operator()(Engine& rng) const;

    // Calculate specular spike reflection
    inline CELER_FUNCTION PhotonPhasor calc_specular_spike() const;

    // Calculate specular lobe reflection
    inline CELER_FUNCTION PhotonPhasor calc_specular_lobe() const;

    // Calculate back-scattering reflection
    inline CELER_FUNCTION PhotonPhasor calc_back_scattering() const;

    // Sample diffuse Lambertian reflection
    template<class Engine>
    inline CELER_FUNCTION PhotonPhasor
    sample_lambertian_reflection(Engine& rng) const;

  private:
    UnifiedModeProbs const& mode_probs_;
    PhotonPhasor const& inc_photon_;
    Real3 const& global_normal_;
    Real3 const& facet_normal_;

    // Calculate specular reflection about the given normal
    inline CELER_FUNCTION PhotonPhasor
    calc_specular_reflection(Real3 const& normal) const;
};

//---------------------------------------------------------------------------//
/*!
 * Construct calculator from probabilities, photon, and surface data.
 */
CELER_FUNCTION UnifiedReflectionSampler::UnifiedReflectionSampler(
    UnifiedModeProbs const& probs,
    PhotonPhasor const& inc_photon,
    Real3 const& global_normal,
    Real3 const& facet_normal)
    : mode_probs_(probs)
    , inc_photon_(inc_photon)
    , global_normal_(global_normal)
    , facet_normal_(facet_normal)
{
    CELER_EXPECT(inc_photon_.is_valid());
    CELER_EXPECT(is_soft_unit_vector(global_normal_));
    CELER_EXPECT(is_soft_unit_vector(facet_normal_));
    CELER_EXPECT(is_entering_surface(inc_photon_.direction, global_normal_));
    CELER_EXPECT(is_entering_surface(inc_photon_.direction, facet_normal_));
    CELER_EXPECT(soft_equal(
        std::accumulate(mode_probs_.begin(), mode_probs_.end(), real_type{0}),
        real_type{1}));
}

//---------------------------------------------------------------------------//
/*!
 * Sample reflection mode from probabilities and calculate reflection.
 */
template<class Engine>
CELER_FUNCTION PhotonPhasor UnifiedReflectionSampler::operator()(Engine& rng) const
{
    auto result = celeritas::make_selector(
        [this](UnifiedReflectionMode m) { return mode_probs_[m]; },
        UnifiedReflectionMode::size_)(rng);

    CELER_ASSERT(result != UnifiedReflectionMode::size_);

    switch (result)
    {
        case UnifiedReflectionMode::specular_spike:
            return this->calc_specular_spike();
        case UnifiedReflectionMode::specular_lobe:
            return this->calc_specular_lobe();
        case UnifiedReflectionMode::back_scattering:
            return this->calc_back_scattering();
        case UnifiedReflectionMode::diffuse_lambertian:
            return this->sample_lambertian_reflection(rng);
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Calculate specular spike reflection.
 *
 * This is geometric reflection about the global normal.
 */
CELER_FUNCTION PhotonPhasor UnifiedReflectionSampler::calc_specular_spike() const
{
    return this->calc_specular_reflection(global_normal_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate specular lobe reflection.
 *
 * This is geometric reflection about the facet normal.
 */
CELER_FUNCTION PhotonPhasor UnifiedReflectionSampler::calc_specular_lobe() const
{
    return this->calc_specular_reflection(facet_normal_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate back-scattering reflection.
 *
 * The photon direction and polarization are reversed.
 */
CELER_FUNCTION PhotonPhasor UnifiedReflectionSampler::calc_back_scattering() const
{
    return {-inc_photon_.direction, -inc_photon_.polarization};
}

//---------------------------------------------------------------------------//
/*!
 * Sample diffuse Lambertian reflection.
 *
 * Ideal diffuse reflection following Lambert's cosine law.
 */
template<class Engine>
CELER_FUNCTION PhotonPhasor
UnifiedReflectionSampler::sample_lambertian_reflection(Engine& rng) const
{
    return LambertianDistribution{global_normal_}(rng);
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to calculate geometric reflection about a given normal.
 */
CELER_FUNCTION PhotonPhasor
UnifiedReflectionSampler::calc_specular_reflection(Real3 const& normal) const
{
    return {geometric_reflection(inc_photon_.direction, normal),
            -geometric_reflection(inc_photon_.polarization, normal)};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
