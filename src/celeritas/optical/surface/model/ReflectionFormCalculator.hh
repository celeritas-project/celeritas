//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/ReflectionFormCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/ParticleTrackView.hh"
#include "celeritas/optical/surface/SurfacePhysicsTrackView.hh"
#include "celeritas/optical/surface/SurfacePhysicsUtils.hh"

#include "LambertianDistribution.hh"
#include "SurfaceInteraction.hh"

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
 */
class ReflectionFormCalculator
{
  public:
    // Construct from photon and surface data
    explicit inline CELER_FUNCTION
    ReflectionFormCalculator(Real3 const& direction,
                             Real3 const& polarization,
                             Real3 const& global_normal,
                             Real3 const& facet_normal);

    // Construct from track views
    explicit inline CELER_FUNCTION
    ReflectionFormCalculator(Real3 const& inc_direction,
                             ParticleTrackView const& photon,
                             SurfacePhysicsTrackView const& surface_physics);

    // Calculate specular spike reflection
    inline CELER_FUNCTION SurfaceInteraction calc_specular_spike() const;

    // Calculate specular lobe reflection
    inline CELER_FUNCTION SurfaceInteraction calc_specular_lobe() const;

    // Calculate back-scattering reflection
    inline CELER_FUNCTION SurfaceInteraction calc_backscatter() const;

    // Sample diffuse Lambertian reflection
    template<class Engine>
    inline CELER_FUNCTION SurfaceInteraction
    sample_lambertian_reflection(Engine& rng) const;

  private:
    Real3 const& direction_;
    Real3 const& polarization_;
    Real3 global_normal_;
    Real3 const& facet_normal_;

    // Calculate specular reflection about the given normal
    inline CELER_FUNCTION SurfaceInteraction
    calc_specular_reflection(Real3 const& normal) const;
};

//---------------------------------------------------------------------------//
/*!
 * Construct calculator from photon and surface data.
 */
CELER_FUNCTION
ReflectionFormCalculator::ReflectionFormCalculator(Real3 const& direction,
                                                   Real3 const& polarization,
                                                   Real3 const& global_normal,
                                                   Real3 const& facet_normal)
    : direction_(direction)
    , polarization_(polarization)
    , global_normal_(global_normal)
    , facet_normal_(facet_normal)
{
    CELER_EXPECT(is_soft_unit_vector(global_normal_));
    CELER_EXPECT(is_soft_unit_vector(facet_normal_));
    CELER_EXPECT(is_entering_surface(direction_, global_normal_));
    CELER_EXPECT(is_entering_surface(direction_, facet_normal_));
}

//---------------------------------------------------------------------------//
/*!
 * Construct calculator from a given track's views.
 */
CELER_FUNCTION ReflectionFormCalculator::ReflectionFormCalculator(
    Real3 const& inc_direction,
    ParticleTrackView const& photon,
    SurfacePhysicsTrackView const& surface_physics)
    : ReflectionFormCalculator(inc_direction,
                               photon.polarization(),
                               surface_physics.global_normal(),
                               surface_physics.facet_normal())
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate specular spike reflection.
 *
 * This is geometric reflection about the global normal.
 */
CELER_FUNCTION SurfaceInteraction
ReflectionFormCalculator::calc_specular_spike() const
{
    return this->calc_specular_reflection(global_normal_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate specular lobe reflection.
 *
 * This is geometric reflection about the facet normal.
 */
CELER_FUNCTION SurfaceInteraction
ReflectionFormCalculator::calc_specular_lobe() const
{
    return this->calc_specular_reflection(facet_normal_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate back-scattering reflection.
 *
 * The photon direction and polarization are reversed.
 */
CELER_FUNCTION SurfaceInteraction ReflectionFormCalculator::calc_backscatter() const
{
    SurfaceInteraction result;
    result.action = SurfaceInteraction::Action::reflected;
    result.direction = -direction_;
    result.polarization = -polarization_;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Sample diffuse Lambertian reflection.
 *
 * Ideal diffuse reflection following Lambert's cosine law.
 */
template<class Engine>
CELER_FUNCTION SurfaceInteraction
ReflectionFormCalculator::sample_lambertian_reflection(Engine& rng) const
{
    SurfaceInteraction result;
    result.action = SurfaceInteraction::Action::reflected;
    result.direction = LambertianDistribution{global_normal_}(rng);
    auto n = make_unit_vector(result.direction - direction_);
    result.polarization = -geometric_reflected_from(polarization_, n);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to calculate geometric reflection about a given normal.
 */
CELER_FUNCTION SurfaceInteraction
ReflectionFormCalculator::calc_specular_reflection(Real3 const& normal) const
{
    SurfaceInteraction result;
    result.action = SurfaceInteraction::Action::reflected;
    result.direction = geometric_reflected_from(direction_, normal);
    result.polarization = -geometric_reflected_from(polarization_, normal);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
