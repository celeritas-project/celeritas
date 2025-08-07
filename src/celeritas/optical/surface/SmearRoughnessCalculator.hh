//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SmearRoughnessCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/ArrayUtils.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"
#include "celeritas/Constants.hh"

#include "SurfacePhysicsUtils.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Sample a facet normal from a smear roughness model.
 *
 * The smear roughness model is parameterized by a single roughness value
 * in [0,1] where:
 *
 * - 0 roughness is polished (specular spike reflection)
 * - 1 roughness is rough (diffuse reflection)
 *
 * A smear direction is uniformly sampled within a sphere of radius 1, which is
 * then scaled by the roughness parameter and added to the global normal. The
 * resulting unit vector is the facet normal provided it points opposite the
 * incident track direction, otherwise the facet normal is resampled.
 */
class SmearRoughnessCalculator
{
  public:
    // Construct from roughness, global normal, and incident direction
    inline CELER_FUNCTION SmearRoughnessCalculator(real_type roughness,
                                                   Real3 const& normal,
                                                   Real3 const& dir);

    // Sample facet normal
    template<class Engine>
    inline CELER_FUNCTION Real3 operator()(Engine& rng) const;

  private:
    real_type roughness_;
    Real3 const& normal_;
    Real3 const& dir_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from roughness, global normal, and incident track direction.
 */
CELER_FUNCTION
SmearRoughnessCalculator::SmearRoughnessCalculator(real_type roughness,
                                                   Real3 const& normal,
                                                   Real3 const& dir)
    : roughness_(roughness), normal_(normal), dir_(dir)
{
    CELER_EXPECT(0 <= roughness_ && roughness_ <= 1);
    CELER_EXPECT(is_soft_unit_vector(normal_));
    CELER_EXPECT(is_soft_unit_vector(dir_));
    CELER_EXPECT(is_entering_surface(normal_, dir_));
}

//---------------------------------------------------------------------------//
/*!
 * Sample a facet normal via the smear roughness model.
 */
template<class Engine>
CELER_FUNCTION Real3 SmearRoughnessCalculator::operator()(Engine& rng) const
{
    UniformRealDistribution<real_type> sample_phi(
        0, real_type(2 * constants::pi));
    UniformRealDistribution<real_type> sample_cos_theta(-1, 1);
    UniformRealDistribution<real_type> sample_r(0, 1);

    Real3 local_normal;
    do
    {
        local_normal = normal_;
        axpy(cbrt(sample_r(rng)) * roughness_,
             from_spherical(sample_cos_theta(rng), sample_phi(rng)),
             &local_normal);
    } while (!is_entering_surface(local_normal, dir_));

    return make_unit_vector(local_normal);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
