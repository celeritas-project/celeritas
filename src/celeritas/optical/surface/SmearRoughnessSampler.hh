//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SmearRoughnessSampler.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/ArrayUtils.hh"
#include "corecel/random/distribution/IsotropicDistribution.hh"
#include "corecel/random/distribution/PowerDistribution.hh"
#include "geocel/Types.hh"

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
 * then scaled by the roughness parameter and added to the global normal.
 */
class SmearRoughnessSampler
{
  public:
    // Construct from roughness and global normal
    inline CELER_FUNCTION
    SmearRoughnessSampler(Real3 const& normal, real_type roughness);

    // Sample facet normal
    template<class Engine>
    inline CELER_FUNCTION Real3 operator()(Engine& rng) const;

  private:
    Real3 normal_;
    real_type roughness_;
    PowerDistribution<> sample_r_{2};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from roughness and global normal.
 */
CELER_FUNCTION
SmearRoughnessSampler::SmearRoughnessSampler(Real3 const& normal,
                                             real_type roughness)
    : normal_(normal), roughness_(roughness)
{
    CELER_EXPECT(0 <= roughness_ && roughness_ <= 1);
    CELER_EXPECT(is_soft_unit_vector(normal_));
}

//---------------------------------------------------------------------------//
/*!
 * Sample a facet normal via the smear roughness model.
 */
template<class Engine>
CELER_FUNCTION Real3 SmearRoughnessSampler::operator()(Engine& rng) const
{
    real_type r = sample_r_(rng);
    Real3 local_normal = normal_;
    axpy(r * roughness_, IsotropicDistribution{}(rng), &local_normal);

    return make_unit_vector(local_normal);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
