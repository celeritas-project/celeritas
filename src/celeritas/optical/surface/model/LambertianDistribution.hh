//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/LambertianDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Sample a reflected vector from a diffuse Lambertian distribution.
 *
 * Lambertian reflectance is an approximation of a diffuse material where the
 * apparent brightness is equal for observers at all angles. Reflected vectors
 * follow Lambert's cosine law, which states the intensity of reflected light
 * is proportional to the cosine of the reflection angle.
 */
class LambertianDistribution
{
  public:
    // Construct distribution about a given normal
    explicit inline CELER_FUNCTION LambertianDistribution(Real3 const& normal);

    // Sample by Lambertian cosine law
    template<class Engine>
    inline CELER_FUNCTION Real3 operator()(Engine&) const;

  private:
    Real3 const& normal_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct distribution about a normal.
 */
CELER_FUNCTION
LambertianDistribution::LambertianDistribution(Real3 const& normal)
    : normal_(normal)
{
    CELER_EXPECT(is_soft_unit_vector(normal_));
}

//---------------------------------------------------------------------------//
/*!
 * Sample by Lambertian cosine law.
 */
template<class Engine>
CELER_FUNCTION Real3 LambertianDistribution::operator()(Engine&) const
{
    return {0, 0, 0};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
