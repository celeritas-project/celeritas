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

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
class SmearRoughnessCalculator
{
  public:
    inline CELER_FUNCTION SmearRoughnessCalculator(real_type roughness,
                                                   Real3 const& normal,
                                                   Real3 const& dir);

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
    CELER_EXPECT(dot_product(normal_, dir_) < 0);
}

template<class Engine>
CELER_FUNCTION Real3 SmearRoughnessCalculator::operator()(Engine& rng) const
{
    UniformRealDistribution<real_type> sample_smear(-1, 1);
    Real3 local_normal;
    Real3 smear;
    do
    {
        do
        {
            smear = Real3{
                sample_smear(rng), sample_smear(rng), sample_smear(rng)};
        } while (dot_product(smear, smear) > 1);
        local_normal = normal_ + roughness_ * smear;
    } while (dot_product(local_normal, dir_) >= 0);

    return local_normal;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
