//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/calc/GaussianRoughnessCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/ArrayUtils.hh"
#include "corecel/random/distribution/NormalDistribution.hh"
#include "corecel/random/distribution/RejectionSampler.hh"
#include "celeritas/Constants.hh"
#include "celeritas/phys/InteractionUtils.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Sample a facet normal from a Gaussian roughness model.
 *
 * The Gaussian roughness model is parameterized by a positive real number \c
 * sigma_alpha. The angle \c alpha between the facet normal and the global
 * normal is sampled from the distribution
 * \f[
 * p(\alpha) = N(\alpha; 0, \sigma_\alpha) * \sin(\alpha)
 * \f]
 * where alpha is in the range [0, pi/2).
 */
class GaussianRoughnessCalculator
{
  public:
    // Construct from sigma_alpha, global normal, and incident direction
    inline CELER_FUNCTION
    GaussianRoughnessCalculator(real_type sigma_alpha, Real3 const& normal);

    // Sample facet normal
    template<class Engine>
    inline CELER_FUNCTION Real3 operator()(Engine& rng);

  private:
    NormalDistribution<real_type> sample_alpha_;
    real_type f_max_;
    Real3 const& normal_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from sigma_alpha and global normal.
 */
CELER_FUNCTION
GaussianRoughnessCalculator::GaussianRoughnessCalculator(real_type sigma_alpha,
                                                         Real3 const& normal)
    : sample_alpha_(0, sigma_alpha)
    , f_max_(fmin(real_type{1}, 4 * sigma_alpha))
    , normal_(normal)
{
    CELER_EXPECT(sigma_alpha > 0);
    CELER_EXPECT(is_soft_unit_vector(normal_));
}

//---------------------------------------------------------------------------//
/*!
 * Sample a facet normal via the Gaussian roughness model.
 */
template<class Engine>
CELER_FUNCTION Real3 GaussianRoughnessCalculator::operator()(Engine& rng)
{
    real_type cos_alpha = 0;
    real_type sin_alpha = 0;
    do
    {
        real_type alpha = sample_alpha_(rng);
        sincos(alpha, &sin_alpha, &cos_alpha);
    } while (cos_alpha <= 0 || RejectionSampler{fabs(sin_alpha), f_max_}(rng));

    // Rotate normal by alpha and then sample azimuth rotation uniformly
    return ExitingDirectionSampler{cos_alpha, normal_}(rng);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
