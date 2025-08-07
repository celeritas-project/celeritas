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
#include "celeritas/optical/surface/SurfacePhysicsUtils.hh"
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
    inline CELER_FUNCTION GaussianRoughnessCalculator(real_type sigma_alpha,
                                                      Real3 const& normal,
                                                      Real3 const& dir);

    // Sample facet normal
    template<class Engine>
    inline CELER_FUNCTION Real3 operator()(Engine& rng) const;

  private:
    real_type sigma_alpha_;
    Real3 const& normal_;
    Real3 const& dir_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from sigma_alpha, global normal, and incident track direction.
 */
CELER_FUNCTION
GaussianRoughnessCalculator::GaussianRoughnessCalculator(real_type sigma_alpha,
                                                         Real3 const& normal,
                                                         Real3 const& dir)
    : sigma_alpha_(sigma_alpha), normal_(normal), dir_(dir)
{
    CELER_EXPECT(sigma_alpha_ > 0);
    CELER_EXPECT(is_soft_unit_vector(normal_));
    CELER_EXPECT(is_soft_unit_vector(dir_));
    CELER_EXPECT(is_entering_surface(normal_, dir_));
}

//---------------------------------------------------------------------------//
/*!
 * Sample a facet normal via the Gaussian roughness model.
 */
template<class Engine>
CELER_FUNCTION Real3 GaussianRoughnessCalculator::operator()(Engine& rng) const
{
    NormalDistribution<real_type> sample_alpha(0, sigma_alpha_);
    real_type f_max = min(real_type{1}, 4 * sigma_alpha_);

    Real3 local_normal;
    do
    {
        real_type cos_alpha, sin_alpha;
        do
        {
            cos_alpha = cos(sample_alpha(rng));
            sin_alpha = sqrt(1 - ipow<2>(sin_alpha));
        } while (cos_alpha <= 0 || RejectionSampler{sin_alpha, f_max}(rng));

        local_normal = ExitingDirectionSampler{cos_alpha, normal_}(rng);
    } while (!is_entering_surface(local_normal, dir_));

    return local_normal;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
