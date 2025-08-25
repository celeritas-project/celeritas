//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/GaussianRoughnessSampler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/math/Algorithms.hh"
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
 * The Gaussian roughness model was introduced in
 * \citet{levin-morephysical-1996, https://doi.org/10.1109/NSSMIC.1996.591410}
 * . The "facet slope", an angle \c alpha between the facet normal and the
 * global normal, is sampled from a normal distribution with standard deviation
 * \c sigma_alpha . The paper justifies this distribution based on surface
 * roughness measurements with a bismuth germanate (BGO) crystal.
 */
class GaussianRoughnessSampler
{
  public:
    // Construct from sigma_alpha, global normal, and incident direction
    inline CELER_FUNCTION
    GaussianRoughnessSampler(Real3 const& normal, real_type sigma_alpha);

    // Sample facet normal
    template<class Engine>
    inline CELER_FUNCTION Real3 operator()(Engine& rng);

  private:
    Real3 const& normal_;
    NormalDistribution<real_type> sample_alpha_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from sigma_alpha and global normal.
 */
CELER_FUNCTION
GaussianRoughnessSampler::GaussianRoughnessSampler(Real3 const& normal,
                                                   real_type sigma_alpha)
    : normal_(normal), sample_alpha_(0, sigma_alpha)
{
    CELER_EXPECT(sigma_alpha > 0);
    CELER_EXPECT(is_soft_unit_vector(normal_));
}

//---------------------------------------------------------------------------//
/*!
 * Sample a facet normal via the Gaussian roughness model.
 */
template<class Engine>
CELER_FUNCTION Real3 GaussianRoughnessSampler::operator()(Engine& rng)
{
    real_type cos_alpha{};
    do
    {
        // Sample angle according to gaussian (chances of having a nonpositive
        // slope are vanishingly small)
        cos_alpha = std::cos(sample_alpha_(rng));
    } while (cos_alpha <= 0);

    // Rotate normal by alpha and then sample azimuth rotation uniformly
    return ExitingDirectionSampler{cos_alpha, normal_}(rng);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
