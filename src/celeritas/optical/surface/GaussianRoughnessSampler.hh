//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/GaussianRoughnessSampler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/math/Algorithms.hh"
#include "corecel/random/distribution/NormalDistribution.hh"
#include "corecel/random/distribution/RejectionSampler.hh"
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
 * . The "facet slope", an angle \f$ \alpha \f$ along a linear slice of a
 * crystal surface, is approximated as a normal distribution standard deviation
 * \f$ \sigma_\alpha \f$ .  (The paper justifies this distribution based on
 * surface roughness measurements with a bismuth germanate [BGO] crystal.)
 * Assuming an azimuthally isotropic surface, the polar distribution must be
 * expressed in terms of the tilt \f$ \theta \f$. The Jacobian factor for
 * spherical coordinates contributes a \f$ \sin \theta \f$ term, leading to the
 * spherical PDF \f[
   p(\sigma_\alpha; \theta,\phi) = \frac{1}{2\pi}\,\frac{1}{\mathcal N}\,
   \exp\!\left(-\frac{\theta^{2}}{2\sigma_\alpha^{2}}\right)\sin\theta \,.
 * \f]
 *
 * The polar angle \f$ \theta \f$ is sampled using rejection:
 * - Draw \f$ \alpha \f$ from the positive half of a normal distribution
 * - Reject angles greater than 90 degrees (for physicality) or 4 sigma (for
 *   sampling efficiency)
 * - Use an acceptance function \f$ \sin \theta \f$ bounded by the maximum
 *   theta
 *
 * The extra limitation of the angle being less than 4 sigma reduces the
 * rejection fraction by a factor of ~25 for smooth crystals (sigma = 0.01).
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
    Real3 normal_;
    NormalDistribution<real_type> sample_alpha_;
    real_type f_max_;
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
    : normal_(normal)
    , sample_alpha_(0, sigma_alpha)
    , f_max_(fmin(real_type{1}, 4 * sigma_alpha))
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
    using std::fabs;

    real_type cos_alpha{};
    real_type sin_alpha{};
    do
    {
        real_type alpha{};
        do
        {
            // Sample positive angle according to gaussian (chances of having a
            // nonpositive slope are generally vanishingly small)
            alpha = std::fabs(sample_alpha_(rng));
        } while (alpha >= real_type(constants::pi / 2));
        sincos(alpha, &sin_alpha, &cos_alpha);

        // Transform to polar angle using rejection
    } while (sin_alpha < f_max_ && RejectionSampler{sin_alpha, f_max_}(rng));

    // Rotate normal by alpha and then sample azimuthal rotation uniformly
    return ExitingDirectionSampler{cos_alpha, normal_}(rng);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
