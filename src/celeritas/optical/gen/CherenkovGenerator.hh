//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/CherenkovGenerator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/random/distribution/RejectionSampler.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"

#include "CherenkovDndxCalculator.hh"
#include "GeneratorData.hh"
#include "../MaterialView.hh"
#include "../TrackInitializer.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample Cherenkov photons from a generator distribution.
 *
 * Cherenkov radiation is emitted when a charged particle passes through a
 * dielectric medium faster than the speed of light in that medium. Photons are
 * emitted on the surface of a cone, with the cone angle, \f$ \theta \f$,
 * measured with respect to the incident particle direction. As the particle
 * slows down, the cone angle and the number of emitted photons decreases and
 * the frequency of the emitted photons increases.
 *
 * An incident charged particle with speed \f$ \beta \f$ will emit photons at
 * an angle \f$ \theta \f$ given by \f$ \cos\theta = 1 / (\beta n) \f$ where
 * \f$ n \f$ is the index of refraction of the matarial. The photon energy
 * \f$ \epsilon \f$ is sampled from the PDF \f[
   f(\epsilon) = \left[1 - \frac{1}{n^2(\epsilon)\beta^2}\right]
 * \f]
 */
class CherenkovGenerator
{
  public:
    // Construct from optical materials and distribution parameters
    inline CELER_FUNCTION
    CherenkovGenerator(optical::MaterialView const& material,
                       NativeCRef<CherenkovData> const& shared,
                       GeneratorDistributionData const& dist);

    // Sample a Cherenkov photon from the distribution
    template<class Generator>
    inline CELER_FUNCTION optical::TrackInitializer operator()(Generator& rng);

  private:
    //// TYPES ////

    using UniformRealDist = UniformRealDistribution<real_type>;

    //// DATA ////

    GeneratorDistributionData const& dist_;
    NonuniformGridCalculator calc_refractive_index_;
    UniformRealDist sample_phi_;
    UniformRealDist sample_num_photons_;
    UniformRealDist sample_energy_;
    Real3 dir_;
    Real3 delta_pos_;
    units::LightSpeed delta_speed_;
    real_type delta_num_photons_;
    real_type dndx_pre_;
    real_type sin_max_sq_;
    real_type inv_beta_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from optical materials and distribution parameters.
 */
CELER_FUNCTION
CherenkovGenerator::CherenkovGenerator(optical::MaterialView const& material,
                                       NativeCRef<CherenkovData> const& shared,
                                       GeneratorDistributionData const& dist)
    : dist_(dist)
    , calc_refractive_index_(material.make_refractive_index_calculator())
    , sample_phi_(0, real_type(2 * constants::pi))
{
    CELER_EXPECT(shared);
    CELER_EXPECT(dist_);
    CELER_EXPECT(material.material_id() == dist_.material);

    using LS = units::LightSpeed;

    // Calculate the mean number of photons produced per unit length at the
    // pre- and post-step energies
    auto const& pre_step = dist_.points[StepPoint::pre];
    auto const& post_step = dist_.points[StepPoint::post];
    CherenkovDndxCalculator calc_dndx(material, shared, dist_.charge);
    dndx_pre_ = calc_dndx(pre_step.speed);
    real_type dndx_post = calc_dndx(post_step.speed);

    // Helper used to sample the displacement
    sample_num_photons_ = UniformRealDist(0, max(dndx_pre_, dndx_post));

    // Helper to sample exiting photon energies
    auto const& energy_grid = calc_refractive_index_.grid();
    sample_energy_ = UniformRealDist(energy_grid.front(), energy_grid.back());

    // Calculate 1 / beta and the max sin^2 theta
    inv_beta_
        = 2 / (value_as<LS>(pre_step.speed) + value_as<LS>(post_step.speed));
    CELER_ASSERT(inv_beta_ > 1);
    real_type cos_max = inv_beta_ / calc_refractive_index_(energy_grid.back());
    sin_max_sq_ = 1 - ipow<2>(cos_max);

    // Calculate changes over the step
    delta_pos_ = post_step.pos - pre_step.pos;
    delta_num_photons_ = dndx_post - dndx_pre_;
    delta_speed_ = post_step.speed - pre_step.speed;

    // Incident particle direction
    dir_ = make_unit_vector(delta_pos_);
}

//---------------------------------------------------------------------------//
/*!
 * Sample Cherenkov photons from the distribution.
 */
template<class Generator>
CELER_FUNCTION optical::TrackInitializer
CherenkovGenerator::operator()(Generator& rng)
{
    // Sample energy and direction
    real_type energy;
    real_type cos_theta;
    real_type sin_theta_sq;
    do
    {
        // Sample an energy uniformly within the grid bounds, rejecting
        // if the refractive index at the sampled energy is such that the
        // incident particle's average speed is subluminal at that photon
        // wavelength.
        // We could improve sampling efficiency for this edge case by
        // increasing the minimum energy (as is done in
        // CherenkovDndxCalculator) to where the refractive index satisfies
        // this condition, but since fewer photons are emitted at lower
        // energies in general, relatively few rejections will take place
        // here.
        do
        {
            energy = sample_energy_(rng);
            cos_theta = inv_beta_ / calc_refractive_index_(energy);
        } while (cos_theta > 1);
        sin_theta_sq = 1 - ipow<2>(cos_theta);
    } while (RejectionSampler{sin_theta_sq, sin_max_sq_}(rng));

    // Sample azimuthal photon direction
    real_type phi = sample_phi_(rng);
    optical::TrackInitializer photon;
    photon.direction = rotate(from_spherical(cos_theta, phi), dir_);
    photon.energy = units::MevEnergy(energy);

    // Photon polarization is perpendicular to the cone's surface
    photon.polarization = make_orthogonal(
        rotate(from_spherical(-std::sqrt(sin_theta_sq), phi), dir_),
        photon.direction);
    CELER_ASSERT(soft_zero(dot_product(photon.polarization, photon.direction)));

    // Sample fraction along the step
    UniformRealDistribution<> sample_step_fraction;
    real_type u;
    do
    {
        u = sample_step_fraction(rng);
    } while (sample_num_photons_(rng) > dndx_pre_ + u * delta_num_photons_);

    real_type delta_time
        = u * dist_.step_length
          / (native_value_from(dist_.points[StepPoint::pre].speed)
             + u * real_type(0.5) * native_value_from(delta_speed_));
    photon.time = dist_.time + delta_time;
    photon.position = dist_.points[StepPoint::pre].pos;
    axpy(u, delta_pos_, &photon.position);
    return photon;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
