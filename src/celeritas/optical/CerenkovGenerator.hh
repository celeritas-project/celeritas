//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CerenkovGenerator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/grid/GenericCalculator.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/GenerateCanonical.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

#include "CerenkovDndxCalculator.hh"
#include "GeneratorDistributionData.hh"
#include "MaterialData.hh"
#include "Primary.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Sample Cerenkov photons from the given distribution.
 *
 * Cerenkov radiation is emitted when a charged particle passes through a
 * dielectric medium faster than the speed of light in that medium. Photons are
 * emitted on the surface of a cone, with the cone angle, \f$ \theta \f$,
 * measured with respect to the incident particle direction. As the particle
 * slows down, the cone angle and the number of emitted photons decreases and
 * the frequency of the emitted photons increases.
 *
 * An incident charged particle with speed \f$ \beta \f$ will emit photons at
 * an angle \f$ \theta \f$ given by \f$ \cos\theta = 1 / (\beta n) \f$ where
 * \f$ n \f$ is the index of refraction of the matarial. The photon energy \f$
 * \epsilon \f$ is sampled from the PDF \f[
   f(\epsilon) = \left[1 - \frac{1}{n^2(\epsilon)\beta^2}\right]
 * \f]
 */
class CerenkovGenerator
{
  public:
    // Construct from optical materials and distribution parameters
    inline CELER_FUNCTION
    CerenkovGenerator(NativeCRef<MaterialParamsData> const& materials,
                      NativeCRef<CerenkovData> const& shared,
                      GeneratorDistributionData const& dist,
                      Span<Primary> photons);

    // Sample Cerenkov photons from the distribution
    template<class Generator>
    inline CELER_FUNCTION Span<Primary> operator()(Generator& rng);

  private:
    //// TYPES ////

    using UniformRealDist = UniformRealDistribution<real_type>;

    //// DATA ////

    GeneratorDistributionData const& dist_;
    Span<Primary> photons_;
    GenericCalculator calc_refractive_index_;
    UniformRealDist sample_phi_;
    UniformRealDist sample_energy_;
    UniformRealDist sample_num_photons_;
    Real3 dir_;
    Real3 delta_pos_;
    units::LightSpeed delta_speed_;
    real_type delta_num_photons_;
    real_type dndx_pre_;
    real_type sin_max_sq_;
    real_type inv_beta_;

    //// HELPER FUNCTIONS ////

    GenericCalculator
    make_calculator(NativeCRef<MaterialParamsData> const& material,
                    OpticalMaterialId mat_id);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from optical materials and distribution parameters.
 */
CELER_FUNCTION
CerenkovGenerator::CerenkovGenerator(
    NativeCRef<MaterialParamsData> const& material,
    NativeCRef<CerenkovData> const& shared,
    GeneratorDistributionData const& dist,
    Span<Primary> photons)
    : dist_(dist)
    , photons_(photons)
    , calc_refractive_index_(this->make_calculator(material, dist_.material))
    , sample_phi_(0, 2 * constants::pi)

{
    CELER_EXPECT(material);
    CELER_EXPECT(shared);
    CELER_EXPECT(dist_.material < material.refractive_index.size());
    CELER_EXPECT(dist_);
    CELER_EXPECT(photons_.size() == dist_.num_photons);

    auto const& energy_grid = calc_refractive_index_.grid();
    sample_energy_ = UniformRealDist(energy_grid.front(), energy_grid.back());

    // Calculate the mean number of photons produced per unit length at the
    // pre- and post-step energies
    auto const& pre_step = dist_.points[StepPoint::pre];
    auto const& post_step = dist_.points[StepPoint::post];
    CerenkovDndxCalculator calc_dndx(
        material, shared, dist_.material, dist_.charge);
    dndx_pre_ = calc_dndx(pre_step.speed);
    real_type dndx_post = calc_dndx(post_step.speed);

    // Helper used to sample the displacement
    sample_num_photons_ = UniformRealDist(0, max(dndx_pre_, dndx_post));

    // Calculate 1 / beta and the max sin^2 theta
    inv_beta_ = 2 / (pre_step.speed.value() + post_step.speed.value());
    real_type cos_max = inv_beta_ / calc_refractive_index_(energy_grid.back());
    sin_max_sq_ = diffsq(real_type(1), cos_max);

    // Calculate changes over the step
    delta_pos_ = post_step.pos - pre_step.pos;
    delta_num_photons_ = dndx_post - dndx_pre_;
    delta_speed_ = post_step.speed - pre_step.speed;

    // Incident particle direction
    dir_ = make_unit_vector(delta_pos_);
}

//---------------------------------------------------------------------------//
/*!
 * Sample Cerenkov photons from the distribution.
 */
template<class Generator>
CELER_FUNCTION Span<Primary> CerenkovGenerator::operator()(Generator& rng)
{
    for (auto i : range(dist_.num_photons))
    {
        // Sample energy and direction
        real_type energy;
        real_type cos_theta;
        real_type sin_theta_sq;
        do
        {
            // Sample an energy uniformly within the grid bounds
            energy = sample_energy_(rng);
            // Note that cos(theta) can be slightly larger than 1
            cos_theta = inv_beta_ / calc_refractive_index_(energy);
            sin_theta_sq = diffsq(real_type(1), cos_theta);
        } while (generate_canonical(rng) * sin_max_sq_ > sin_theta_sq);
        real_type phi = sample_phi_(rng);
        photons_[i].direction = rotate(from_spherical(cos_theta, phi), dir_);
        photons_[i].energy = units::MevEnergy(energy);

        // Photon polarization is perpendicular to the cone's surface
        photons_[i].polarization
            = rotate(from_spherical(-std::sqrt(sin_theta_sq), phi), dir_);

        // Sample position and time
        real_type u;
        do
        {
            u = generate_canonical(rng);
        } while (sample_num_photons_(rng) > dndx_pre_ + u * delta_num_photons_);
        real_type delta_time
            = u * dist_.step_length
              / (native_value_from(dist_.points[StepPoint::pre].speed)
                 + u * real_type(0.5) * native_value_from(delta_speed_));
        photons_[i].time = dist_.time + delta_time;
        photons_[i].position = dist_.points[StepPoint::pre].pos;
        axpy(u, delta_pos_, &photons_[i].position);
    }
    return photons_;
}

//---------------------------------------------------------------------------//
/*!
 * Return a calculator to compute index of refraction.
 */
CELER_FUNCTION GenericCalculator CerenkovGenerator::make_calculator(
    NativeCRef<MaterialParamsData> const& material, OpticalMaterialId mat_id)
{
    CELER_EXPECT(material);
    CELER_EXPECT(mat_id < material.refractive_index.size());

    return GenericCalculator(material.refractive_index[mat_id], material.reals);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
