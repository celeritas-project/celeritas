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
#include "celeritas/grid/GenericGridData.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/GenerateCanonical.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

#include "CerenkovDistribution.hh"
#include "CerenkovDndxCalculator.hh"
#include "OpticalPrimary.hh"
#include "OpticalPropertyData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample Cerenkov photons from the given distribution.
 *
 * Cerenkov radiation is emitted when a charged particle passes through a
 * dielectric medium faster than the speed of light in that medium. A cone of
 * Cerenkov photons is emitted, with the cone angle \f$ \theta \f$, measured
 * with respect to the incident particle direction, decreasing as the particle
 * slows down.
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
    // Construct from optical properties and distribution parameters
    inline CELER_FUNCTION
    CerenkovGenerator(OpticalPropertyRef const& properties,
                      CerenkovRef const& shared,
                      CerenkovDistribution const& dist,
                      MaterialId material,
                      Span<OpticalPrimary> photons);

    // Sample Cerenkov photons from the distribution
    template<class Generator>
    inline CELER_FUNCTION void operator()(Generator& rng);

  private:
    //// TYPES ////

    using UniformRealDist = UniformRealDistribution<real_type>;

    //// DATA ////

    CerenkovDistribution const& dist_;
    Span<OpticalPrimary> photons_;
    GenericCalculator calc_refractive_index_;
    UniformRealDist sample_phi_;
    UniformRealDist sample_energy_;
    UniformRealDist sample_num_photons_;
    Real3 dir_;
    Real3 delta_pos_;
    real_type dndx_pre_;
    real_type delta_num_photons_;
    real_type delta_velocity_;
    real_type sin_max_sq_;
    real_type inv_beta_;

    //// HELPER FUNCTIONS ////

    GenericCalculator
    make_calculator(OpticalPropertyRef const& properties, MaterialId material);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from optical properties and distribution parameters.
 */
CELER_FUNCTION
CerenkovGenerator::CerenkovGenerator(OpticalPropertyRef const& properties,
                                     CerenkovRef const& shared,
                                     CerenkovDistribution const& dist,
                                     MaterialId material,
                                     Span<OpticalPrimary> photons)
    : dist_(dist)
    , photons_(photons)
    , calc_refractive_index_(this->make_calculator(properties, material))
    , sample_phi_(0, 2 * constants::pi)

{
    CELER_EXPECT(properties);
    CELER_EXPECT(shared);
    CELER_EXPECT(material < properties.materials.size());
    CELER_EXPECT(dist_);
    CELER_EXPECT(photons_.size() == dist_.num_photons);

    using constants::c_light;

    auto const& energy_grid = calc_refractive_index_.grid();
    sample_energy_ = UniformRealDist(energy_grid.front(), energy_grid.back());

    // Calculate the mean number of photons produced per unit length at the
    // pre- and post-step energies
    CerenkovDndxCalculator calc_dndx(
        properties, shared, material, dist_.charge);
    dndx_pre_ = calc_dndx(c_light / dist_.pre.velocity);
    real_type dndx_post = calc_dndx(c_light / dist_.post.velocity);
    sample_num_photons_ = UniformRealDist(0, max(dndx_pre_, dndx_post));

    inv_beta_ = 2 * c_light / (dist_.pre.velocity + dist_.post.velocity);
    real_type cos_max = inv_beta_ / calc_refractive_index_(energy_grid.back());
    sin_max_sq_ = diffsq(real_type(1), cos_max);

    // Calculate changes over the step
    delta_pos_ = dist_.post.pos - dist_.pre.pos;
    dir_ = make_unit_vector(delta_pos_);
    delta_num_photons_ = dndx_post - dndx_pre_;
    delta_velocity_ = dist_.post.velocity - dist_.pre.velocity;
}

//---------------------------------------------------------------------------//
/*!
 * Sample Cerenkov photons from the distribution.
 */
template<class Generator>
CELER_FUNCTION void CerenkovGenerator::operator()(Generator& rng)
{
    for (auto i : range(dist_.num_photons))
    {
        // Sample energy and direction
        real_type energy;
        real_type cos_theta;
        real_type sin_theta_sq;
        do
        {
            energy = sample_energy_(rng);
            cos_theta = inv_beta_ / calc_refractive_index_(energy);
            sin_theta_sq = diffsq(real_type(1), cos_theta);
        } while (!BernoulliDistribution(sin_max_sq_ / sin_theta_sq)(rng));
        real_type phi = sample_phi_(rng);
        photons_[i].direction = rotate(from_spherical(cos_theta, phi), dir_);
        photons_[i].energy = units::MevEnergy(energy);

        // Determine polarization
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
              / (dist_.pre.velocity + u * real_type(0.5) * delta_velocity_);
        photons_[i].time = dist_.time + delta_time;
        photons_[i].position = dist_.pre.pos;
        axpy(u, delta_pos_, &photons_[i].position);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Return a calculator to compute index of refraction.
 */
CELER_FUNCTION GenericCalculator CerenkovGenerator::make_calculator(
    OpticalPropertyRef const& properties, MaterialId material)
{
    CELER_EXPECT(properties);
    CELER_EXPECT(material < properties.materials.size());

    return GenericCalculator(properties.materials[material].refractive_index,
                             properties.reals);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
