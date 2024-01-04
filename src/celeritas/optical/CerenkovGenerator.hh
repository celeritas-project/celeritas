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
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/grid/GenericXsCalculator.hh"
#include "celeritas/grid/XsGridData.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/GenerateCanonical.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

#include "CerenkovDistribution.hh"
#include "PhotonPrimary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample Cerenkov photons.
 */
class CerenkovGenerator
{
  public:
    // Placeholder struct for tabulated index of refraction values
    // TODO: remove once we store optical properties
    struct RefractiveIndex
    {
        using Values
            = Collection<real_type, Ownership::const_reference, MemSpace::native>;

        GenericGridData const& data;
        Values const& reals;
    };

  public:
    // Construct from optical properties and distribution parameters
    inline CELER_FUNCTION
    CerenkovGenerator(RefractiveIndex const& refractive_index,
                      CerenkovDistribution const& dist,
                      Span<PhotonPrimary> photons);

    // Sample Cerenkov photons from the distribution
    template<class Generator>
    inline CELER_FUNCTION void operator()(Generator& rng);

  private:
    RefractiveIndex const& refractive_index_;
    CerenkovDistribution const& dist_;
    Span<PhotonPrimary> photons_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from optical properties and distribution parameters.
 */
CELER_FUNCTION
CerenkovGenerator::CerenkovGenerator(RefractiveIndex const& refractive_index,
                                     CerenkovDistribution const& dist,
                                     Span<PhotonPrimary> photons)
    : refractive_index_(refractive_index), dist_(dist), photons_(photons)

{
    CELER_EXPECT(refractive_index_.data);
    CELER_EXPECT(dist_);
    CELER_EXPECT(photons_.size() == dist_.num_photons);
}

//---------------------------------------------------------------------------//
/*!
 * Sample Cerenkov photons from the distribution.
 */
template<class Generator>
CELER_FUNCTION void CerenkovGenerator::operator()(Generator& rng)
{
    NonuniformGrid<real_type> energy_grid(refractive_index_.data.grid,
                                          refractive_index_.reals);
    real_type e_min = energy_grid.front();
    real_type e_max = energy_grid.back();

    // TODO: rename GenericXsCalculator to something more generic?
    GenericXsCalculator calc_refractive_index(refractive_index_.data,
                                              refractive_index_.reals);
    real_type inv_beta = 2 * constants::c_light
                         / (dist_.pre.velocity + dist_.post.velocity);
    real_type cos_max = inv_beta / calc_refractive_index(e_max);
    real_type sin_max_sq = (1 - cos_max) * (1 + cos_max);

    // Calculate changes over the step
    Real3 delta_pos = dist_.post.pos;
    for (int j = 0; j < 3; ++j)
    {
        delta_pos[j] -= dist_.pre.pos[j];
    }
    real_type delta_num_photons = dist_.post.mean_num_photons
                                  - dist_.pre.mean_num_photons;
    real_type delta_velocity = dist_.post.velocity - dist_.pre.velocity;

    for (auto i : range(dist_.num_photons))
    {
        // Sample energy and direction
        real_type energy;
        real_type cos_theta;
        UniformRealDistribution<real_type> sample_energy(e_min, e_max);
        do
        {
            energy = sample_energy(rng);
            cos_theta = inv_beta / calc_refractive_index(energy);
        } while (!BernoulliDistribution(
            sin_max_sq / ((1 - cos_theta) * (1 + cos_theta)))(rng));

        photons_[i].energy = units::MevEnergy(energy);

        UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);
        real_type phi = sample_phi(rng);
        photons_[i].direction = rotate(from_spherical(cos_theta, phi),
                                       make_unit_vector(delta_pos));

        // Sample time
        real_type u;
        UniformRealDistribution<real_type> sample_num_photons(
            0, max(dist_.pre.mean_num_photons, dist_.post.mean_num_photons));
        do
        {
            u = generate_canonical(rng);
        } while (sample_num_photons(rng)
                 > dist_.pre.mean_num_photons + u * delta_num_photons);
        real_type delta_time
            = u * dist_.step_length
              / (dist_.pre.velocity + u * real_type(0.5) * delta_velocity);
        photons_[i].time = dist_.time + delta_time;

        // Sample position
        photons_[i].position = dist_.pre.pos;
        for (int j = 0; j < 3; ++j)
        {
            photons_[i].position[j] += u * delta_pos[j];
        }

        // TODO: polarization
        // TODO: save parent ID?
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
