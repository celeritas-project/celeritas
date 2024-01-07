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
#include "celeritas/grid/GenericXsCalculator.hh"
#include "celeritas/grid/XsGridData.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/GenerateCanonical.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

#include "CerenkovDistribution.hh"
#include "OpticalPrimary.hh"

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
    inline CELER_FUNCTION CerenkovGenerator(RefractiveIndex const& ref_index,
                                            CerenkovDistribution const& dist,
                                            Span<OpticalPrimary> photons);

    // Sample Cerenkov photons from the distribution
    template<class Generator>
    inline CELER_FUNCTION void operator()(Generator& rng);

  private:
    using UniformRealDist = UniformRealDistribution<real_type>;

    CerenkovDistribution const& dist_;
    Span<OpticalPrimary> photons_;
    // TODO: rename GenericXsCalculator to something more generic?
    GenericXsCalculator calc_refractive_index_;
    UniformRealDist sample_energy_;
    UniformRealDist sample_phi_;
    UniformRealDist sample_num_photons_;
    Real3 delta_pos_;
    Real3 dir_;
    real_type sin_max_sq_;
    real_type inv_beta_;
    real_type delta_num_photons_;
    real_type delta_velocity_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from optical properties and distribution parameters.
 */
CELER_FUNCTION
CerenkovGenerator::CerenkovGenerator(RefractiveIndex const& ref_index,
                                     CerenkovDistribution const& dist,
                                     Span<OpticalPrimary> photons)
    : dist_(dist)
    , photons_(photons)
    , calc_refractive_index_(ref_index.data, ref_index.reals)
    , sample_phi_(0, 2 * constants::pi)

{
    CELER_EXPECT(ref_index.data);
    CELER_EXPECT(dist_);
    CELER_EXPECT(photons_.size() == dist_.num_photons);

    NonuniformGrid<real_type> energy_grid(ref_index.data.grid, ref_index.reals);
    sample_energy_ = UniformRealDist(energy_grid.front(), energy_grid.back());

    sample_num_photons_ = UniformRealDist(
        0, max(dist_.pre.mean_num_photons, dist_.post.mean_num_photons));

    inv_beta_ = 2 * constants::c_light
                / (dist_.pre.velocity + dist_.post.velocity);
    real_type cos_max = inv_beta_ / calc_refractive_index_(energy_grid.back());
    sin_max_sq_ = (1 - cos_max) * (1 + cos_max);

    // Calculate changes over the step
    delta_pos_ = dist_.post.pos - dist_.pre.pos;
    dir_ = make_unit_vector(delta_pos_);
    delta_num_photons_ = dist_.post.mean_num_photons
                         - dist_.pre.mean_num_photons;
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
        do
        {
            energy = sample_energy_(rng);
            cos_theta = inv_beta_ / calc_refractive_index_(energy);
        } while (!BernoulliDistribution(
            sin_max_sq_ / ((1 - cos_theta) * (1 + cos_theta)))(rng));
        real_type phi = sample_phi_(rng);
        photons_[i].direction = rotate(from_spherical(cos_theta, phi), dir_);
        photons_[i].energy = units::MevEnergy(energy);

        // Determine polarization
        photons_[i].polarization = rotate(
            from_spherical(-std::sqrt(1 - ipow<2>(cos_theta)), phi), dir_);

        // Sample position and time
        real_type u;
        do
        {
            u = generate_canonical(rng);
        } while (sample_num_photons_(rng)
                 > dist_.pre.mean_num_photons + u * delta_num_photons_);
        real_type delta_time
            = u * dist_.step_length
              / (dist_.pre.velocity + u * real_type(0.5) * delta_velocity_);
        photons_[i].time = dist_.time + delta_time;
        photons_[i].position = dist_.pre.pos;
        axpy(u, delta_pos_, &photons_[i].position);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
