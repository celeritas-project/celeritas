//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CerenkovPreGenerator.cc
//---------------------------------------------------------------------------//
#include "CerenkovPreGenerator.hh"

#include "corecel/Assert.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/random/distribution/PoissonDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from particle, material, and optical step collection data.
 */
CerenkovPreGenerator::CerenkovPreGenerator(
    std::shared_ptr<ParticleTrackView> particle_view,
    NativeCRef<OpticalPropertyData> const& properties,
    NativeCRef<CerenkovData> const& shared,
    OpticalMaterialId mat_id,
    OpticalStepCollectorData optical_step)
    : particle_view_(particle_view)
    , mat_id_(mat_id)
    , step_collector_data_(optical_step)
    , dndx_calculator_(properties, shared, mat_id, particle_view->charge())
{
    CELER_EXPECT(mat_id_);
    CELER_EXPECT(step_collector_data_);
}

//---------------------------------------------------------------------------//
/*!
 * Return an \c OpticalDistributionData object. If no photons are sampled, an
 * empty object is returned and can be verified via its own operator bool.
 *
 * The number of photons is sampled from a Poisson distribution with a mean
 * \f[
   \langle n \rangle = \ell_\text{step} \frac{dN}{dx}
 * \f]
 * where \f$ \ell_\text{step} \f$ is the step length.
 */
template<class Generator>
OpticalDistributionData CELER_FUNCTION
CerenkovPreGenerator::operator()(Generator& rng)
{
    auto const& pre = step_collector_data_.points[StepPoint::pre];
    auto const& post = step_collector_data_.points[StepPoint::post];
    real_type avg_beta = real_type{0.5}
                         * (pre.speed.value() + post.speed.value());

    real_type photons_per_unit_len
        = dndx_calculator_(units::LightSpeed{avg_beta});
    if (photons_per_unit_len == 0)
    {
        // Not an optical material or this step is below production threshold
        return OpticalDistributionData{};
    }

    // Sample number of photons from a Poisson distribution
    real_type sampled_num_photons = PoissonDistribution<real_type>(
        photons_per_unit_len * step_collector_data_.step_length)(rng);

    // Populate optical distribution data
    OpticalDistributionData data;
    data.num_photons = sampled_num_photons;
    data.charge = particle_view_->charge();
    data.step_length = step_collector_data_.step_length;
    data.time = step_collector_data_.time;
    data.points = step_collector_data_.points;
    data.material = mat_id_;

    CELER_ENSURE(data);
    return data;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
