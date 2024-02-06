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
    ParticleTrackView particle_view,
    NativeCRef<OpticalPropertyData> const& properties,
    NativeCRef<CerenkovData> const& shared,
    OpticalMaterialId mat_id,
    OpticalStepCollectorData optical_step)
    : particle_view_(particle_view)
    , mat_id_(mat_id)
    , step_collector_data_(optical_step)
    , dndx_calculator_(properties, shared, mat_id, particle_view.charge())
{
    CELER_EXPECT(step_collector_data_);
}

//---------------------------------------------------------------------------//
/*!
 * Return an \c OpticalDistributionData . If no photons are sampled, an empty
 * object is returned and can be verified via its own operator bool.
 */
template<class Generator>
OpticalDistributionData CELER_FUNCTION
CerenkovPreGenerator::operator()(Generator& rng)
{
    OpticalDistributionData data;

    auto const avg_beta = units::LightSpeed{
        real_type{0.5}
        * ((step_collector_data_.points[StepPoint::pre].speed.value())
           + step_collector_data_.points[StepPoint::post].speed.value())};

    real_type avg_num_photons = dndx_calculator_(avg_beta);
    if (avg_num_photons == 0)
    {
        // Not an optical material or particle is below threshold
        return data;
    }

    // Sample number of photons within a step from a Poisson distribution
    // G4 samples it using step length (se G4Cerenkov::PostStepDoIt)
    real_type sampled_num_photons = PoissonDistribution<real_type>(
        avg_num_photons * step_collector_data_.step_length)(rng);

    // Populate optical distribution data
    data.num_photons = sampled_num_photons;
    data.charge = particle_view_.charge();
    data.step_length = step_collector_data_.step_length;
    data.time = step_collector_data_.time;
    data.points = step_collector_data_.points;
    data.material = mat_id_;

    CELER_ENSURE(data);
    return data;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
