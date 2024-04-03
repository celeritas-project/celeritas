//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ScintillationPreGenerator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/random/distribution/NormalDistribution.hh"
#include "celeritas/random/distribution/PoissonDistribution.hh"
#include "celeritas/track/SimTrackView.hh"

#include "OpticalDistributionData.hh"
#include "OpticalPropertyData.hh"
#include "ScintillationData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample the number of Scintillation photons to be generated by
 * \c ScintillationGenerator and populate \c OpticalDistributionData values
 * using Param and State data.
 */
class ScintillationPreGenerator
{
  public:
    // Placeholder for data that is not available through Views
    // TODO: Merge Cerenkov and Scintillation pre-gen data into one struct
    struct OpticalPreGenStepData
    {
        real_type time{};  //!< Pre-step time
        units::MevEnergy energy_dep;  //!< Step energy deposition
        EnumArray<StepPoint, OpticalStepData> points;  //!< Pre- and post-steps

        //! Check whether the data are assigned
        explicit CELER_FUNCTION operator bool() const
        {
            return energy_dep > zero_quantity()
                   && points[StepPoint::pre].speed > zero_quantity();
        }
    };

    // Construct with optical properties, scintillation, and step data
    inline CELER_FUNCTION
    ScintillationPreGenerator(ParticleTrackView const& particle_view,
                              SimTrackView const& sim_view,
                              OpticalMaterialId material,
                              NativeCRef<ScintillationData> const& shared,
                              OpticalPreGenStepData const& step_data);

    // Populate an optical distribution data for the Scintillation Generator
    template<class Generator>
    inline CELER_FUNCTION OpticalDistributionData operator()(Generator& rng);

  private:
    units::ElementaryCharge charge_;
    real_type step_len_;
    OpticalMaterialId mat_id_;
    NativeCRef<ScintillationData> const& shared_;
    OpticalPreGenStepData step_;
    real_type mean_num_photons_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with input parameters.
 */
CELER_FUNCTION ScintillationPreGenerator::ScintillationPreGenerator(
    ParticleTrackView const& particle_view,
    SimTrackView const& sim_view,
    OpticalMaterialId material,
    NativeCRef<ScintillationData> const& shared,
    OpticalPreGenStepData const& step_data)
    : charge_(particle_view.charge())
    , step_len_(sim_view.step_length())
    , mat_id_(material)
    , shared_(shared)
    , step_(step_data)
{
    CELER_EXPECT(step_len_ > 0);
    CELER_EXPECT(mat_id_);
    CELER_EXPECT(shared_);
    CELER_EXPECT(step_);

    if (shared_.scintillation_by_particle())
    {
        // TODO: implement sampling for particles, assert particle data, and
        // cache mean number of photons
        CELER_ASSERT_UNREACHABLE();
    }
    else
    {
        // Scintillation will be performed on materials only
        CELER_EXPECT(!shared_.materials.empty()
                     && mat_id_ < shared_.materials.size());
        auto const& material = shared_.materials[mat_id_];
        CELER_ASSERT(material);
        // TODO: Use visible energy deposition when Birks law is implemented
        mean_num_photons_ = material.yield_per_energy
                            * step_.energy_dep.value();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Return an \c OpticalDistributionData object. If no photons are sampled, an
 * empty object is returned and can be verified via its own operator bool.
 */
template<class Generator>
CELER_FUNCTION OpticalDistributionData
ScintillationPreGenerator::operator()(Generator& rng)
{
    // Material-only sampling
    OpticalDistributionData result;
    if (mean_num_photons_ > 10)
    {
        real_type sigma = shared_.resolution_scale[mat_id_]
                          * std::sqrt(mean_num_photons_);
        result.num_photons = clamp_to_nonneg(
            NormalDistribution<real_type>(mean_num_photons_, sigma)(rng)
            + real_type{0.5});
    }
    else
    {
        result.num_photons
            = PoissonDistribution<real_type>(mean_num_photons_)(rng);
    }

    if (result.num_photons > 0)
    {
        // Assign remaining data
        result.charge = charge_;
        result.material = mat_id_;
        result.step_length = step_len_;
        result.time = step_.time;
        result.points = step_.points;
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
