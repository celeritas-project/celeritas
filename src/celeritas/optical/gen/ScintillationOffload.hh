//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/ScintillationOffload.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/random/distribution/NormalDistribution.hh"
#include "corecel/random/distribution/PoissonDistribution.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/track/SimTrackView.hh"

#include "GeneratorData.hh"
#include "OffloadData.hh"
#include "ScintillationData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample the number of scintillation photons to be generated.
 *
 * This populates the \c GeneratorDistributionData used by the \c
 * ScintillationGenerator to generate optical photons using post-step and
 * cached pre-step data.
 *
 * The mean number of photons is a product of the energy deposition and a
 * material-dependent yield fraction (photons per MeV). The actual number of
 * photons sampled is determined by sampling:
 * - for large (n > 10) mean yield, from a Gaussian distribution with a
 *   material-dependent spread, or
 * - for small yields, from a Poisson distribution.
 */
class ScintillationOffload
{
  public:
    // Construct with optical properties, scintillation, and step data
    inline CELER_FUNCTION
    ScintillationOffload(ParticleTrackView const& particle,
                         SimTrackView const& sim,
                         Real3 const& pos,
                         units::MevEnergy energy_deposition,
                         NativeCRef<ScintillationData> const& shared,
                         OffloadPreStepData const& step_data);

    // Gather the input data needed to sample scintillation photons
    template<class Generator>
    inline CELER_FUNCTION optical::GeneratorDistributionData
    operator()(Generator& rng);

  private:
    units::ElementaryCharge charge_;
    real_type step_length_;
    OffloadPreStepData const& pre_step_;
    optical::GeneratorStepData post_step_;
    NativeCRef<ScintillationData> const& shared_;
    real_type mean_num_photons_{0};

    static CELER_CONSTEXPR_FUNCTION real_type poisson_threshold()
    {
        return 10;
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with input parameters.
 */
CELER_FUNCTION ScintillationOffload::ScintillationOffload(
    ParticleTrackView const& particle,
    SimTrackView const& sim,
    Real3 const& pos,
    units::MevEnergy energy_deposition,
    NativeCRef<ScintillationData> const& shared,
    OffloadPreStepData const& step_data)
    : charge_(particle.charge())
    , step_length_(sim.step_length())
    , pre_step_(step_data)
    , post_step_({particle.speed(), pos})
    , shared_(shared)
{
    CELER_EXPECT(step_length_ > 0);
    CELER_EXPECT(shared_);
    CELER_EXPECT(pre_step_);

    if (shared_.scintillation_by_particle())
    {
        //! \todo Implement sampling for particles
        CELER_ASSERT_UNREACHABLE();
    }
    else
    {
        // Scintillation will be performed on materials only
        CELER_ASSERT(pre_step_.material < shared_.materials.size());
        auto const& material = shared_.materials[pre_step_.material];

        //! \todo Use visible energy deposition when Birks law is implemented
        if (material)
        {
            mean_num_photons_ = material.yield_per_energy
                                * energy_deposition.value();
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Collect the distribution data needed to sample scintillation photons.
 *
 * If no photons are sampled an empty object is returned.
 */
template<class Generator>
CELER_FUNCTION optical::GeneratorDistributionData
ScintillationOffload::operator()(Generator& rng)
{
    // Material-only sampling
    optical::GeneratorDistributionData result;
    if (mean_num_photons_ > poisson_threshold())
    {
        real_type sigma = shared_.resolution_scale[pre_step_.material]
                          * std::sqrt(mean_num_photons_);
        result.num_photons = static_cast<size_type>(clamp_to_nonneg(
            NormalDistribution<real_type>(mean_num_photons_, sigma)(rng)
            + real_type{0.5}));
    }
    else if (mean_num_photons_ > 0)
    {
        result.num_photons = static_cast<size_type>(
            PoissonDistribution<real_type>(mean_num_photons_)(rng));
    }

    if (result.num_photons > 0)
    {
        // Assign remaining data
        result.time = pre_step_.time;
        result.step_length = step_length_;
        result.charge = charge_;
        result.material = pre_step_.material;
        result.points[StepPoint::pre].speed = pre_step_.speed;
        result.points[StepPoint::pre].pos = pre_step_.pos;
        result.points[StepPoint::post] = post_step_;
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
