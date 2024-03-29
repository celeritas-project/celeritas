//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/data/CollectionStateStore.hh"
#include "corecel/data/StackAllocator.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/OpticalDistributionData.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/track/SimData.hh"

#include "DiagnosticRngEngine.hh"
#include "Test.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// FWD declarations
class ParticleParams;
class ParticleTrackView;
class SimParams;
class SimTrackView;
class PDGNumber;

//---------------------------------------------------------------------------//
/*!
 * Test harness base class for optical physics.
 *
 * Constructs particle params, particle track views, and add some
 * functionality for multiple tests.
 *
 * May be expanded to encompass material data if needed.
 */
namespace test
{
class OpticalTestBase : public Test
{
  public:
    //!@{
    //! \name Type aliases
    using DistributionAllocator = StackAllocator<OpticalDistributionData>;
    //!@}

  public:
    //!@{
    //! Initialize and destroy
    OpticalTestBase();
    ~OpticalTestBase();
    //!@}

    //!@{
    //! Distribution stack storage and access
    void resize_distributions(int count);
    DistributionAllocator& distribution_allocator()
    {
        CELER_EXPECT(distribution_allocator_);
        return *distribution_allocator_;
    }
    //!@}

    //! Initialize particle state data with given energy
    ParticleTrackView
    make_particle_track_view(units::MevEnergy energy, PDGNumber pdg);

    //! Initialize sim track state
    SimTrackView make_sim_track_view(real_type step_len_cm);

    //! Get particle params data
    std::shared_ptr<ParticleParams> const& particle_params() const
    {
        return particle_params_;
    }

    //! Get SimTrackView
    std::shared_ptr<SimParams> const& sim_params() const
    {
        return sim_params_;
    }

  private:
    template<template<Ownership, MemSpace> class S>
    using StateStore = CollectionStateStore<S, MemSpace::host>;
    template<Ownership W, MemSpace M>
    using DistributionStackData
        = StackAllocatorData<OpticalDistributionData, W, M>;

    std::shared_ptr<ParticleParams> particle_params_;
    std::shared_ptr<SimParams> sim_params_;
    std::shared_ptr<DistributionAllocator> distribution_allocator_;

    StateStore<ParticleStateData> particle_state_;
    StateStore<SimStateData> sim_state_;
    StateStore<DistributionStackData> distributions_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
