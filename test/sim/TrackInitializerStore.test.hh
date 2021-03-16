//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackInitializerStore.test.hh
//---------------------------------------------------------------------------//
#include "base/DeviceVector.hh"
#include "physics/base/Interaction.hh"
#include "base/StackAllocator.hh"
#include "physics/base/Secondary.hh"
#include "sim/SimTrackView.hh"
#include "sim/TrackInterface.hh"
#include "sim/TrackInitializerInterface.hh"
#include <vector>

namespace celeritas_test
{
using namespace celeritas;

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Interactor
struct Interactor
{
    CELER_FUNCTION Interactor(StackAllocator<Secondary>& allocate_secondaries,
                              size_type                  alloc_size,
                              char                       alive)
        : allocate_secondaries(allocate_secondaries)
        , alloc_size(alloc_size)
        , alive(alive)
    {
    }

    CELER_FUNCTION Interaction operator()()
    {
        Interaction result;

        // Kill the particle
        if (!alive)
        {
            result = Interaction::from_absorption();
        }

        // Create secondaries
        if (alloc_size > 0)
        {
            Secondary* allocated = this->allocate_secondaries(alloc_size);
            if (!allocated)
            {
                return Interaction::from_failure();
            }

            result.secondaries = {allocated, alloc_size};
            for (auto& secondary : result.secondaries)
            {
                secondary.particle_id = ParticleId(0);
                secondary.energy      = units::MevEnergy(5.);
                secondary.direction   = {1., 0., 0.};
            }
        }

        return result;
    }

    StackAllocator<Secondary>& allocate_secondaries;
    size_type               alloc_size;
    char                    alive;
};

//! Input data
struct ITTestInputPointers
{
    Span<const size_type> alloc_size;
    Span<const char>      alive;
};

struct ITTestInput
{
    ITTestInput(std::vector<size_type>& host_alloc_size,
                std::vector<char>&      host_alive);

    ITTestInputPointers device_pointers();

    // Number of secondaries each track will produce
    DeviceVector<size_type> alloc_size;
    // Whether the track is alive
    DeviceVector<char> alive;
};

//! Output data
struct ITTestOutput
{
    std::vector<unsigned int> track_id;
    std::vector<unsigned int> initializer_id;
    std::vector<size_type>    vacancy;
};

using SecondaryAllocatorPointers
    = celeritas::StackAllocatorData<Secondary,
                                    celeritas::Ownership::reference,
                                    celeritas::MemSpace::device>;

//---------------------------------------------------------------------------//
//! Launch a kernel to produce secondaries and apply cutoffs
void interact(StatePointers              states,
              SecondaryAllocatorPointers secondaries,
              ITTestInputPointers        input);

//---------------------------------------------------------------------------//
//! Launch a kernel to get the track IDs of the initialized tracks
std::vector<unsigned int> tracks_test(StatePointers states);

//---------------------------------------------------------------------------//
//! Launch a kernel to get the track IDs of the track initializers created from
//! primaries or secondaries
std::vector<unsigned int> initializers_test(TrackInitializerPointers inits);

//---------------------------------------------------------------------------//
//! Launch a kernel to get the indices of the vacant slots in the track vector
std::vector<size_type> vacancies_test(TrackInitializerPointers inits);

//---------------------------------------------------------------------------//
} // namespace celeritas_test
