//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackInit.test.hh
//---------------------------------------------------------------------------//
#include <vector>

#include "celeritas_config.h"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/StackAllocator.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/Secondary.hh"
#include "celeritas/track/SimTrackView.hh"
#include "celeritas/track/TrackInitData.hh"

namespace celeritas
{
namespace test
{

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
        result.action    = Interaction::Action::unchanged;
        result.direction = {0, 0, 1};

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
    size_type                  alloc_size;
    char                       alive;
};

//! Input data
struct ITTestInputData
{
    Span<const size_type> alloc_size;
    Span<const char>      alive;
};

struct ITTestInput
{
    ITTestInput(std::vector<size_type>& host_alloc_size,
                std::vector<char>&      host_alive);

    ITTestInputData device_ref();

    // Number of secondaries each track will produce
    DeviceVector<size_type> alloc_size;
    // Whether the track is alive
    DeviceVector<char> alive;
};

//! Output data
struct ITTestOutput
{
    std::vector<unsigned int> track_ids;
    std::vector<int>          parent_ids;
    std::vector<unsigned int> init_ids;
    std::vector<size_type>    vacancies;
};

using SecondaryAllocatorData
    = StackAllocatorData<Secondary, Ownership::reference, MemSpace::device>;

//---------------------------------------------------------------------------//
//! Launch a kernel to produce secondaries and apply cutoffs
void interact(CoreStateDeviceRef states, ITTestInputData input);

#if !CELER_USE_DEVICE
inline void interact(CoreStateDeviceRef, ITTestInputData)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
