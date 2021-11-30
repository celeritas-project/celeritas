//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimpleUnitTracker.test.hh
//---------------------------------------------------------------------------//

#include "orange/Data.hh"
#include "orange/universes/SimpleUnitTracker.hh"

namespace celeritas_test
{
using celeritas::MemSpace;
using celeritas::Ownership;
using ParamsDeviceRef
    = celeritas::OrangeParamsData<Ownership::const_reference, MemSpace::device>;
using StateDeviceRef
    = celeritas::OrangeStateData<Ownership::reference, MemSpace::device>;

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//

//! Initialize
template<MemSpace M = MemSpace::native>
struct InitializingLauncher
{
    using ThreadId = celeritas::ThreadId;
    using SenseId  = celeritas::ItemId<celeritas::Sense>;

    celeritas::OrangeParamsData<Ownership::const_reference, M> params;
    celeritas::OrangeStateData<Ownership::reference, M>        states;

    CELER_CONSTEXPR_FUNCTION SenseId begin_sense_id(ThreadId tid) const
    {
        return SenseId(params.scalars.max_faces * tid.unchecked_get());
    }

    CELER_CONSTEXPR_FUNCTION ThreadId next_thread(ThreadId tid) const
    {
        return ThreadId{tid.unchecked_get() + 1};
    }

    CELER_FUNCTION void operator()(ThreadId tid) const
    {
        using celeritas::ItemRange;
        using celeritas::Sense;
        using celeritas::SimpleUnitTracker;

        // Create local state from global memory
        celeritas::detail::LocalState lstate;
        lstate.pos         = states.pos[tid];
        lstate.dir         = states.dir[tid];
        lstate.volume      = states.vol[tid];
        lstate.surface     = {states.surf[tid], states.sense[tid]};
        lstate.temp_sense  = states.temp_sense[ItemRange<Sense>(
            begin_sense_id(tid), begin_sense_id(next_thread(tid)))];

        // Instantiate tracker and initialize
        SimpleUnitTracker tracker(this->params);
        auto              init = tracker.initialize(lstate);

        // Update state with post-initialization result
        states.vol[tid]   = init.volume;
        states.surf[tid]  = init.surface.id();
        states.sense[tid] = init.surface.unchecked_sense();
    }
};

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device
void test_initialize(const ParamsDeviceRef&, const StateDeviceRef&);

#if !CELERITAS_USE_CUDA
void test_initialize(const ParamsDeviceRef&, const StateDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas_test
