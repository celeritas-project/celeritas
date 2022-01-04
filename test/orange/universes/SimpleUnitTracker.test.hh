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
using celeritas::ThreadId;
using LocalState = celeritas::detail::LocalState;

template<MemSpace M>
using ParamsRef = celeritas::OrangeParamsData<Ownership::const_reference, M>;
template<MemSpace M>
using StateRef = celeritas::OrangeStateData<Ownership::reference, M>;

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//

template<class T>
CELER_CONSTEXPR_FUNCTION celeritas::ItemRange<T> build_range(
    celeritas::size_type stride,
    ThreadId tid)
{
    CELER_EXPECT(tid);
    using IdT = celeritas::ItemId<T>;
    auto t = tid.unchecked_get();
    return {IdT{t * stride}, IdT{(t + 1) * stride}};
}

template<MemSpace M = MemSpace::native>
inline CELER_FUNCTION LocalState build_local_state(
    ParamsRef<M> params,
    StateRef<M>        states,
    ThreadId tid)
{
    using namespace celeritas;

    // Create local state from global memory
    LocalState lstate;
    lstate.pos     = states.pos[tid];
    lstate.dir     = states.dir[tid];
    lstate.volume  = states.vol[tid];
    lstate.surface = {states.surf[tid], states.sense[tid]};

    const size_type max_faces = params.scalars.max_faces;
    lstate.temp_sense = states.temp_sense[build_range<Sense>(max_faces, tid)];

    const size_type max_isect = params.scalars.max_intersections;
    lstate.temp_next.face
        = states.temp_face[build_range<FaceId>(max_isect, tid)].data();
    lstate.temp_next.distance
        = states.temp_distance[build_range<real_type>(max_isect, tid)].data();
    lstate.temp_next.isect
        = states.temp_isect[build_range<size_type>(max_isect, tid)].data();
    lstate.temp_next.size = max_isect;
    return lstate;
}

//! Initialization heuristic
template<MemSpace M = MemSpace::native>
struct InitializingLauncher
{
    ParamsRef<M> params;
    StateRef<M>        states;

    CELER_FUNCTION void operator()(ThreadId tid) const
    {
        // Instantiate tracker and initialize
        celeritas::SimpleUnitTracker tracker(this->params);
        auto lstate = build_local_state(params, states, tid);
        auto init = tracker.initialize(lstate);

        // Update state with post-initialization result
        states.vol[tid]   = init.volume;
        states.surf[tid]  = init.surface.id();
        states.sense[tid] = init.surface.unchecked_sense();

        lstate.volume = init.volume;
        auto isect = tracker.intersect(lstate);

        // BOGUS
        states.surf[tid]  = isect.surface.id();
        states.sense[tid] = isect.surface.unchecked_sense();
    }
};

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device
void test_initialize(const ParamsRef<MemSpace::device>&, const StateRef<MemSpace::device>&);

#if !CELERITAS_USE_CUDA
inline void test_initialize(const ParamsRef<MemSpace::device>&, const StateRef<MemSpace::device>&)
{
    CELER_NOT_CONFIGURED("CUDA");
}

#endif

//---------------------------------------------------------------------------//
} // namespace celeritas_test
