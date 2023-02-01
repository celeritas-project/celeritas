//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/SimpleUnitTracker.test.hh
//---------------------------------------------------------------------------//

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "orange/OrangeData.hh"
#include "orange/detail/LevelStateAccessor.hh"
#include "orange/univ/SimpleUnitTracker.hh"

namespace celeritas
{
namespace test
{
using LocalState = detail::LocalState;

template<MemSpace M>
using ParamsRef = OrangeParamsData<Ownership::const_reference, M>;
template<MemSpace M>
using StateRef = OrangeStateData<Ownership::reference, M>;

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//

template<class T>
CELER_CONSTEXPR_FUNCTION ItemRange<T>
build_range(size_type stride, ThreadId tid)
{
    CELER_EXPECT(tid);
    using IdT = ItemId<T>;
    auto t = tid.unchecked_get();
    return {IdT{t * stride}, IdT{(t + 1) * stride}};
}

template<MemSpace M = MemSpace::native>
inline CELER_FUNCTION LocalState build_local_state(ParamsRef<M> params,
                                                   StateRef<M> states,
                                                   ThreadId tid)
{
    // Create local state from global memory
    LocalState lstate;

    LevelStateAccessor lsa(&states, tid, LevelId{0});
    lstate.pos = lsa.pos();
    lstate.dir = lsa.dir();
    lstate.volume = lsa.vol();

    lstate.surface = {lsa.surf(), lsa.sense()};

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
    StateRef<M> states;

    CELER_FUNCTION void operator()(ThreadId tid) const
    {
        // Instantiate tracker and initialize
        SimpleUnitTracker tracker(this->params, SimpleUnitId{0});
        auto lstate = build_local_state(params, states, tid);
        auto init = tracker.initialize(lstate);

        // Update state with post-initialization result

        // TODO: for multiuniverses tests, we acutally have to iterative
        // through daughter universes to assign the level and volume
        LevelStateAccessor lsa(&states, tid, LevelId{0});
        lsa.set_vol(init.volume);

        lsa.set_surf(init.surface.id());
        lsa.set_sense(init.surface.unchecked_sense());

        lstate.volume = init.volume;
        auto isect = tracker.intersect(lstate);

        // BOGUS
        lsa.set_surf(isect.surface.id());
        lsa.set_sense(flip_sense(isect.surface.unchecked_sense()));
    }
};

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device
void test_initialize(ParamsRef<MemSpace::device> const&,
                     StateRef<MemSpace::device> const&);

#if !CELER_USE_DEVICE
inline void test_initialize(ParamsRef<MemSpace::device> const&,
                            StateRef<MemSpace::device> const&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
