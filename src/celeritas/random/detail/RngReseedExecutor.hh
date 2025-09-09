//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/detail/RngReseedExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/random/data/RngData.hh"
#include "corecel/random/engine/RngEngine.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Reinitialize a track's random state from a unique event ID.
 */
class RngReseedExecutor
{
  public:
    using ParamsCRef = NativeCRef<RngParamsData>;
    using StateRef = NativeRef<RngStateData>;

  public:
    // Construct with state and event ID
    inline CELER_FUNCTION
    RngReseedExecutor(ParamsCRef const&, StateRef const&, UniqueEventId id);

    // Initialize the given track slot
    inline CELER_FUNCTION void operator()(TrackSlotId tid) const;

    //! Initialize from the given thread
    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        return (*this)(TrackSlotId{tid.unchecked_get()});
    }

  private:
    ParamsCRef const params_;
    StateRef const state_;
    UniqueEventId::size_type stride_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with state and event ID.
 */
CELER_FUNCTION RngReseedExecutor::RngReseedExecutor(ParamsCRef const& params,
                                                    StateRef const& state,
                                                    UniqueEventId id)
    : params_{params}, state_{state}, stride_{id.unchecked_get() * state.size()}
{
    CELER_EXPECT(params_ && state_);
    CELER_EXPECT(id);
    static_assert(sizeof(ull_int) == sizeof(UniqueEventId::size_type));
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the given track slot.
 */
CELER_FUNCTION void RngReseedExecutor::operator()(TrackSlotId tid) const
{
    CELER_EXPECT(tid < state_.size());
    RngEngine::Initializer_t init;
    init.seed = params_.seed;
    init.subsequence = stride_ + tid.unchecked_get();

    RngEngine engine(params_, state_, tid);
    engine = init;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
