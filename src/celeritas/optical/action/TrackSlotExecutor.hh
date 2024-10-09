//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/TrackSlotExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/optical/CoreTrackData.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/SimTrackView.hh"
#include "celeritas/track/SimFunctors.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Transform a thread or track slot ID into a core track view.
 *
 * This class can be used to call a functor that applies to \c
 * optical::CoreTrackView using a \c TrackSlotId, so that the tracks can be
 * easily looped over as a group on CPU or GPU.
 *
 * To facilitate kernel launches, the class can also directly map \c ThreadId
 * to \c TrackSlotId, which will have the same numerical value because optical
 * photons do not implement sorting.
 */
template<class T>
class TrackSlotExecutor
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsPtr = CoreParamsPtr<MemSpace::native>;
    using StatePtr = CoreStatePtr<MemSpace::native>;
    using Applier = T;
    //!@}

  public:
    //! Construct with core data and executor
    CELER_FUNCTION
    TrackSlotExecutor(ParamsPtr params, StatePtr state, T&& execute_track)
        : params_{params}
        , state_{state}
        , execute_track_{celeritas::forward<T>(execute_track)}
    {
    }

    //! Call the underlying function based on the index in the state array
    CELER_FUNCTION void operator()(TrackSlotId ts)
    {
        CELER_EXPECT(ts < state_->size());
        CoreTrackView track(*params_, *state_, ts);
        return execute_track_(track);
    }

    //! Call the underlying function using the thread index
    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId thread)
    {
        // For optical photons, thread index maps exactly to
        return (*this)(TrackSlotId{thread.unchecked_get()});
    }

  private:
    ParamsPtr const params_;
    StatePtr const state_;
    T execute_track_;
};

//---------------------------------------------------------------------------//
/*!
 * Launch the track only when a certain condition applies to the sim state.
 *
 * The condition \c C must have the signature \code
 * (SimTrackView const&) -> bool
  \endcode
 *
 * see \c make_active_track_executor for an example where this is used to apply
 * only to active (or killed) tracks.
 */
template<class C, class T>
class ConditionalTrackSlotExecutor
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsPtr = CoreParamsPtr<MemSpace::native>;
    using StatePtr = CoreStatePtr<MemSpace::native>;
    using Applier = T;
    //!@}

  public:
    //! Construct with condition and operator
    CELER_FUNCTION
    ConditionalTrackSlotExecutor(ParamsPtr params,
                                 StatePtr state,
                                 C&& applies,
                                 T&& execute_track)
        : params_{params}
        , state_{state}
        , applies_{celeritas::forward<C>(applies)}
        , execute_track_{celeritas::forward<T>(execute_track)}
    {
    }

    //! Launch the given thread if the track meets the condition
    CELER_FUNCTION void operator()(TrackSlotId ts)
    {
        CELER_EXPECT(ts < state_->size());
        CoreTrackView track(*params_, *state_, ts);
        if (!applies_(track.sim()))
        {
            return;
        }

        return execute_track_(track);
    }

    //! Call the underlying function using the thread index
    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId thread)
    {
        // For optical photons, thread index maps exactly to
        return (*this)(TrackSlotId{thread.unchecked_get()});
    }

  private:
    ParamsPtr const params_;
    StatePtr const state_;
    C applies_;
    T execute_track_;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class T>
CELER_FUNCTION TrackSlotExecutor(CoreParamsPtr<MemSpace::native>,
                                 CoreStatePtr<MemSpace::native>,
                                 T&&) -> TrackSlotExecutor<T>;

template<class C, class T>
CELER_FUNCTION
ConditionalTrackSlotExecutor(CoreParamsPtr<MemSpace::native>,
                             CoreStatePtr<MemSpace::native>,
                             C&&,
                             T&&) -> ConditionalTrackSlotExecutor<C, T>;

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Return a track executor that only applies to active, non-errored tracks.
 */
template<class T>
inline CELER_FUNCTION decltype(auto)
make_active_thread_executor(CoreParamsPtr<MemSpace::native> params,
                            CoreStatePtr<MemSpace::native> const& state,
                            T&& apply_track)
{
    return ConditionalTrackSlotExecutor{
        params, state, AppliesValid{}, celeritas::forward<T>(apply_track)};
}

//---------------------------------------------------------------------------//
/*!
 * Return a track executor that only applies if the action ID matches.
 *
 * \note This should generally only be used for post-step actions and other
 * cases where the IDs \em explicitly are set. Many explicit actions apply to
 * all threads, active or not.
 */
template<class T>
inline CELER_FUNCTION decltype(auto)
make_action_thread_executor(CoreParamsPtr<MemSpace::native> params,
                            CoreStatePtr<MemSpace::native> state,
                            ActionId action,
                            T&& apply_track)
{
    CELER_EXPECT(action);
    return ConditionalTrackSlotExecutor{params,
                                        state,
                                        IsStepActionEqual{action},
                                        celeritas::forward<T>(apply_track)};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
