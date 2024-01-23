//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/TrackExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/CoreTrackDataFwd.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/track/SimTrackView.hh"

#include "CoreTrackData.hh"
#include "detail/TrackExecutorImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Call a \c CoreTrackView executor for a given ThreadId.
 *
 * This class can be used to call a functor that applies to \c CoreTrackView
 * using a \c ThreadId, so that the tracks can be easily looped over as a
 * group on CPU or GPU.
 *
 * This is primarily used by \c ActionLauncher .
 *
 * TODO: maybe rename this \c ThreadExecutor (since it takes a thread?) and
 * call the embedded class a \c TrackExecutor (since it takes a CoreTrackView?)
 *
 * \code
void foo_kernel(CoreParamsPtr const params,
                CoreStatePtr const state)
{
    TrackExecutor execute{params, state, MyTrackApplier{}};

    for (auto tid : range(ThreadID{123}))
    {
        execute(tid);
    }
}
\endcode
 */
template<class T>
class TrackExecutor
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
    TrackExecutor(ParamsPtr params, StatePtr state, T&& execute_track)
        : params_{params}
        , state_{state}
        , execute_track_{celeritas::forward<T>(execute_track)}
    {
    }

    //! Call the underlying function using the core track for this thread.
    CELER_FUNCTION void operator()(ThreadId thread)
    {
        CELER_EXPECT(thread < state_->size());
        CoreTrackView const track(*params_, *state_, thread);
        return execute_track_(track);
    }

  private:
    ParamsPtr const params_;
    StatePtr const state_;
    T execute_track_;
};

//---------------------------------------------------------------------------//
/*!
 * Launch the track only when a certain condition applies.
 *
 * The condition C must have the signature \code
 * <bool(SimTrackView const&)>
  \endcode
 *
 * see \c make_active_track_executor for an example where this is used to apply
 * only to active (or killed) tracks.
 */
template<class C, class T>
class ConditionalTrackExecutor
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
    ConditionalTrackExecutor(ParamsPtr params,
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
    CELER_FUNCTION void operator()(ThreadId thread)
    {
        CELER_EXPECT(thread < state_->size());
        CoreTrackView const track(*params_, *state_, thread);
        if (!applies_(track.make_sim_view()))
        {
            return;
        }

        return execute_track_(track);
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
CELER_FUNCTION TrackExecutor(CoreParamsPtr<MemSpace::native>,
                             CoreStatePtr<MemSpace::native>,
                             T&&) -> TrackExecutor<T>;

template<class C, class T>
CELER_FUNCTION ConditionalTrackExecutor(CoreParamsPtr<MemSpace::native>,
                                        CoreStatePtr<MemSpace::native>,
                                        C&&,
                                        T&&) -> ConditionalTrackExecutor<C, T>;

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Return a track executor that only applies to alive tracks.
 */
template<class T>
inline CELER_FUNCTION decltype(auto)
make_active_track_executor(CoreParamsPtr<MemSpace::native> params,
                           CoreStatePtr<MemSpace::native> const& state,
                           T&& apply_track)
{
    return ConditionalTrackExecutor{params,
                                    state,
                                    detail::AppliesActive{},
                                    celeritas::forward<T>(apply_track)};
}

//---------------------------------------------------------------------------//
/*!
 * Return a track executor that only applies if the action ID matches.
 *
 * \note This should generally only be used for post-step actions and other
 * cases where the IDs *explicitly* are set. Many explicit actions apply to all
 * threads, active or not.
 */
template<class T>
inline CELER_FUNCTION decltype(auto)
make_action_track_executor(CoreParamsPtr<MemSpace::native> params,
                           CoreStatePtr<MemSpace::native> state,
                           ActionId action,
                           T&& apply_track)
{
    CELER_EXPECT(action);
    return ConditionalTrackExecutor{params,
                                    state,
                                    detail::IsStepActionEqual{action},
                                    celeritas::forward<T>(apply_track)};
}

//---------------------------------------------------------------------------//
/*!
 * Return a track executor that only applies for the given along-step action.
 */
template<class T>
inline CELER_FUNCTION decltype(auto)
make_along_step_track_executor(CoreParamsPtr<MemSpace::native> params,
                               CoreStatePtr<MemSpace::native> state,
                               ActionId action,
                               T&& apply_track)
{
    CELER_EXPECT(action);
    return ConditionalTrackExecutor{params,
                                    state,
                                    detail::IsAlongStepActionEqual{action},
                                    celeritas::forward<T>(apply_track)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
