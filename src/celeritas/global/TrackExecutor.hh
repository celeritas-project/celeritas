//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/TrackExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/track/TrackFunctors.hh"

#include "CoreTrackData.hh"
#include "CoreTrackDataFwd.hh"
#include "CoreTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Call a \c CoreTrackView executor for a given ThreadId.
 *
 * This class can be used to call a functor that applies to \c CoreTrackView
 * using a \c ThreadId, so that the tracks can be easily looped over as a
 * group on CPU or GPU. It applies a remapping from \em thread to \em slot if
 * the tracks are sorted. Otherwise, thread and track slot have the same
 * numerical value.
 *
 * This is primarily used by \c ActionLauncher .
 *
 * \code
void foo_kernel(CoreParamsPtr const params,
                CoreStatePtr const state)
{
    TrackExecutor execute{params, state, MyTrackApplier{}};

    for (auto tid : range(ThreadID{123}))
    {
        step(tid);
    }
}
\endcode
 *
 * \todo Rename to ThreadExecutor. The template parameter, which must operate
 * on a core track view, is a track executor.
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

    //! Call the underlying function, using indirection array if needed
    CELER_FUNCTION void operator()(ThreadId thread)
    {
        CELER_EXPECT(thread < state_->size());
        CoreTrackView track(*params_, *state_, thread);
        return execute_track_(track);
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
        CoreTrackView track(*params_, *state_, thread);
        if (!applies_(track))
        {
            return;
        }

        // NOTE: "return value type" error means the executor function is
        // incorrectly returning a value
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
 * Return a track executor that only applies to active, non-errored tracks.
 */
template<class T>
inline CELER_FUNCTION decltype(auto)
make_active_track_executor(CoreParamsPtr<MemSpace::native> params,
                           CoreStatePtr<MemSpace::native> const& state,
                           T&& apply_track)
{
    return ConditionalTrackExecutor{
        params, state, AppliesValid{}, celeritas::forward<T>(apply_track)};
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
                                    IsStepActionEqual{action},
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
                                    IsAlongStepActionEqual{action},
                                    celeritas::forward<T>(apply_track)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
