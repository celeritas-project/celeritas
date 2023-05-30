//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
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
 * Function-like class to execute a CoreTrackView-dependent function.
 *
 * This class should be used primarily by generated kernel functions:
 *
 * \code
__global__ void foo_kernel(CoreParamsPtr const params,
                           CoreParamsPtr const state,
                           OtherData const other)
{
    TrackExecutor execute{params, state, apply_to_track, OtherView{other}};
    execute(KernelParamCalculator::thread_id());
}
\endcode
 *
 * where the user-written code defines an inline function using the core track
 * view:
 *
 * \code
inline CELER_FUNCTION void apply_to_track(
    CoreTrackView const& track,
    OtherView const& other)
{
    // ...
}
   \endcode
 *
 * It will call the function with *all* thread slots, including inactive.
 *
 * \note The ThreadId can be no greater than the state size on CPU (since we
 * always loop over the exact range), but on GPU we automatically ignore thread
 * IDs outside the valid bounds to simplify the kernel.
 */
template<class... Ts>
class TrackExecutor
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsPtr = CoreParamsPtr<MemSpace::native>;
    using StatePtr = CoreStatePtr<MemSpace::native>;
    //!@}

  public:
    CELER_FUNCTION
    TrackExecutor(ParamsPtr params, StatePtr state, Ts... args)
        : params_{params}
        , state_{state}
        , execute_impl_{celeritas::forward<Ts>(args)...}
    {
    }

    CELER_FUNCTION void operator()(ThreadId thread) const
    {
        CELER_EXPECT(thread);
#if CELER_DEVICE_COMPILE
        if (!(thread < state_->size()))
        {
            return;
        }
#else
        CELER_EXPECT(thread < state_->size());
#endif
        CoreTrackView const track(*params_, *state_, thread);

        return execute_impl_(track);
    }

  private:
    ParamsPtr const params_;
    StatePtr const state_;
    detail::TrackExecutorImpl<Ts...> execute_impl_;
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
template<class C, class... Ts>
class ConditionalTrackExecutor
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsPtr = CoreParamsPtr<MemSpace::native>;
    using StatePtr = CoreStatePtr<MemSpace::native>;
    //!@}

  public:
    CELER_FUNCTION
    ConditionalTrackExecutor(ParamsPtr params,
                             StatePtr state,
                             C applies,
                             Ts... args)
        : params_{params}
        , state_{state}
        , applies_{celeritas::forward<C>(applies)}
        , execute_impl_{celeritas::forward<Ts>(args)...}
    {
    }

    CELER_FUNCTION void operator()(ThreadId thread)
    {
#if CELER_DEVICE_COMPILE
        CELER_EXPECT(thread);
        if (!(thread < state_->size()))
        {
            return;
        }
#else
        CELER_EXPECT(thread < state_->size());
#endif
        CoreTrackView const track(*params_, *state_, thread);
        if (!applies_(track.make_sim_view()))
        {
            return;
        }

        return execute_impl_(track);
    }

  private:
    ParamsPtr params_;
    StatePtr state_;
    C applies_;
    detail::TrackExecutorImpl<Ts...> execute_impl_;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class... Ts>
CELER_FUNCTION TrackExecutor(CoreParamsPtr<MemSpace::native>,
                             CoreStatePtr<MemSpace::native>,
                             Ts...)
    ->TrackExecutor<Ts...>;

template<class C, class... Ts>
CELER_FUNCTION ConditionalTrackExecutor(CoreParamsPtr<MemSpace::native>,
                                        CoreStatePtr<MemSpace::native>,
                                        C,
                                        Ts...)
    ->ConditionalTrackExecutor<C, Ts...>;

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Return a track executor that only applies to alive tracks.
 */
template<class... Ts>
inline CELER_FUNCTION decltype(auto)
make_active_track_executor(CoreParamsPtr<MemSpace::native> params,
                           CoreStatePtr<MemSpace::native> const& state,
                           Ts&&... args)
{
    return ConditionalTrackExecutor{
        params, state, detail::applies_active, celeritas::forward<Ts>(args)...};
}

//---------------------------------------------------------------------------//
/*!
 * Return a track executor that only applies if the action ID matches.
 *
 * \note This should generally only be used for post-step actions and other
 * cases where the IDs *explicitly* are set. Many explicit actions apply to all
 * threads, active or not.
 *
 * \todo Replace action-gen script and simplify BoundaryActionImpl and
 * DiscreteSelectActionImpl.
 */
template<class... Ts>
inline CELER_FUNCTION decltype(auto)
make_action_track_executor(CoreParamsPtr<MemSpace::native> params,
                           CoreStatePtr<MemSpace::native> state,
                           ActionId action,
                           Ts&&... args)
{
    CELER_EXPECT(action);
    return ConditionalTrackExecutor{params,
                                    state,
                                    detail::IsStepActionEqual{action},
                                    celeritas::forward<Ts>(args)...};
}

//---------------------------------------------------------------------------//
/*!
 * Return a track executor that only applies for the given along-step action.
 */
template<class... Ts>
inline CELER_FUNCTION decltype(auto)
make_along_step_track_executor(CoreParamsPtr<MemSpace::native> params,
                               CoreStatePtr<MemSpace::native> state,
                               ActionId action,
                               Ts&&... args)
{
    CELER_EXPECT(action);
    return ConditionalTrackExecutor{params,
                                    state,
                                    detail::IsAlongStepActionEqual{action},
                                    celeritas::forward<Ts>(args)...};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
