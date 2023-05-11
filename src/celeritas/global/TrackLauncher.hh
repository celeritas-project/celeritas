//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/TrackLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/track/SimTrackView.hh"

#include "CoreTrackData.hh"
#include "detail/TrackLauncherImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Function-like class to launch a CoreTrackView-dependent function.
 *
 * This class should be used primarily by generated kernel functions:
 *
 * \code
__global__ void foo_kernel(CoreParamsRef const params,
                           CoreParamsRef const state,
                           OtherData const other)
{
    TrackLauncher launch{params, state, apply_to_track, OtherView{other}};
    launch(KernelParamCalculator::thread_id());
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
class TrackLauncher
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<CoreParamsData>;
    using StateRef = NativeRef<CoreStateData>;
    //!@}

  public:
    CELER_FUNCTION
    TrackLauncher(ParamsRef const& params, StateRef const& state, Ts... args)
        : params_{params}
        , state_{state}
        , launch_impl_{celeritas::forward<Ts>(args)...}
    {
    }

    CELER_FUNCTION void operator()(ThreadId thread) const
    {
        CELER_EXPECT(thread);
#if CELER_DEVICE_COMPILE
        if (!(thread < state_.size()))
        {
            return;
        }
#else
        CELER_EXPECT(thread < state_.size());
#endif
        CoreTrackView const track(params_, state_, thread);

        return launch_impl_(track);
    }

  private:
    ParamsRef const& params_;
    StateRef const& state_;
    detail::TrackLauncherImpl<Ts...> launch_impl_;
};

//---------------------------------------------------------------------------//
/*!
 * Launch the track only when a certain condition applies.
 *
 * The condition C must have the signature \code
 * <bool(SimTrackView const&)>
  \endcode
 *
 * see \c make_active_track_launcher for an example where this is used to apply
 * only to active (or killed) tracks.
 */
template<class C, class... Ts>
class ConditionalTrackLauncher
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<CoreParamsData>;
    using StateRef = NativeRef<CoreStateData>;
    //!@}

  public:
    CELER_FUNCTION
    ConditionalTrackLauncher(ParamsRef const& params,
                             StateRef const& state,
                             C applies,
                             Ts... args)
        : params_{params}
        , state_{state}
        , applies_{celeritas::forward<C>(applies)}
        , launch_impl_{celeritas::forward<Ts>(args)...}
    {
    }

    CELER_FUNCTION void operator()(ThreadId thread)
    {
        CELER_EXPECT(thread);
#if CELER_DEVICE_COMPILE
        if (!(thread < state_.size()))
        {
            return;
        }
#else
        CELER_EXPECT(thread < state_.size());
#endif
        CoreTrackView const track(params_, state_, thread);
        if (!applies_(track.make_sim_view()))
        {
            return;
        }

        return launch_impl_(track);
    }

  private:
    ParamsRef const& params_;
    StateRef const& state_;
    C applies_;
    detail::TrackLauncherImpl<Ts...> launch_impl_;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class... Ts>
CELER_FUNCTION TrackLauncher(NativeCRef<CoreParamsData> const&,
                             NativeRef<CoreStateData> const&,
                             Ts...)
    ->TrackLauncher<Ts...>;

template<class C, class... Ts>
CELER_FUNCTION ConditionalTrackLauncher(NativeCRef<CoreParamsData> const&,
                                        NativeRef<CoreStateData> const&,
                                        C,
                                        Ts...)
    ->ConditionalTrackLauncher<C, Ts...>;

//---------------------------------------------------------------------------//
/*!
 * Return a track launcher that only applies to alive tracks.
 */
template<class... Ts>
inline CELER_FUNCTION decltype(auto)
make_active_track_launcher(NativeCRef<CoreParamsData> const& params,
                           NativeRef<CoreStateData> const& state,
                           Ts&&... args)
{
    return ConditionalTrackLauncher{
        params, state, detail::applies_active, celeritas::forward<Ts>(args)...};
}

//---------------------------------------------------------------------------//
/*!
 * Return a track launcher that only applies if the action ID matches.
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
make_action_track_launcher(NativeCRef<CoreParamsData> const& params,
                           NativeRef<CoreStateData> const& state,
                           ActionId action,
                           Ts&&... args)
{
    CELER_EXPECT(action);
    return ConditionalTrackLauncher{params,
                                    state,
                                    detail::IsStepActionEqual{action},
                                    celeritas::forward<Ts>(args)...};
}

//---------------------------------------------------------------------------//
/*!
 * Return a track launcher that only applies for the given along-step action.
 */
template<class... Ts>
inline CELER_FUNCTION decltype(auto)
make_along_step_track_launcher(NativeCRef<CoreParamsData> const& params,
                               NativeRef<CoreStateData> const& state,
                               ActionId action,
                               Ts&&... args)
{
    CELER_EXPECT(action);
    return ConditionalTrackLauncher{params,
                                    state,
                                    detail::IsAlongStepActionEqual{action},
                                    celeritas::forward<Ts>(args)...};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
