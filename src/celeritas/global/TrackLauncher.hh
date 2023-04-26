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
__global__ void foo_kernel(CoreDeviceRef const data, OtherData const other)
{
    TrackLauncher launch{data, apply_to_track, other};
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
    OtherData const& other)
{
    // ...
}
   \endcode
 * It will call the function with *all* thread slots.
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
    TrackLauncher(ParamsRef const& params, StateRef const& state, Ts&&... args)
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
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class... Ts>
TrackLauncher(NativeCRef<CoreParamsData> const&,
              NativeRef<CoreStateData> const&,
              Ts...) -> TrackLauncher<Ts...>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
