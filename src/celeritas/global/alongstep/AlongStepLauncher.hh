//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/global/CoreTrackData.hh"

#include "detail/AlongStepLauncherImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Return a functor for executing an along-step kernel.
 *
 * This function is used to implement along-step actions for individual tracks.
 */
template<class M, class P, class E, class F>
CELER_FUNCTION detail::AlongStepLauncherImpl<M, P, E, F>
make_along_step_launcher(CoreRef<MemSpace::native> const& core_data,
                         M&&                              msc_data,
                         P&&                              propagator_data,
                         E&&                              eloss_data,
                         F&&                              call_with_track)
{
    return {core_data,
            ::celeritas::forward<M>(msc_data),
            ::celeritas::forward<P>(propagator_data),
            ::celeritas::forward<E>(eloss_data),
            ::celeritas::forward<F>(call_with_track)};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
