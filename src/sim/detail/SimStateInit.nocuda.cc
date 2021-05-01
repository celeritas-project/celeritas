//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimStateInit.nocuda.cc
//---------------------------------------------------------------------------//
#include "SimStateInit.hh"

#include "base/Assert.hh"
#include "../SimTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Initialize the sim states on device.
 */
void sim_state_init(
    const SimStateData<Ownership::reference, MemSpace::device>& data)
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the sim states on host.
 */
void sim_state_init(
    const SimStateData<Ownership::reference, MemSpace::host>& data)
{
    for (auto id : range(ThreadId{data.size()}))
        data.state[id] = SimTrackView::Initializer_t{};
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
