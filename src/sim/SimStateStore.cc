//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimStateStore.cc
//---------------------------------------------------------------------------//
#include "SimStateStore.hh"

#include <vector>
#include "base/Array.hh"
#include "SimInterface.hh"
#include "detail/SimStateInit.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with number of parallel tracks.
 */
SimStateStore::SimStateStore(size_type size) : vars_(size)
{
    CELER_EXPECT(size > 0);

    detail::sim_state_init_device(this->device_pointers());
}

//---------------------------------------------------------------------------//
/*!
 * Number of threads stored in the state.
 */
size_type SimStateStore::size() const
{
    return vars_.size();
}

//---------------------------------------------------------------------------//
/*!
 * View to on-device state data.
 */
SimStatePointers SimStateStore::device_pointers()
{
    SimStatePointers result;
    result.vars = vars_.device_pointers();

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
