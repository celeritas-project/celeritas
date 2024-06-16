//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/UserStateVec.cc
//---------------------------------------------------------------------------//
#include "UserStateVec.hh"

#include "corecel/cont/Range.hh"

#include "UserParamsRegistry.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Create from params on a device/host stream.
 */
UserStateVec::UserStateVec(UserParamsRegistry const& registry,
                           MemSpace m,
                           StreamId sid,
                           size_type size)
{
    CELER_EXPECT(m == MemSpace::host || m == MemSpace::device);
    CELER_EXPECT(sid);
    CELER_EXPECT(size > 0);

    states_.reserve(registry.size());

    for (auto uid : range(UserId{registry.size()}))
    {
        states_.emplace_back(registry.at(uid)->create_state(m, sid, size));
        CELER_ENSURE(states_.back());
    }
    CELER_ENSURE(this->size() == registry.size());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
