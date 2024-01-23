//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/Utils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
struct IsEqual
{
    TrackSlotId value;

    CELER_FUNCTION bool operator()(TrackSlotId x) const { return x == value; }
};

//---------------------------------------------------------------------------//
//! Indicate that a track slot is occupied by a still-alive track
CELER_CONSTEXPR_FUNCTION TrackSlotId occupied()
{
    return TrackSlotId{};
}

//---------------------------------------------------------------------------//
//! Get an initializer index where thread 0 has the last valid element
CELER_FORCEINLINE_FUNCTION size_type index_before(size_type size, ThreadId tid)
{
    CELER_EXPECT(tid.get() + 1 <= size);
    return size - tid.unchecked_get() - 1;
}

//---------------------------------------------------------------------------//
//! Get an initializer index a certain number of threads past the end
CELER_FORCEINLINE_FUNCTION size_type index_after(size_type size, ThreadId tid)
{
    CELER_EXPECT(tid);
    return size + tid.unchecked_get();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
