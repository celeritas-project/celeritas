//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/Utils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/math/NumericLimits.hh"
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
//! Get a track slot ID a certain number of threads from the end
CELER_FORCEINLINE_FUNCTION size_type from_back(size_type size, ThreadId tid)
{
    CELER_EXPECT(tid.get() + 1 <= size);
    return size - tid.get() - 1;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
