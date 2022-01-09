//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Utils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "base/NumericLimits.hh"
#include "base/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
struct IsEqual
{
    size_type value;

    CELER_FUNCTION bool operator()(size_type x) const { return x == value; }
};

//---------------------------------------------------------------------------//
//! Invalid index flag
CELER_CONSTEXPR_FUNCTION size_type flag_id()
{
    return numeric_limits<size_type>::max();
}

//---------------------------------------------------------------------------//
//! Get the thread ID of the last element
CELER_FORCEINLINE_FUNCTION ThreadId from_back(size_type size, ThreadId tid)
{
    CELER_EXPECT(tid.get() + 1 <= size);
    return ThreadId{size - tid.get() - 1};
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
