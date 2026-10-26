//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AsyncCompactedCopy.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Item that is valid if nonzero
struct CompactTestItem
{
    int value{0};

    explicit CELER_FUNCTION operator bool() const { return value != 0; }
};

//---------------------------------------------------------------------------//
using VecVecInt = std::vector<std::vector<int>>;

// Compact and copy batches of equally sized items as a stepping loop would
VecVecInt compacted_copy_test(VecVecInt const& batches);

// Destroy a copier while a compaction and a copy are pending
void destroy_pending_test();

#if !CELER_USE_DEVICE
inline VecVecInt compacted_copy_test(VecVecInt const&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline void destroy_pending_test()
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
