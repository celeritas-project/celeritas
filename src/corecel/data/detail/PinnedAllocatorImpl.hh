//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/PinnedAllocatorImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdlib>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Allocate pinned memory
void* malloc_pinned(std::size_t n, std::size_t sizeof_t);

// Free pinned memory
void free_pinned(void* ptr) noexcept;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
