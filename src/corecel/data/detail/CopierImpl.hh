//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/CopierImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>

#include "corecel/Types.hh"
#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class Stream;

namespace detail
{
//---------------------------------------------------------------------------//
// NOTE: these are not named copy_bytes so that argument-dependent lookup of
// celeritas::copy_bytes with detail types is never ambiguous

// Copy bytes between two memory spaces
void copy_bytes_impl(MemSpace dstmem,
                     void* dst,
                     MemSpace srcmem,
                     void const* src,
                     std::size_t count);

// Asynchronously copy bytes between two memory spaces on a stream ID
void copy_bytes_impl(MemSpace dstmem,
                     void* dst,
                     MemSpace srcmem,
                     void const* src,
                     std::size_t count,
                     StreamId stream);

// Asynchronously copy bytes between two memory spaces on a stream
void copy_bytes_impl(MemSpace dstmem,
                     void* dst,
                     MemSpace srcmem,
                     void const* src,
                     std::size_t count,
                     Stream const& stream);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
