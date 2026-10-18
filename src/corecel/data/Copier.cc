//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Copier.cc
//---------------------------------------------------------------------------//
#include "Copier.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform a memcpy on the data.
 */
void copy_bytes(MemSpace dstmem,
                void* dst,
                MemSpace srcmem,
                void const* src,
                std::size_t count)
{
    detail::copy_bytes_impl(dstmem, dst, srcmem, src, count);
}

//---------------------------------------------------------------------------//
/*!
 * Perform an asynchronous memcpy on the data.
 */
void copy_bytes(MemSpace dstmem,
                void* dst,
                MemSpace srcmem,
                void const* src,
                std::size_t count,
                StreamId stream)
{
    detail::copy_bytes_impl(dstmem, dst, srcmem, src, count, stream);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
