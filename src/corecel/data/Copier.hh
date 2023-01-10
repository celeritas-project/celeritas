//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Copier.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Copy spans of data.
 *
 * Example of copying data from host to device:
 * \code
    Copier<int, MemSpace::host> copy_from_host_to{host_ints};
    copy_from_host_to(MemSpace::device, device_ints);
 * \endcode
 */
template<class T, MemSpace M>
struct Copier
{
    static_assert(std::is_trivially_copyable<T>::value,
                  "Data is not trivially copyable");

    Span<const T> src;

    inline void operator()(MemSpace dstmem, Span<T> dst) const;
};

//---------------------------------------------------------------------------//
// Copy bytes between two memory spaces
void copy_bytes(MemSpace    dstmem,
                void*       dst,
                MemSpace    srcmem,
                const void* src,
                std::size_t count);

//---------------------------------------------------------------------------//
/*!
 * Copy data to the given destination and memory space.
 */
template<class T, MemSpace M>
void Copier<T, M>::operator()(MemSpace dstmem, Span<T> dst) const
{
    CELER_EXPECT(src.size() == dst.size());
    copy_bytes(dstmem, dst.data(), M, src.data(), src.size() * sizeof(T));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
