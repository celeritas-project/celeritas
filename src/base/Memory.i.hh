//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Memory.i.hh
//---------------------------------------------------------------------------//

#include <type_traits>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Call memset on device data.
 */
template<class T>
void device_memset_zero(Span<T> device_pointers)
{
    static_assert(std::is_trivially_copyable<T>::value,
                  "Only trivially copyable types may be memset");
    device_memset(
        device_pointers.data(), 0, device_pointers.size() * sizeof(T));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
