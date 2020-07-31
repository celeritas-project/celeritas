//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Memory.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Call memset on device data.
 */
template<class T>
void device_memset_zero(span<T> device_pointers)
{
    device_memset(
        device_pointers.data(), 0, device_pointers.size() * sizeof(T));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
