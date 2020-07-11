//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Utils.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Call memset on device data.
 */
template<class T>
void device_memset_zero(span<T> device_view)
{
    device_memset(device_view.data(), 0, device_view.size() * sizeof(T));
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
