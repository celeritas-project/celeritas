//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Utils.hh
//---------------------------------------------------------------------------//
#ifndef base_detail_Utils_hh
#define base_detail_Utils_hh

#include "../Span.hh"
#include "../Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T>
inline void device_memset_zero(span<T> data);

//---------------------------------------------------------------------------//
void device_memset(void* data, int fill_value, size_type count);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "Utils.i.hh"

#endif // base_detail_Utils_hh
