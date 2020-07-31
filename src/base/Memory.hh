//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Memory.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Span.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class T>
inline void device_memset_zero(span<T> data);

//---------------------------------------------------------------------------//
void device_memset(void* data, int fill_value, size_type count);

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "Memory.i.hh"

//---------------------------------------------------------------------------//
