//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include "Array.hh"
#include "OpaqueId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
using size_type    = std::size_t;
using ull_int      = unsigned long long int; //!< Compatible with CUDA atomics
using real_type    = double;
using RealPointer3 = Array<real_type*, 3>;
using Real3        = Array<real_type, 3>;

//! Index of the current CUDA thread, with type safety for containers.
using ThreadId = OpaqueId<struct Thread, unsigned int>;

//---------------------------------------------------------------------------//
// ENUMERATIONS
//---------------------------------------------------------------------------//
//! Whether an interpolation is linear or logarithmic (template parameter)
enum class Interp
{
    linear,
    log
};

//! Non-convertible type for raw data modeled after std::byte (C++17)
enum class Byte : unsigned char
{
};

//---------------------------------------------------------------------------//
} // namespace celeritas
