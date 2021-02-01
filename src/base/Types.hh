//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//! Type definitions for common Celeritas functionality
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include "Array.hh"
#include "OpaqueId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Standard type for container sizes.
using size_type = std::size_t;

//! Equivalent to container size but compatible with CUDA atomics
using ull_int = unsigned long long int;

//! Numerical type for real numbers
using real_type = double;

//! Fixed-size array for R3 calculations
using Real3 = Array<real_type, 3>;

//! Index of the current CUDA thread, with type safety for containers.
using ThreadId = OpaqueId<struct Thread>;

//---------------------------------------------------------------------------//
// ENUMERATIONS
//---------------------------------------------------------------------------//
//! Whether an interpolation is linear or logarithmic (template parameter)
enum class Interp
{
    linear,
    log
};

//! Memory location of data, or a pointer
enum class MemSpace
{
    host,
    device,
#ifdef __CUDA_ARCH__
    native = device,
#else
    native = host,
#endif
};

//! Non-convertible type for raw data modeled after std::byte (C++17)
enum class Byte : unsigned char
{
};

//---------------------------------------------------------------------------//
} // namespace celeritas
