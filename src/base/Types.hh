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

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Standard type for container sizes.
using size_type = std::size_t;

//! Equivalent to container size but compatible with CUDA atomics
using ull_int = unsigned long long int;

//! Numerical type for real numbers
using real_type = double;


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
