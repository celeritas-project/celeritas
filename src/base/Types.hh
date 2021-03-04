//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//! Type definitions for common Celeritas functionality
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Standard type for container sizes, optimized for GPU use.
using size_type = unsigned int;

//! Numerical type for real numbers
using real_type = double;

//! Equivalent to std::size_t but compatible with CUDA atomics
using ull_int = unsigned long long int;

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
//! Memory location of data
enum class MemSpace
{
    host,
    device,
#ifdef __CUDACC__
    native = device, // Included by a CUDA file
#else
    native = host,
#endif
};

//! Data ownership flag
enum class Ownership
{
    value,           //!< Ownership of the data, only on host
    reference,       //!< Mutable reference to the data
    const_reference, //!< Immutable reference to the data
};

//---------------------------------------------------------------------------//
} // namespace celeritas
