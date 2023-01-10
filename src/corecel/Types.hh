//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/Types.hh
//! \brief Type definitions for common Celeritas functionality
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>

#include "Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
#if CELER_USE_DEVICE
//! Standard type for container sizes, optimized for GPU use.
using size_type = unsigned int;
#else
using size_type = std::size_t;
#endif

//! Numerical type for real numbers
using real_type = double;

//! Equivalent to std::size_t but compatible with CUDA atomics
using ull_int = unsigned long long int;

//---------------------------------------------------------------------------//
// ENUMERATIONS
//---------------------------------------------------------------------------//
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
#ifdef CELER_DEVICE_SOURCE
    native = device,  // Included by a CUDA/HIP file
#else
    native = host,
#endif
};

//! Data ownership flag
enum class Ownership
{
    value,  //!< Ownership of the data, only on host
    reference,  //!< Mutable reference to the data
    const_reference,  //!< Immutable reference to the data
};

#if !defined(SWIG) || SWIG_VERSION > 0x050000
//---------------------------------------------------------------------------//
//!@{
//! \name Convenience typedefs for params and states.

//! Managed host memory
template<template<Ownership, MemSpace> class P>
using HostVal = P<Ownership::value, MemSpace::host>;
//! Immutable reference to host memory
template<template<Ownership, MemSpace> class P>
using HostCRef = P<Ownership::const_reference, MemSpace::host>;
//! Mutable reference to host memory
template<template<Ownership, MemSpace> class S>
using HostRef = S<Ownership::reference, MemSpace::host>;

//! Immutable reference to device memory
template<template<Ownership, MemSpace> class P>
using DeviceCRef = P<Ownership::const_reference, MemSpace::device>;
//! Mutable reference to device memory
template<template<Ownership, MemSpace> class S>
using DeviceRef = S<Ownership::reference, MemSpace::device>;

//! Immutable reference to native memory
template<template<Ownership, MemSpace> class P>
using NativeCRef = P<Ownership::const_reference, MemSpace::native>;
//! Mutable reference to native memory
template<template<Ownership, MemSpace> class S>
using NativeRef = S<Ownership::reference, MemSpace::native>;

//!@}
#endif
//---------------------------------------------------------------------------//
}  // namespace celeritas
