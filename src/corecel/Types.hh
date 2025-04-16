//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/*!
 * \file corecel/Types.hh
 * \brief Type definitions for common Celeritas functionality.
 *
 * This file includes types and properties particular to the build
 * configuration.
 */
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <string>

#include "corecel/Config.hh"

#include "Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
#if CELER_USE_DEVICE || defined(__DOXYGEN__)
//! Standard type for container sizes, optimized for GPU use
using size_type = unsigned int;
#else
using size_type = std::size_t;
#endif

#if CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE
//! Numerical type for real numbers
using real_type = double;
#elif CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT
using real_type = float;
#else
using real_type = void;
#endif

//! Equivalent to std::size_t but compatible with CUDA atomics
using ull_int = unsigned long long int;

//---------------------------------------------------------------------------//
// ENUMERATIONS
//---------------------------------------------------------------------------//
//! Memory location of data
enum class MemSpace
{
    host,  //!< CPU memory
    device,  //!< GPU memory
    mapped,  //!< Unified virtual address space (both host and device)
    size_,
#if defined(CELER_DEVICE_SOURCE) || defined(__DOXYGEN__)
    native = device,  //!< When compiling CUDA files, \c device else \c host
#else
    native = host,
#endif
};

//! Data ownership flag
enum class Ownership
{
    value,  //!< The collection \em owns the data
    reference,  //!< Mutable reference to data
    const_reference,  //!< Immutable reference to data
};

//---------------------------------------------------------------------------//
//! Unit system used by Celeritas
enum class UnitSystem
{
    none,  //!< Invalid unit system
    cgs,  //!< Gaussian CGS
    si,  //!< International System
    clhep,  //!< Geant4 native
    size_,
    native = CELERITAS_UNITS,  //!< Compile time selected system
};

//---------------------------------------------------------------------------//
//! Interpolation type
enum class Interp
{
    linear,
    log,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Which of two bounding points, faces, energies, etc.
 *
 * Here, lo/hi can correspond to left/right, back/front, bottom/top. It's used
 * for the two points in a bounding box.
 */
enum class Bound : unsigned char
{
    lo,
    hi,
    size_
};

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

template<class T, MemSpace M>
class ObserverPtr;

//! Pointer to same-memory *const* collection group
template<template<Ownership, MemSpace> class P, MemSpace M>
using CRefPtr = ObserverPtr<P<Ownership::const_reference, M> const, M>;
//! Pointer to same-memory *mutable* collection group
template<template<Ownership, MemSpace> class S, MemSpace M>
using RefPtr = ObserverPtr<S<Ownership::reference, M>, M>;

//!@}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS (HOST)
//---------------------------------------------------------------------------//

// Get a string corresponding to a memory space
char const* to_cstring(MemSpace m);

// Get a string corresponding to a unit system
char const* to_cstring(UnitSystem);

// Get a unit system corresponding to a string
UnitSystem to_unit_system(std::string const& s);

//! Convert Bound enum value to int
CELER_CONSTEXPR_FUNCTION int to_int(Bound b)
{
    return static_cast<int>(b);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
