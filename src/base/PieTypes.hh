//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PieTypes.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
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

/*!
 * Element indexing type for Pie access.
 *
 * The size type is plain "unsigned int" (32-bit in CUDA) rather than
 * \c celeritas::size_type (64-bit) because CUDA currently uses native 32-bit
 * pointer arithmetic. In general this should be the same type as the default
 * OpaqueId::value_type. It's possible that in large problems 4 billion
 * elements won't be enough (for e.g. cross sections), but in that case the
 * PieBuilder will throw an assertion during construction.
 */
using pie_size_type = unsigned int;

//---------------------------------------------------------------------------//
} // namespace celeritas
