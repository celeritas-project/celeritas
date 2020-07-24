//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ArrayUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Array.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Perform y <- ax + y
template<typename T, std::size_t N>
inline CELER_FUNCTION void axpy(T a, const array<T, N>& x, array<T, N>* y);

//---------------------------------------------------------------------------//
// Calculate product of two vectors
template<typename T, std::size_t N>
inline CELER_FUNCTION T dot_product(const array<T, N>& x, const array<T, N>& y);

//---------------------------------------------------------------------------//
// Calculate the Euclidian (2) norm of a vector
template<typename T, std::size_t N>
inline CELER_FUNCTION T norm(const array<T, N>& vec);

//---------------------------------------------------------------------------//
// Divide the given vector by its Euclidian norm
inline CELER_FUNCTION void normalize_direction(array<real_type, 3>* direction);

//---------------------------------------------------------------------------//
// Rotate the direction about the given polar coordinates
inline CELER_FUNCTION void
rotate_polar(real_type costheta, real_type phi, array<real_type, 3>* direction);

//---------------------------------------------------------------------------//
// Test for being approximately a unit vector
template<typename T, std::size_t N, class SoftEq>
inline CELER_FUNCTION bool
is_soft_unit_vector(const array<T, N>& v, SoftEq cmp);

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "ArrayUtils.i.hh"
