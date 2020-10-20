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
// Calculate product of two vectors
template<typename T>
inline CELER_FUNCTION array<T, 3>
                      cross_product(const array<T, 3>& x, const array<T, 3>& y);

//---------------------------------------------------------------------------//
// Calculate the Euclidian (2) norm of a vector
template<typename T, std::size_t N>
inline CELER_FUNCTION T norm(const array<T, N>& vec);

//---------------------------------------------------------------------------//
// Divide the given vector by its Euclidian norm
inline CELER_FUNCTION void normalize_direction(Real3* direction);

//---------------------------------------------------------------------------//
// Calculate a cartesian unit vector from spherical coordinates
inline CELER_FUNCTION Real3 from_spherical(real_type costheta, real_type phi);

//---------------------------------------------------------------------------//
// Rotate the direction 'dir' according to the reference rotation axis 'rot'
inline CELER_FUNCTION Real3 rotate(const Real3& dir, const Real3& rot);

//---------------------------------------------------------------------------//
// Test for being approximately a unit vector
template<typename T, std::size_t N, class SoftEq>
inline CELER_FUNCTION bool
is_soft_unit_vector(const array<T, N>& v, SoftEq cmp);

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "ArrayUtils.i.hh"
