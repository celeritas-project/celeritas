//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/ArraySoftUnit.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"

#include "detail/SoftEqualTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Test for being approximately a unit vector.
 *
 * Consider a unit vector \em v with a small perturbation along a unit vector
 * \em e : \f[
   \vec v + \epsilon \vec e
  \f]
 * The magnitude squared is
 * \f[
  m^2 = (v + \epsilon e) \cdot (v + \epsilon e)
   = v \cdot v + 2 \epsilon v \cdot e +  \epsilon^2 e \cdot e
   = 1 + 2 \epsilon v \cdot e + \epsilon^2
 \f]
 *
 * Since \f[ |v \cdot e|  <= |v||e| = 1 \f] by the triangle inequality,
 * then the magnitude squared of a perturbed unit vector is bounded
 * \f[
  m^2 = 1 \pm 2 \epsilon + \epsilon^2
  \f]
 *
 * Instead of calculating the square of the tolerance we loosely bound with
 * another epsilon.
 */
template<class T = ::celeritas::real_type>
class ArraySoftUnit
{
  public:
    //!@{
    //! \name Type aliases
    using value_type = T;
    //!@}

  public:
    // Construct with explicit tolerance
    CELER_FUNCTION inline ArraySoftUnit(value_type tol);

    // Construct with default tolerance
    CELER_CONSTEXPR_FUNCTION ArraySoftUnit();

    // Calculate whether the array is nearly a unit vector
    template<::celeritas::size_type N>
    CELER_FUNCTION bool operator()(Array<T, N> const& arr) const;

  private:
    value_type tol_;
};

//---------------------------------------------------------------------------//
// TEMPLATE DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class T>
CELER_FUNCTION ArraySoftUnit(T) -> ArraySoftUnit<T>;

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Test for being approximately a unit vector
template<class T, size_type N>
CELER_CONSTEXPR_FUNCTION bool is_soft_unit_vector(Array<T, N> const& v);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with explicit tolereance.
 */
template<class T>
CELER_FUNCTION ArraySoftUnit<T>::ArraySoftUnit(T tol) : tol_{3 * tol}
{
    CELER_EXPECT(tol_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with default tolereance.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION ArraySoftUnit<T>::ArraySoftUnit()
    : tol_{3 * detail::SoftEqualTraits<T>::rel_prec()}
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate whether the array is nearly a unit vector.
 *
 * The calculation below is equivalent to
 * \code
 * return SoftEqual{tol, tol}(1, dot_product(arr, arr));
 * \endcode.
 */
template<class T>
template<::celeritas::size_type N>
CELER_FUNCTION bool ArraySoftUnit<T>::operator()(Array<T, N> const& arr) const
{
    T length_sq{};
    for (size_type i = 0; i != N; ++i)
    {
        length_sq = std::fma(arr[i], arr[i], length_sq);
    }
    return std::fabs(length_sq - 1) < tol_ * std::fmax(1, length_sq);
}

//---------------------------------------------------------------------------//
//! Test with default tolerance for being a unit vector
template<class T, size_type N>
CELER_CONSTEXPR_FUNCTION bool is_soft_unit_vector(Array<T, N> const& v)
{
    return ArraySoftUnit<T>{}(v);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
