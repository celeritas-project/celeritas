//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/SpanUtils.hh
//! \brief Math functions using celeritas::Span
//! \sa SpanUtils.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>

#include "corecel/cont/Span.hh"

#include "ArrayUtils.hh"

#include "detail/SpanUtilsImpl.hh"

namespace celeritas
{
//! \cond (CELERITAS_DOC_DEV)
//---------------------------------------------------------------------------//
/*!
 * Copy all elements from src to dst.
 *
 * \pre src and dst must not overlap.
 *
 * \internal The current implementation does not take advantage of the lack of
 * overlap.
 */
template<class T, std::size_t N>
inline CELER_FUNCTION void copy(Span<T const, N> src, Span<T, N> dst)
{
    for (std::size_t i = 0; i != N; ++i)
        dst[i] = src[i];
}

//---------------------------------------------------------------------------//
/*!
 * Fill all elements of dst with a constant value.
 */
template<class T, std::size_t N>
inline CELER_FUNCTION void fill(T value, Span<T, N> dst)
{
    for (std::size_t i = 0; i != N; ++i)
        dst[i] = value;
}

//---------------------------------------------------------------------------//
/*!
 * Increment a vector span by another vector span multiplied by a scalar.
 *
 * \f[
 * y \gets \alpha x + y
 * \f]
 *
 * \pre x and y must not overlap.
 */
template<class T, std::size_t N>
inline CELER_FUNCTION void axpy(T a, Span<T const, N> x, Span<T, N> y)
{
    Array<T, N> xa = detail::load_array(x);
    Array<T, N> ya = detail::load_array(Span<T const, N>{y.data(), N});
    axpy(a, xa, &ya);
    detail::store_array(ya, y);
}

//---------------------------------------------------------------------------//
/*!
 * Dot product of two vector spans.
 */
template<class T, std::size_t N>
[[nodiscard]] inline CELER_FUNCTION T dot_product(Span<T const, N> x,
                                                  Span<T const, N> y)
{
    return dot_product(detail::load_array(x), detail::load_array(y));
}

//---------------------------------------------------------------------------//
/*!
 * Cross product of two space vector spans, written into dst.
 *
 * \pre x, y, and dst must not overlap.
 */
template<class T>
inline CELER_FUNCTION void
cross_product(Span<T const, 3> x, Span<T const, 3> y, Span<T, 3> dst)
{
    detail::store_array(
        cross_product(detail::load_array(x), detail::load_array(y)), dst);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the Euclidean (2) norm of a vector span.
 */
template<class T, std::size_t N>
[[nodiscard]] inline CELER_FUNCTION T norm(Span<T const, N> v)
{
    return norm(detail::load_array(v));
}

//---------------------------------------------------------------------------//
/*!
 * Write a unit-magnitude version of v into dst.
 *
 * \pre v and dst must not overlap.
 */
template<class T, std::size_t N>
inline CELER_FUNCTION void make_unit_vector(Span<T const, N> v, Span<T, N> dst)
{
    detail::store_array(make_unit_vector(detail::load_array(v)), dst);
}

//---------------------------------------------------------------------------//
/*!
 * Write the component of x orthogonal to the unit vector y into dst.
 *
 * In this implementation, y must be normalized, and the result is not
 * normalized.
 *
 * \f[
   \mathbf{x}' \gets \mathbf{x} - (\mathbf{x} \cdot \mathbf{y}) \mathbf{y}
   \, , \quad \|\mathbf{y}\| = 1
 * \f]
 *
 * \pre x, y, and dst must not overlap.
 */
template<class T, std::size_t N>
inline CELER_FUNCTION void
make_orthogonal(Span<T const, N> x, Span<T const, N> y, Span<T, N> dst)
{
    detail::store_array(
        make_orthogonal(detail::load_array(x), detail::load_array(y)), dst);
}

//---------------------------------------------------------------------------//
/*!
 * Check whether two unit vector spans are approximately orthogonal.
 */
template<class T, std::size_t N>
inline CELER_FUNCTION bool
is_soft_orthogonal(Span<T const, N> x, Span<T const, N> y)
{
    return is_soft_orthogonal(detail::load_array(x), detail::load_array(y));
}

//---------------------------------------------------------------------------//
/*!
 * Check whether two unit vector spans are approximately collinear.
 *
 * \pre Spans must represent normalized vectors.
 */
template<class T, std::size_t N>
inline CELER_FUNCTION bool
is_soft_collinear(Span<T const, N> x, Span<T const, N> y)
{
    return is_soft_collinear(detail::load_array(x), detail::load_array(y));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the Euclidean (2) distance between two point spans.
 */
template<class T, std::size_t N>
[[nodiscard]] inline CELER_FUNCTION T distance(Span<T const, N> x,
                                               Span<T const, N> y)
{
    return distance(detail::load_array(x), detail::load_array(y));
}

//---------------------------------------------------------------------------//
/*!
 * Write a Cartesian unit vector from spherical coordinates into dst.
 *
 * Theta is the angle between the z axis and the outgoing vector, and phi is
 * the angle between the x axis and the projection onto the x-y plane.
 */
template<class T>
inline CELER_FUNCTION void from_spherical(T costheta, T phi, Span<T, 3> dst)
{
    detail::store_array(from_spherical(costheta, phi), dst);
}

//---------------------------------------------------------------------------//
/*!
 * Rotate a direction span by the rotation defined by rot, writing to dst.
 *
 * \pre dir, rot, and dst must not overlap.
 * \sa celeritas::rotate in ArrayUtils.hh for the full description.
 */
template<class T>
inline CELER_FUNCTION void
rotate(Span<T const, 3> dir, Span<T const, 3> rot, Span<T, 3> dst)
{
    detail::store_array(
        rotate(detail::load_array(dir), detail::load_array(rot)), dst);
}

//---------------------------------------------------------------------------//
//! \endcond

}  // namespace celeritas
