//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/ArrayUtils.hh
//! \brief Math functions using celeritas::Array
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"

#include "Algorithms.hh"
#include "SoftEqual.hh"
#include "detail/ArrayUtilsImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Perform y <- ax + y
template<class T, size_type N>
inline CELER_FUNCTION void axpy(T a, Array<T, N> const& x, Array<T, N>* y);

//---------------------------------------------------------------------------//
// Calculate product of two vectors
template<class T, size_type N>
[[nodiscard]] inline CELER_FUNCTION T dot_product(Array<T, N> const& x,
                                                  Array<T, N> const& y);

//---------------------------------------------------------------------------//
// Calculate product of two vectors
template<class T>
[[nodiscard]] inline CELER_FUNCTION Array<T, 3>
cross_product(Array<T, 3> const& x, Array<T, 3> const& y);

//---------------------------------------------------------------------------//
// Calculate the Euclidian (2) norm of a vector
template<class T, size_type N>
[[nodiscard]] inline CELER_FUNCTION T norm(Array<T, N> const& vec);

//---------------------------------------------------------------------------//
// Construct a vector with unit magnitude
template<class T, size_type N>
[[nodiscard]] inline CELER_FUNCTION Array<T, N>
make_unit_vector(Array<T, N> const& v);

//---------------------------------------------------------------------------//
// Calculate the Euclidian (2) distance between two points
template<class T, size_type N>
[[nodiscard]] inline CELER_FUNCTION T distance(Array<T, N> const& x,
                                               Array<T, N> const& y);

//---------------------------------------------------------------------------//
// Calculate a cartesian unit vector from spherical coordinates
template<class T>
[[nodiscard]] inline CELER_FUNCTION Array<T, 3>
from_spherical(T costheta, T phi);

//---------------------------------------------------------------------------//
// Rotate the direction 'dir' according to the reference rotation axis 'rot'
template<class T>
[[nodiscard]] inline CELER_FUNCTION Array<T, 3>
rotate(Array<T, 3> const& dir, Array<T, 3> const& rot);

//---------------------------------------------------------------------------//
// Test for being approximately a unit vector
template<class T, size_type N>
inline CELER_FUNCTION bool is_soft_unit_vector(Array<T, N> const& v);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Increment a vector by another vector multiplied by a scalar.
 */
template<class T, size_type N>
CELER_FUNCTION void axpy(T a, Array<T, N> const& x, Array<T, N>* y)
{
    CELER_EXPECT(y);
    for (size_type i = 0; i != N; ++i)
    {
        (*y)[i] = std::fma(a, x[i], (*y)[i]);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Dot product of two vectors.
 */
template<class T, size_type N>
CELER_FUNCTION T dot_product(Array<T, N> const& x, Array<T, N> const& y)
{
    T result{};
    for (size_type i = 0; i != N; ++i)
    {
        result = std::fma(x[i], y[i], result);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Cross product of two space vectors.
 */
template<class T>
CELER_FUNCTION Array<T, 3>
cross_product(Array<T, 3> const& x, Array<T, 3> const& y)
{
    return {x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0]};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the Euclidian (2) norm of a vector.
 */
template<class T, size_type N>
CELER_FUNCTION T norm(Array<T, N> const& v)
{
    return std::sqrt(dot_product(v, v));
}

//---------------------------------------------------------------------------//
/*!
 * Construct a unit vector.
 *
 * Unit vectors have an Euclidian norm magnitude of 1.
 */
template<class T, size_type N>
CELER_FUNCTION Array<T, N> make_unit_vector(Array<T, N> const& v)
{
    Array<T, N> result{v};
    T const scale_factor = 1 / norm(result);
    for (auto& el : result)
    {
        el *= scale_factor;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the Euclidian (2) distance between two points.
 */
template<class T, size_type N>
CELER_FUNCTION T distance(Array<T, N> const& x, Array<T, N> const& y)
{
    T dist_sq = 0;
    for (size_type i = 0; i != N; ++i)
    {
        dist_sq += ipow<2>(y[i] - x[i]);
    }
    return std::sqrt(dist_sq);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate a Cartesian vector from spherical coordinates.
 *
 * Theta is the angle between the Z axis and the outgoing vector, and phi is
 * the angle between the x axis and the projection of the vector onto the x-y
 * plane.
 */
template<class T>
inline CELER_FUNCTION Array<T, 3> from_spherical(T costheta, T phi)
{
    CELER_EXPECT(costheta >= -1 && costheta <= 1);

    T const sintheta = std::sqrt(1 - costheta * costheta);
    return {sintheta * std::cos(phi), sintheta * std::sin(phi), costheta};
}

//---------------------------------------------------------------------------//
/*!
 * Rotate the direction about the given Z-based scatter direction.
 *
 * The equivalent to calling the Shift transport code's \code
    void cartesian_vector_transform(
        double      costheta,
        double      phi,
        Vector_View vector);
 * \endcode
 * is the call
 * \code
    vector = rotate(from_spherical(costheta, phi), vector);
 * \endcode
 *
 * This code effectively decomposes the given rotation vector \c rot into two
 * sequential transform matrices, one with an angle \em theta about the \em y
 * axis and one about \em phi rotating around the \em z axis. These two angles
 * are the spherical coordinate transform of the given \c rot cartesian
 * direction vector.
 *
 * There is some extra code in here to deal with loss of precision when the
 * incident direction is along the \em z axis. As \c rot approaches \em z, the
 * azimuthal angle \em phi must be calculated carefully from both the x and y
 * components of the vector, not independently. If \c rot actually equals \em z
 * then the azimuthal angle is completely indeterminate so we arbitrarily
 * choose \c phi = 0.
 *
 * This function is often used for calculating exiting scattering angles. In
 * that case, \c dir is the exiting angle from the scattering calculation, and
 * \c rot is the original direction of the particle. The direction vectors are
 * defined as
 * \f[
   \Omega =   \sin\theta\cos\phi\mathbf{i}
            + \sin\theta\sin\phi\mathbf{j}
            + \cos\theta\mathbf{k} \,.
 * \f]
 */
template<class T>
inline CELER_FUNCTION Array<T, 3>
rotate(Array<T, 3> const& dir, Array<T, 3> const& rot)
{
    CELER_EXPECT(is_soft_unit_vector(dir));
    CELER_EXPECT(is_soft_unit_vector(rot));

    // Direction enumeration
    enum
    {
        X = 0,
        Y = 1,
        Z = 2
    };

    // Transform direction vector into theta, phi so we can use it as a
    // rotation matrix
    T sintheta = std::sqrt(1 - ipow<2>(rot[Z]));
    T cosphi;
    T sinphi;

    if (sintheta >= detail::RealVecTraits<T>::min_accurate_sintheta())
    {
        // Typical case: far enough from z axis to assume the X and Y
        // components have a hypotenuse of 1 within epsilon tolerance
        T const inv_sintheta = 1 / (sintheta);
        cosphi = rot[X] * inv_sintheta;
        sinphi = rot[Y] * inv_sintheta;
    }
    else if (sintheta > 0)
    {
        // Avoid catastrophic roundoff error by normalizing x/y components
        cosphi = rot[X] / std::sqrt(ipow<2>(rot[X]) + ipow<2>(rot[Y]));
        sinphi = std::sqrt(1 - ipow<2>(cosphi));
    }
    else
    {
        // NaN or 0: choose an arbitrary azimuthal angle for the incident dir
        cosphi = 1;
        sinphi = 0;
    }

    Array<T, 3> result
        = {(rot[Z] * dir[X] + sintheta * dir[Z]) * cosphi - sinphi * dir[Y],
           (rot[Z] * dir[X] + sintheta * dir[Z]) * sinphi + cosphi * dir[Y],
           -sintheta * dir[X] + rot[Z] * dir[Z]};

    // Always normalize to prevent roundoff error from propagating
    return make_unit_vector(result);
}

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
 *
 * Example:
 * \code
    CELER_EXPECT(is_soft_unit_vector(v));
 * \endcode
 */
template<class T, size_type N>
CELER_FUNCTION bool is_soft_unit_vector(Array<T, N> const& v)
{
    constexpr SoftEqual<T> default_soft_eq;
    SoftEqual cmp{3 * default_soft_eq.rel(), 3 * default_soft_eq.abs()};
    return cmp(T{1}, dot_product(v, v));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
