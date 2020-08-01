//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ArrayUtils.i.hh
//---------------------------------------------------------------------------//

#include <cmath>
#include "base/Assert.hh"
#include "base/SoftEqual.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Increment a vector by another vector multiplied by a scalar.
 */
template<typename T, std::size_t N>
CELER_FUNCTION void axpy(T a, const array<T, N>& x, array<T, N>* y)
{
    for (std::size_t i = 0; i != N; ++i)
    {
        (*y)[i] = a * x[i] + (*y)[i];
    }
}

//---------------------------------------------------------------------------//
/*!
 * Increment a vector by another vector multiplied by a scalar.
 */
template<typename T, std::size_t N>
CELER_FUNCTION T dot_product(const array<T, N>& x, const array<T, N>& y)
{
    T result{};
    for (std::size_t i = 0; i != N; ++i)
    {
        result += x[i] * y[i];
    }
    return result;
}

//---------------------------------------------------------------------------//
// Calculate the Euclidian (2) norm of a vector
template<typename T, std::size_t N>
CELER_FUNCTION T norm(const array<T, N>& v)
{
    return std::sqrt(dot_product(v, v));
}

//---------------------------------------------------------------------------//
// Divide the given vector by its Euclidian norm
CELER_FUNCTION void normalize_direction(array<real_type, 3>* direction)
{
    real_type scale_factor = 1 / norm(*direction);
    (*direction)[0] *= scale_factor;
    (*direction)[1] *= scale_factor;
    (*direction)[2] *= scale_factor;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate a Cartesian vector from spherical coordinates.
 *
 * Theta is the angle between the Z axis and the outgoing vector, and phi is
 * the angle between the x axis and the projection of the vector onto the x-y
 * plane.
 */
inline CELER_FUNCTION Real3 from_spherical(real_type costheta, real_type phi)
{
    REQUIRE(costheta >= -1 && costheta <= 1);

    const real_type sintheta = std::sqrt(1 - costheta * costheta);
    return {sintheta * std::cos(phi), sintheta * std::sin(phi), costheta};
}

//---------------------------------------------------------------------------//
/*!
 * Rotate the direction about the given Z-based scatter direction.
 *
 * The equivalent to Shift's \code
 * void cartesian_vector_transform(
    double      costheta,
    double      phi,
    Vector_View vector)
 * \endcode
 * is the call
 * \code
   rotate(from_spherical(costheta, phi), &vector);
 * \endcode
 *
 * Note that rotation has loss of precision when the original direction is
 * close to the z axis. If this is a problem we should consider using
 * quaternions or another method robust for rotations.
 *
 * Also note that the input "scatter" vector is const *value*, to avoid
 * aliasing.
 */
inline CELER_FUNCTION void rotate(const Real3 scat, Real3* dir)
{
    REQUIRE(is_soft_unit_vector(scat, SoftEqual<real_type>(1e-6)));
    REQUIRE(is_soft_unit_vector(*dir, SoftEqual<real_type>(1e-6)));

    // Direction enumeration
    enum
    {
        X = 0,
        Y = 1,
        Z = 2
    };

    // Copy the original directions to avoid aliasing
    const Real3 orig = *dir;

    // Calculate how close the original axis is to the pole of the polar
    // direction
    real_type alpha = 1 - orig[Z] * orig[Z];
    if (alpha >= detail::SoftEqualTraits<real_type>::abs_thresh())
    {
        alpha = std::sqrt(alpha);

        (*dir)[X] = orig[X] * scat[Z]
                    + (1 / alpha)
                          * (orig[X] * orig[Z] * scat[X] - orig[Y] * scat[Y]);
        (*dir)[Y] = orig[Y] * scat[Z]
                    + (1 / alpha)
                          * (orig[Y] * orig[Z] * scat[X] + orig[X] * scat[Y]);
        (*dir)[Z] = orig[Z] * scat[Z] - alpha * scat[X];
    }
    else
    {
        // Degenerate case: snap to polar axis
        (*dir)[X] = scat[X];
        (*dir)[Y] = scat[Y];
        (*dir)[Z] = (orig[Z] < 0 ? -1 : 1) * scat[Z];
    }

    normalize_direction(dir);
}

//---------------------------------------------------------------------------//
/*!
 * Test for being approximately a unit vector.
 *
 * Example:
 * \code
  REQUIRE(is_soft_unit_vector(v, SoftEqual(1e-12)))
  \endcode;
 */
template<typename T, std::size_t N, class SoftEq>
CELER_FUNCTION bool is_soft_unit_vector(const array<T, N>& v, SoftEq cmp)
{
    return cmp(T(1), dot_product(v, v));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
