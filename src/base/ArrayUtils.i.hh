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
 * Rotate the direction about the given Z-based polar coordinates.
 *
 * Note that rotation has loss of precision near the z axis. If this is a
 * problem we should consider using quaternions or another method robust for
 * rotations.
 */
inline CELER_FUNCTION void
rotate_polar(real_type costheta, real_type phi, array<real_type, 3>* dir)
{
    REQUIRE(is_soft_unit_vector(*dir, SoftEqual<real_type>(1e-6)));
    REQUIRE(costheta >= -1 && costheta <= 1);

    // Direction enumeration
    enum
    {
        X = 0,
        Y = 1,
        Z = 2
    };

    const double sintheta = std::sqrt(1.0 - costheta * costheta);
    const double stcosphi = sintheta * std::cos(phi);
    const double stsinphi = sintheta * std::sin(phi);

    // Copy the original direction
    const array<real_type, 3> orig = *dir;
    // Calculate how close the original axis is to the pole of the polar
    // direction
    const double alpha = std::sqrt(1 - orig[Z] * orig[Z]);

    if (alpha >= real_type(1e-6))
    {
        const real_type inv_alpha = 1.0 / alpha;

        (*dir)[X] = orig[X] * costheta
                    + inv_alpha
                          * (orig[X] * orig[Z] * stcosphi - orig[Y] * stsinphi);
        (*dir)[Y] = orig[Y] * costheta
                    + inv_alpha
                          * (orig[Y] * orig[Z] * stcosphi + orig[X] * stsinphi);
        (*dir)[Z] = orig[Z] * costheta - alpha * stcosphi;
    }
    else
    {
        // Degenerate case: snap to polar axis
        (*dir)[X] = stcosphi;
        (*dir)[Y] = stsinphi;
        (*dir)[Z] = std::copysign(costheta, orig[Z]); // +-costheta
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
