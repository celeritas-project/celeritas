//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ArrayUtils.i.hh
//---------------------------------------------------------------------------//
#include <cmath>
#include "Algorithms.hh"
#include "Assert.hh"
#include "SoftEqual.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Traits for operations on Real3 vectors
template<class T>
struct RealVecTraits;

template<>
struct RealVecTraits<float>
{
    //! Threshold for rotation
    static CELER_CONSTEXPR_FUNCTION float min_accurate_sintheta()
    {
        return 0.07f;
    }
};

template<>
struct RealVecTraits<double>
{
    //! Threshold for rotation
    static CELER_CONSTEXPR_FUNCTION double min_accurate_sintheta()
    {
        return 0.005;
    }
};

} // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Increment a vector by another vector multiplied by a scalar.
 */
template<class T, size_type N>
CELER_FUNCTION void axpy(T a, const Array<T, N>& x, Array<T, N>* y)
{
    CELER_EXPECT(y);
    for (size_type i = 0; i != N; ++i)
    {
        (*y)[i] = a * x[i] + (*y)[i];
    }
}

//---------------------------------------------------------------------------//
/*!
 * Dot product of two vectors.
 */
template<class T, size_type N>
CELER_FUNCTION T dot_product(const Array<T, N>& x, const Array<T, N>& y)
{
    T result{};
    for (size_type i = 0; i != N; ++i)
    {
        result += x[i] * y[i];
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Cross product of two space vectors.
 */
template<class T>
CELER_FUNCTION Array<T, 3>
               cross_product(const Array<T, 3>& A, const Array<T, 3>& B)
{
    return {A[1] * B[2] - A[2] * B[1],
            A[2] * B[0] - A[0] * B[2],
            A[0] * B[1] - A[1] * B[0]};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the Euclidian (2) norm of a vector.
 */
template<class T, size_type N>
CELER_FUNCTION T norm(const Array<T, N>& v)
{
    return std::sqrt(dot_product(v, v));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the Euclidian (2) distance between two points.
 */
template<class T, size_type N>
CELER_FUNCTION T distance(const Array<T, N>& x, const Array<T, N>& y)
{
    real_type dist_sq = 0;
    for (size_type i = 0; i != N; ++i)
    {
        dist_sq += ipow<2>(y[i] - x[i]);
    }
    return std::sqrt(dist_sq);
}

//---------------------------------------------------------------------------//
/*!
 * Divide the given vector by its Euclidian norm.
 */
CELER_FUNCTION void normalize_direction(Real3* direction)
{
    CELER_EXPECT(direction);
    const real_type scale_factor = 1 / norm(*direction);
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
    CELER_EXPECT(costheta >= -1 && costheta <= 1);

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
 * defined
 * \f[
   \Omega =   \sin\theta\cos\phi\mathbf{i}
            + \sin\theta\sin\phi\mathbf{j}
            + \cos\theta\mathbf{k}
 * \f]
 */
inline CELER_FUNCTION Real3 rotate(const Real3& dir, const Real3& rot)
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
    real_type sintheta = std::sqrt(1 - ipow<2>(rot[Z]));
    real_type cosphi;
    real_type sinphi;

    if (sintheta >= detail::RealVecTraits<real_type>::min_accurate_sintheta())
    {
        // Typical case: far enough from z axis to assume the X and Y
        // components have a hypotenuse of 1 within epsilon tolerance
        const real_type inv_sintheta = 1 / (sintheta);
        cosphi                       = rot[X] * inv_sintheta;
        sinphi                       = rot[Y] * inv_sintheta;
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

    Real3 result
        = {(rot[Z] * dir[X] + sintheta * dir[Z]) * cosphi - sinphi * dir[Y],
           (rot[Z] * dir[X] + sintheta * dir[Z]) * sinphi + cosphi * dir[Y],
           -sintheta * dir[X] + rot[Z] * dir[Z]};

    CELER_ENSURE(is_soft_unit_vector(result));
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Test for being approximately a unit vector.
 *
 * Example:
 * \code
  CELER_EXPECT(is_soft_unit_vector(v))
  \endcode
 */
template<class T, size_type N>
CELER_FUNCTION bool is_soft_unit_vector(const Array<T, N>& v)
{
    // (1 + eps, 0, 0) is not quite allowed for 2*eps precision; increase
    SoftEqual<T> cmp{10 * detail::SoftEqualTraits<T>::rel_prec()};
    return cmp(T(1), dot_product(v, v));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
