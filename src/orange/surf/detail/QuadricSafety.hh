//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/QuadricSafety.hh
//! \sa test/orange/surf/SimpleQuadric.test.cc
//! \sa test/orange/surf/GeneralQuadric.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate a conservative lower bound on the distance to a quadric surface.
 *
 * A quadric surface is defined by \f$ f(\mathbf{x}) = 0 \f$, where \f$
 * f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} + \mathbf{b} \cdot \mathbf{x} + c
 * \f$ and \em A is symmetric. Since \em f is quadratic, its second-order
 * expansion is exact:
 * \f[
   f(\mathbf{x} + \boldsymbol{\delta}) = f(\mathbf{x})
     + \nabla f(\mathbf{x}) \cdot \boldsymbol{\delta}
     + \boldsymbol{\delta}^T A \boldsymbol{\delta} \,.
 * \f]
 *
 * Let \f$ g = |\nabla f(\mathbf{x})| \f$, and let \f$ \mu_- \f$ be a lower
 * bound on the smallest eigenvalue of \em A and \f$ \mu_+ \f$ an upper bound
 * on the largest. For a displacement \f$ \boldsymbol{\delta} \f$ of length \f$
 * s = |\boldsymbol{\delta}| \f$, the linear contribution \f$ \nabla f \cdot
 * \boldsymbol{\delta} \f$ has magnitude at most \f$ gs \f$. The quadratic
 * contribution \f$ \boldsymbol{\delta}^T A \boldsymbol{\delta} \f$ lies
 * between \f$ \mu_- s^2 \f$ and \f$ \mu_+ s^2 \f$.
 *
 * To reach the surface, the value of \em f must increase or decrease to zero,
 * requiring a change of magnitude \f$ |f| \f$. The maximum possible change
 * toward zero is bounded by \f$ gs + \lambda s^2 \f$, where
 * \f[
   \lambda = \begin{cases}
     \max(0, -\mu_-) & f > 0 \\
     \max(0, \mu_+) & f < 0
   \end{cases} \,.
 * \f]
 * Therefore, the surface cannot be reached until \f$ gs + \lambda s^2 \ge |f|
 * \f$. Solving for \em s gives the conservative safety distance:
 * \f[
   s = \frac{2|f|}{g + \sqrt{g^2 + 4 \lambda |f|}} \,.
 * \f]
 * This form avoids subtracting nearly equal values and reduces to \f$ |f|/g
 * \f$ when \f$ \lambda = 0 \f$ and \f$ g > 0 \f$.
 *
 * The bound approaches the true distance near a smooth surface. For spheres
 * and circular cylinders with exact eigenvalue bounds, it is exact inside the
 * surface and underestimates the true distance by at most a factor of two
 * outside. The bound may be more conservative for other quadrics or when the
 * eigenvalue bounds are loose.
 *
 * \param value Quadric expression \f$ f(\mathbf{x}) \f$
 * \param gradient Gradient \f$ \nabla f(\mathbf{x}) \f$ (not normalized)
 * \param min_eigen_bound Lower bound \f$ \mu_- \f$ on the eigenvalues of \em A
 * \param max_eigen_bound Upper bound \f$ \mu_+ \f$ on the eigenvalues of \em A
 */
inline CELER_FUNCTION real_type calc_quadric_safety(real_type value,
                                                    Real3 const& gradient,
                                                    real_type min_eigen_bound,
                                                    real_type max_eigen_bound)
{
    using namespace celeritas::literals;

    CELER_EXPECT(min_eigen_bound <= max_eigen_bound);

    if (value == 0)
    {
        // On the surface (or at a singular point such as a cone apex, where
        // the gradient also vanishes)
        return 0;
    }

    real_type const lambda = value > 0 ? celeritas::max(0_r, -min_eigen_bound)
                                       : celeritas::max(0_r, max_eigen_bound);
    real_type const abs_value = std::fabs(value);
    real_type const g = norm(gradient);
    real_type const result
        = 2 * abs_value / (g + std::sqrt(ipow<2>(g) + 4 * lambda * abs_value));

    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
