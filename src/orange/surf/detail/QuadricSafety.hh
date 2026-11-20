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
 * Calculate a lower bound on the distance from a point to a quadric surface.
 *
 * A quadric \f$ f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} + \mathbf{b} \cdot
 * \mathbf{x} + c \f$ is exactly described by its second-order expansion about
 * any point \f$ \mathbf{x} \f$:
 * \f[
   f(\mathbf{x} + \boldsymbol{\delta}) = f(\mathbf{x})
     + \nabla f(\mathbf{x}) \cdot \boldsymbol{\delta}
     + \boldsymbol{\delta}^T A \boldsymbol{\delta} \,.
   \f]
 * Let \f$ g = |\nabla f(\mathbf{x})| \f$ and let the eigenvalues of \em A
 * lie in \f$ [\mu_-, \mu_+] \f$. For a displacement of length \em s, the
 * linear term changes \em f by at most \f$ g s \f$ and the quadratic term lies
 * in \f$ [\mu_- s^2, \mu_+ s^2] \f$. Reaching \f$ f = 0 \f$ requires a
 * change of \f$ |f| \f$ toward zero, which needs
 * \f$ g s + \lambda s^2 \ge |f| \f$, where only curvature *toward* the
 * surface counts:
 * \f[
   \lambda = \begin{cases}
     \max(0, -\mu_-) & f > 0 \\
     \max(0, \mu_+) & f < 0
   \end{cases} \,.
   \f]
 * The distance to the surface is therefore at least the positive root
 * \f[
   s = \frac{2|f|}{g + \sqrt{g^2 + 4 \lambda |f|}} \,,
   \f]
 * written in the form that is stable as \f$ \lambda \to 0 \f$ (where it
 * becomes the plane distance \f$ |f|/g \f$).
 *
 * The bound approaches the true distance near the surface. It is exact
 * everywhere inside a sphere or circular cylinder; outside, where the surface
 * curves away from the point, it underestimates by up to a factor of two far
 * from the surface.
 *
 * \param value  Quadric expression \f$ f(\mathbf{x}) \f$
 * \param grad   Gradient \f$ \nabla f(\mathbf{x}) \f$ (not normalized)
 * \param min_eigen Lower bound \f$ \mu_- \f$ on the eigenvalues of \em A
 * \param max_eigen Upper bound \f$ \mu_+ \f$ on the eigenvalues of \em A
 */
inline CELER_FUNCTION real_type calc_quadric_safety(real_type value,
                                                    Real3 const& grad,
                                                    real_type min_eigen,
                                                    real_type max_eigen)
{
    if (value == 0)
    {
        // On the surface (or at a singular point such as a cone apex, where
        // the gradient also vanishes)
        return 0;
    }
    real_type const lambda = value > 0
                                 ? celeritas::max(real_type(0), -min_eigen)
                                 : celeritas::max(real_type(0), max_eigen);
    real_type const abs_value = std::fabs(value);
    real_type const g = norm(grad);
    return 2 * abs_value / (g + std::sqrt(ipow<2>(g) + 4 * lambda * abs_value));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
