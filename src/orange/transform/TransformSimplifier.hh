//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/TransformSimplifier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/math/SoftEqual.hh"
#include "orange/OrangeTypes.hh"

#include "VariantTransform.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Return a simplified version of a transformation.
 *
 * Like surface simplification, we want to consider whether two different
 * transformations will result in a distance change of \f$\epsilon\f$ for a
 * point that's at the length scale from the origin. Setting the length scale
 * to unity (the default), we use the relative tolerance.
 *
 * A \em translation can be deleted if its magnitude is less than epsilon.
 *
 * For a \em rotation, we use the fact that the trace (sum of diagonal
 elements)
 * of any proper (non-reflecting) rotation matrix has an angle of rotation \f[
   \mathrm{Tr}[R] = 2 \cos \theta + 1
 * \f] about an axis of rotation. Applying the rotation to a point
 * at a unit distance will yield an iscoceles triangle with sides 1 and inner
 * angle \f$\theta\f$. For the displacement to be no more than \f$\epsilon\f$
 * then the angle must be \f[
  \sin \theta/2 \le \epsilon/2
  \f]
 * which with some manipulation means that a "soft zero" rotation has a trace
 * \f[
 \mathrm{Tr}[R] \ge 3 - \epsilon^2 \,.
 \f]
 *
 * Note that this means no rotational simplifications may be performed when the
 * geometry tolerance is less than the square root of machine precision.
 *
 * \todo Define soft_identity functor
 */
class TransformSimplifier
{
  public:
    // Construct with tolerance
    explicit inline TransformSimplifier(Tolerance<> const& tol);

    //! No simplification can be applied to a null transformation
    VariantTransform operator()(NoTransformation const& nt) { return nt; }

    // Translation may simplify to no transformation
    VariantTransform operator()(Translation const& nt);

    // Simplify, possibly to translation or no transform
    VariantTransform operator()(Transformation const& nt);

  private:
    SoftZero<> soft_zero_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with tolerance.
 */
TransformSimplifier::TransformSimplifier(Tolerance<> const& tol)
    : soft_zero_{tol.rel}
{
    CELER_EXPECT(tol);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
