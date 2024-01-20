//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/VariantTransform.cc
//---------------------------------------------------------------------------//
#include "VariantTransform.hh"

#include "corecel/cont/VariantUtils.hh"
#include "orange/BoundingBoxUtils.hh"

#include "detail/TransformTransformer.hh"
#include "detail/TransformTranslator.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
struct VariantTransformDispatcher
{
    VariantTransform const& right;

    //! Apply an identity transform (no change)
    VariantTransform operator()(NoTransformation const&) const
    {
        return right;
    }

    //! Apply a translation
    VariantTransform operator()(Translation const& left) const
    {
        CELER_ASSUME(!right.valueless_by_exception());
        return std::visit(
            return_as<VariantTransform>(detail::TransformTranslator{left}),
            right);
    }

    //! Apply a transformation
    VariantTransform operator()(Transformation const& left) const
    {
        CELER_ASSUME(!right.valueless_by_exception());
        return std::visit(
            return_as<VariantTransform>(detail::TransformTransformer{left}),
            right);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Apply an identity transform to a bounding box.
 */
template<class T>
BoundingBox<T>
calc_transform(NoTransformation const&, BoundingBox<T> const& bbox)
{
    return bbox;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Apply the left "daughter-to-parent" transform to the right.
 *
 * The resulting variant may be a monostate (identity), translation (no
 * rotation), or full transformation.
 *
 * The resulting transform has rotation
 * \f[
   \mathbf{R}' = \mathbf{R}_2
 * \f]
 * and translation
 * \f[
   \mathbf{t}' = \mathbf{R}_1\mathbf{t}_2 + \mathbf{t}_1
 * \f]
 */
[[nodiscard]] VariantTransform
apply_transform(VariantTransform const& left, VariantTransform const& right)
{
    CELER_ASSUME(!left.valueless_by_exception());
    return std::visit(VariantTransformDispatcher{right}, left);
}

//---------------------------------------------------------------------------//
/*!
 * Dispatch "daughter-to-parent" transform to bounding box utilities.
 */
[[nodiscard]] BBox
apply_transform(VariantTransform const& transform, BBox const& bbox)
{
    CELER_ASSUME(!transform.valueless_by_exception());
    // Dispatch to bounding box utils or "monostate" case above
    return std::visit([&bbox](auto&& t) { return calc_transform(t, bbox); },
                      transform);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
