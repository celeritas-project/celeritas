//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
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
    VariantTransform operator()(std::monostate) const { return right; }

    //! Apply a translation
    VariantTransform operator()(Translation const& left) const
    {
        if (right.valueless_by_exception())
        {
            CELER_ASSERT_UNREACHABLE();
        }
        return std::visit(
            return_as<VariantTransform>(detail::TransformTranslator{left}),
            right);
    }

    //! Apply a transformation
    VariantTransform operator()(Transformation const& left) const
    {
        if (right.valueless_by_exception())
        {
            CELER_ASSERT_UNREACHABLE();
        }
        return std::visit(
            return_as<VariantTransform>(detail::TransformTransformer{left}),
            right);
    }
};

//---------------------------------------------------------------------------//
//! Call the "calc_transform" free function
template<class T>
struct VariantBboxDispatcher
{
    BoundingBox<T> const& right;

    //! Apply an identity transform (no change)
    BoundingBox<T> operator()(std::monostate) const { return right; }

    //! Apply a translation
    template<class U>
    BoundingBox<T> operator()(U const& left) const
    {
        return calc_transform(left, right);
    }
};

// Deduction guide
template<class T>
VariantBboxDispatcher(BoundingBox<T> const&) -> VariantBboxDispatcher<T>;

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
    if (left.valueless_by_exception())
    {
        CELER_ASSERT_UNREACHABLE();
    }
    return std::visit(VariantTransformDispatcher{right}, left);
}

//---------------------------------------------------------------------------//
/*!
 * Dispatch left "daughter-to-parent" transform to bounding box utilities.
 */
[[nodiscard]] BBox
apply_transform(VariantTransform const& left, BBox const& right)
{
    if (left.valueless_by_exception())
    {
        CELER_ASSERT_UNREACHABLE();
    }
    return std::visit(VariantBboxDispatcher{right}, left);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
