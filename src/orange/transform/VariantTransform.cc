//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/VariantTransform.cc
//---------------------------------------------------------------------------//
#include "VariantTransform.hh"

#include "detail/TransformTransformer.hh"
#include "detail/TransformTranslator.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Wrap a Transformer or Translator to return a variant.
 */
template<class T>
struct TransWrapper
{
    T apply;

    // Apply to a monostate
    VariantTransform operator()(std::monostate m) const
    {
        return this->apply(m);
    }

    // Apply to a translation
    VariantTransform operator()(Translation const& right) const
    {
        return this->apply(right);
    }

    // Apply to a translation
    VariantTransform operator()(Transformation const& right) const
    {
        return this->apply(right);
    }
};

// Deduction guide
template<class T>
TransWrapper(T&&) -> TransWrapper<T>;

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
        return std::visit(TransWrapper{detail::TransformTranslator{left}},
                          right);
    }

    //! Apply a transformation
    VariantTransform operator()(Transformation const& left) const
    {
        if (right.valueless_by_exception())
        {
            CELER_ASSERT_UNREACHABLE();
        }
        return std::visit(TransWrapper{detail::TransformTransformer{left}},
                          right);
    }
};

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
   \mathbf{R}' = \mathbf{R}_R
 * \f]
 * and translation
 * \f[
   \mathbf{t}' = \mathbf{R}_L\mathbf{t}_R + \mathbf{t}_L
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
}  // namespace celeritas
