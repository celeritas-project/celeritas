//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/VariantSurface.cc
//---------------------------------------------------------------------------//
#include "VariantSurface.hh"

#include "detail/SurfaceTransformer.hh"
#include "detail/SurfaceTranslator.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Wrap a Transformer or Translator to return a uniform variant.
 */
template<class T>
struct VarSurfWrapper
{
    T apply;

    template<class U>
    VariantSurface operator()(U&& other)
    {
        return this->apply(std::forward<U>(other));
    }
};

// Deduction guide
template<class T>
VarSurfWrapper(T&&) -> VarSurfWrapper<T>;

//---------------------------------------------------------------------------//
struct VariantTransformDispatcher
{
    VariantSurface const& right;

    //! Apply an identity transform (no change)
    VariantSurface operator()(std::monostate) const { return right; }

    //! Apply a translation
    VariantSurface operator()(Translation const& left) const
    {
        if (right.valueless_by_exception())
        {
            CELER_ASSERT_UNREACHABLE();
        }
        return std::visit(VarSurfWrapper{detail::SurfaceTranslator{left}},
                          right);
    }

    //! Apply a transformation
    VariantSurface operator()(Transformation const& left) const
    {
        if (right.valueless_by_exception())
        {
            CELER_ASSERT_UNREACHABLE();
        }
        return std::visit(VarSurfWrapper{detail::SurfaceTransformer{left}},
                          right);
    }
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Apply a variant "daughter-to-parent" transform to a surface.
 */
[[nodiscard]] VariantSurface
apply_transform(VariantTransform const& left, VariantSurface const& right)
{
    if (left.valueless_by_exception())
    {
        CELER_ASSERT_UNREACHABLE();
    }
    return std::visit(VariantTransformDispatcher{right}, left);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
