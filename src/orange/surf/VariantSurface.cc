//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/VariantSurface.cc
//---------------------------------------------------------------------------//
#include "VariantSurface.hh"

#include "corecel/cont/VariantUtils.hh"

#include "detail/SurfaceTransformer.hh"
#include "detail/SurfaceTranslator.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
struct VariantTransformDispatcher
{
    VariantSurface const& right;

    //! Apply an identity transform (no change)
    VariantSurface operator()(NoTransformation const&) const { return right; }

    //! Apply a translation
    VariantSurface operator()(Translation const& left) const
    {
        CELER_ASSUME(!right.valueless_by_exception());
        return std::visit(
            return_as<VariantSurface>(detail::SurfaceTranslator{left}), right);
    }

    //! Apply a transformation
    VariantSurface operator()(Transformation const& left) const
    {
        CELER_ASSUME(!right.valueless_by_exception());
        return std::visit(
            return_as<VariantSurface>(detail::SurfaceTransformer{left}), right);
    }
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Apply a variant "daughter-to-parent" transform to a surface.
 */
[[nodiscard]] VariantSurface apply_transform(VariantTransform const& transform,
                                             VariantSurface const& surface)
{
    CELER_ASSUME(!transform.valueless_by_exception());
    return std::visit(VariantTransformDispatcher{surface}, transform);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
