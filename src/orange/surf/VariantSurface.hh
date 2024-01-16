//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/VariantSurface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <variant>

#include "corecel/cont/VariantUtils.hh"
#include "orange/transform/VariantTransform.hh"

#include "SurfaceTypeTraits.hh"
#include "detail/AllSurfaces.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! std::variant class of all surfaces.
using VariantSurface = EnumVariant<SurfaceType, SurfaceTypeTraits>;

//---------------------------------------------------------------------------//
// Apply a variant "daughter-to-parent" transform to a surface
[[nodiscard]] VariantSurface apply_transform(VariantTransform const& transform,
                                             VariantSurface const& surface);

//---------------------------------------------------------------------------//
}  // namespace celeritas
