//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/VariantTransform.hh
//---------------------------------------------------------------------------//
#pragma once

#include <variant>

#include "Transformation.hh"
#include "Translation.hh"

namespace celeritas
{
template<class T>
class BoundingBox;

//---------------------------------------------------------------------------//
//! std::variant for all transforms, with optional identity transform
using VariantTransform
    = std::variant<std::monostate, Translation, Transformation>;

//---------------------------------------------------------------------------//
// Apply the left "daughter-to-parent" transform to the right.
[[nodiscard]] VariantTransform
apply_transform(VariantTransform const& left, VariantTransform const& right);

//---------------------------------------------------------------------------//
// Dispatch "daughter-to-parent" transform to bounding box utilities
[[nodiscard]] BoundingBox<real_type>
apply_transform(VariantTransform const& transform,
                BoundingBox<real_type> const& bbox);

//---------------------------------------------------------------------------//
}  // namespace celeritas
