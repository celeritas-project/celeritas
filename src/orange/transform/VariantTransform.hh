//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/VariantTransform.hh
//---------------------------------------------------------------------------//
#pragma once

#include <variant>

#include "corecel/cont/VariantUtils.hh"

#include "NoTransformation.hh"
#include "TransformTypeTraits.hh"
#include "Transformation.hh"
#include "Translation.hh"

namespace celeritas
{
template<class T>
class BoundingBox;

//---------------------------------------------------------------------------//
//! std::variant for all transforms.
using VariantTransform = EnumVariant<TransformType, TransformTypeTraits>;

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
