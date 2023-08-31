//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/TransformTypeTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/OrangeTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

class NoTransformation;
class Transformation;
class Translation;

//---------------------------------------------------------------------------//
/*!
 * Map transform enumeration to surface type.
 */
template<TransformType S>
struct TransformTypeTraits;

#define ORANGE_TRANSFORM_TRAITS(ENUM_VALUE, CLS)          \
    template<>                                            \
    struct TransformTypeTraits<TransformType::ENUM_VALUE> \
    {                                                     \
        using type = CLS;                                 \
    }

// clang-format off
ORANGE_TRANSFORM_TRAITS(no_transformation, NoTransformation);
ORANGE_TRANSFORM_TRAITS(translation, Translation);
ORANGE_TRANSFORM_TRAITS(transformation, Transformation);
// clang-format on

#undef ORANGE_TRANSFORM_TRAITS

//---------------------------------------------------------------------------//
}  // namespace celeritas
