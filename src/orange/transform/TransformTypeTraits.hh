//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/TransformTypeTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

class NoTransformation;
class Transformation;
class Translation;

//---------------------------------------------------------------------------//
/*!
 * Map transform enumeration to transform class.
 */
template<TransformType S>
struct TransformTypeTraits;

#define ORANGE_TRANSFORM_TRAITS(ENUM_VALUE, CLS)          \
    template<>                                            \
    struct TransformTypeTraits<TransformType::ENUM_VALUE> \
    {                                                     \
        using type = CLS;                                 \
    }

ORANGE_TRANSFORM_TRAITS(no_transformation, NoTransformation);
ORANGE_TRANSFORM_TRAITS(translation, Translation);
ORANGE_TRANSFORM_TRAITS(transformation, Transformation);

#undef ORANGE_TRANSFORM_TRAITS

//---------------------------------------------------------------------------//
/*!
 * Expand a macro to a switch statement over all possible transform types.
 *
 * The \c func argument should be a functor that takes a single argument which
 * is a TransformTypeTraits instance.
 */
template<class F>
CELER_CONSTEXPR_FUNCTION decltype(auto)
visit_transform_type(F&& func, TransformType st)
{
#define ORANGE_TT_VISIT_CASE(TYPE)          \
    case TransformType::TYPE:               \
        return celeritas::forward<F>(func)( \
            TransformTypeTraits<TransformType::TYPE>{})

    switch (st)
    {
        ORANGE_TT_VISIT_CASE(no_transformation);
        ORANGE_TT_VISIT_CASE(translation);
        ORANGE_TT_VISIT_CASE(transformation);
        default:
            CELER_ASSERT_UNREACHABLE();
    }
#undef ORANGE_TT_VISIT_CASE
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
