//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/UnivTypeTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct SimpleUnitRecord;
class SimpleUnitTracker;
class RectArrayTracker;

//---------------------------------------------------------------------------//
/*!
 * Map universe enumeration to surface data and tracker classes.
 */
template<UnivType U>
struct UnivTypeTraits;

#define ORANGE_UNIV_TRAITS(ENUM_VALUE, CLS)     \
    template<>                                  \
    struct UnivTypeTraits<UnivType::ENUM_VALUE> \
    {                                           \
        using record_type = CLS##Record;        \
        using tracker_type = CLS##Tracker;      \
    }

ORANGE_UNIV_TRAITS(simple, SimpleUnit);
ORANGE_UNIV_TRAITS(rect_array, RectArray);

#undef ORANGE_UNIV_TRAITS

//---------------------------------------------------------------------------//
/*!
 * Expand a macro to a switch statement over all possible universe types.
 *
 * The \c func argument should be a functor that takes a single argument which
 * is a UnivTypeTraits instance.
 */
template<class F>
CELER_CONSTEXPR_FUNCTION decltype(auto) visit_univ_type(F&& func, UnivType ut)
{
#define ORANGE_UT_VISIT_CASE(TYPE) \
    case UnivType::TYPE:           \
        return celeritas::forward<F>(func)(UnivTypeTraits<UnivType::TYPE>{})

    switch (ut)
    {
        ORANGE_UT_VISIT_CASE(simple);
        ORANGE_UT_VISIT_CASE(rect_array);
        default:
            CELER_ASSERT_UNREACHABLE();
    }
#undef ORANGE_UT_VISIT_CASE
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
