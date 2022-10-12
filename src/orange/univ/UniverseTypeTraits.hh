//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/UniverseTypeTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct SimpleUnitRecord;
class SimpleUnitTracker;

//---------------------------------------------------------------------------//
/*!
 * Map universe enumeration to surface data and tracker classes.
 */
template<UniverseType U>
struct UniverseTypeTraits;

#define ORANGE_UNIV_TRAITS(ENUM_VALUE, CLS)             \
    template<>                                          \
    struct UniverseTypeTraits<UniverseType::ENUM_VALUE> \
    {                                                   \
        using record_type  = CLS##Record;               \
        using tracker_type = CLS##Tracker;              \
    }

ORANGE_UNIV_TRAITS(simple, SimpleUnit);

#undef ORANGE_UNIV_TRAITS

//---------------------------------------------------------------------------//
} // namespace celeritas
