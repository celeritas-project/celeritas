//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Types.cc
//---------------------------------------------------------------------------//
#include "Types.hh"

#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/StringEnumMapper.hh"

#include "UnitTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a unit system.
 */
char const* to_cstring(UnitSystem value)
{
    static_assert(static_cast<int>(UnitSystem::cgs) == CELERITAS_UNITS_CGS);
    static_assert(static_cast<int>(UnitSystem::si) == CELERITAS_UNITS_SI);
    static_assert(static_cast<int>(UnitSystem::clhep) == CELERITAS_UNITS_CLHEP);
    static_assert(static_cast<int>(UnitSystem::native) == CELERITAS_UNITS);

    static EnumStringMapper<UnitSystem> const to_cstring_impl{
        "none", "cgs", "si", "clhep"};
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a unit system corresponding to a string value.
 */
UnitSystem to_unit_system(std::string const& s)
{
    static auto const from_string
        = StringEnumMapper<UnitSystem>::from_cstring_func(to_cstring,
                                                          "unit system");
    return from_string(s);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a state of matter.
 */
char const* to_cstring(MatterState value)
{
    static EnumStringMapper<MatterState> const to_cstring_impl{
        "unspecified",
        "solid",
        "liquid",
        "gas",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to an action order.
 */
char const* to_cstring(ActionOrder value)
{
    static EnumStringMapper<ActionOrder> const to_cstring_impl{
        "start",
        "sort_start",
        "pre",
        "sort_pre",
        "along",
        "sort_along",
        "pre_post",
        "sort_pre_post",
        "post",
        "post_post",
        "end",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a track ordering policy.
 */
char const* to_cstring(TrackOrder value)
{
    static EnumStringMapper<TrackOrder> const to_cstring_impl{
        "unsorted",
        "shuffled",
        "partition_status",
        "sort_along_step_action",
        "sort_step_limit_action",
        "sort_action",
        "sort_particle_type",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Checks that the TrackOrder will sort tracks by actions applied at the given
 * ActionOrder. This should match the mapping in the \c SortTracksAction
 * constructor.
 *
 * TODO: Have a single source of truth for mapping TrackOrder to ActionOrder
 */
bool is_action_sorted(ActionOrder action, TrackOrder track)
{
    return (action == ActionOrder::post
            && track == TrackOrder::sort_step_limit_action)
           || (action == ActionOrder::along
               && track == TrackOrder::sort_along_step_action)
           || (track == TrackOrder::sort_action
               && (action == ActionOrder::post || action == ActionOrder::along));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
