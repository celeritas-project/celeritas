//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/DebugIO.json.cc
//---------------------------------------------------------------------------//
#include "DebugIO.json.hh"

#include "celeritas/track/SimTrackView.hh"

#include "ActionRegistry.hh"
#include "CoreParams.hh"
#include "Debug.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
template<class T>
int to_int(OpaqueId<T> id)
{
    if (id)
        return id.unchecked_get();
    return -1;
}

template<class T>
T const& passthrough(T const& inp)
{
    return inp;
}

nlohmann::json from_action(ActionId id)
{
    if (!id)
    {
        return "<invalid>";
    }
    if (g_debug_executing_params)
    {
        auto const& reg = *g_debug_executing_params->action_reg();
        return reg.action(id)->label();
    }
    return to_int(id);
}

//---------------------------------------------------------------------------//
}  // namespace
//---------------------------------------------------------------------------//

#define ASSIGN_TRANSFORMED(NAME, TRANSFORM) j[#NAME] = TRANSFORM(view.NAME())
#define ASSIGN_TRANSFORMED_IF(NAME, TRANSFORM, COND) \
    if (auto&& temp = view.NAME(); COND(temp))       \
    {                                                \
        j[#NAME] = TRANSFORM(temp);                  \
    }

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, SimTrackView const& view)
{
    ASSIGN_TRANSFORMED(status, to_cstring);
    if (view.status() != TrackStatus::inactive)
    {
        ASSIGN_TRANSFORMED(track_id, to_int);
        ASSIGN_TRANSFORMED(parent_id, to_int);
        ASSIGN_TRANSFORMED(event_id, to_int);
        ASSIGN_TRANSFORMED(num_steps, passthrough);
        ASSIGN_TRANSFORMED(event_id, to_int);
        ASSIGN_TRANSFORMED(time, passthrough);
        ASSIGN_TRANSFORMED(step_length, passthrough);
    }

    ASSIGN_TRANSFORMED_IF(num_looping_steps, passthrough, bool);
    ASSIGN_TRANSFORMED_IF(post_step_action, from_action, bool);
    ASSIGN_TRANSFORMED_IF(along_step_action, from_action, bool);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
