//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/DebugIO.json.cc
//---------------------------------------------------------------------------//
#include "DebugIO.json.hh"

#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/LabelIO.json.hh"
#include "corecel/math/QuantityIO.json.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/track/SimTrackView.hh"

#include "ActionRegistry.hh"
#include "CoreParams.hh"
#include "Debug.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Return an opaque ID as an integer value
template<class T>
int to_int(OpaqueId<T> id)
{
    if (id)
        return id.unchecked_get();
    return -1;
}

//---------------------------------------------------------------------------//
//! Pass through "transform" as an identity operation
template<class T>
T const& passthrough(T const& inp)
{
    return inp;
}

//---------------------------------------------------------------------------//
//! Transform to an action label if inside the stepping loop
nlohmann::json from_id(ActionId id)
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
//! Transform to a particle label if inside the stepping loop
nlohmann::json from_id(ParticleId id)
{
    if (!id)
    {
        return "<invalid>";
    }
    if (g_debug_executing_params)
    {
        auto const& params = *g_debug_executing_params->particle();
        return params.id_to_label(id);
    }
    return to_int(id);
}

//---------------------------------------------------------------------------//
//! Transform to a volume label
nlohmann::json from_id(VolumeId id)
{
    if (!id)
    {
        return "<invalid>";
    }
    if (g_debug_executing_params)
    {
        auto const& params = *g_debug_executing_params->geometry();
        return params.id_to_label(id);
    }
    return to_int(id);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
// Helper macros for writing data to JSON

#define ASSIGN_TRANSFORMED(NAME, TRANSFORM) j[#NAME] = TRANSFORM(view.NAME())
#define ASSIGN_TRANSFORMED_IF(NAME, TRANSFORM, COND) \
    if (auto&& temp = view.NAME(); COND(temp))       \
    {                                                \
        j[#NAME] = TRANSFORM(temp);                  \
    }

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, CoreTrackView const& view)
{
    ASSIGN_TRANSFORMED(thread_id, to_int);
    ASSIGN_TRANSFORMED(track_slot_id, to_int);

    j["sim"] = view.make_sim_view();

    if (view.make_sim_view().status() == TrackStatus::inactive)
    {
        // Skip all other output since the track is inactive
        return;
    }

    j["geo"] = view.make_geo_view();
    j["particle"] = view.make_particle_view();
}

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, GeoTrackView const& view)
{
    ASSIGN_TRANSFORMED(pos, passthrough);
    ASSIGN_TRANSFORMED(dir, passthrough);
    ASSIGN_TRANSFORMED(is_outside, passthrough);
    ASSIGN_TRANSFORMED(is_on_boundary, passthrough);

    if (!view.is_outside())
    {
        ASSIGN_TRANSFORMED(volume_id, from_id);
    }
}

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, ParticleTrackView const& view)
{
    ASSIGN_TRANSFORMED(particle_id, from_id);
    ASSIGN_TRANSFORMED(energy, passthrough);
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
    ASSIGN_TRANSFORMED_IF(post_step_action, from_id, bool);
    ASSIGN_TRANSFORMED_IF(along_step_action, from_id, bool);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
