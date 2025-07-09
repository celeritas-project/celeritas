//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/DebugIO.json.cc
//---------------------------------------------------------------------------//
#include "DebugIO.json.hh"

#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/LabelIO.json.hh"
#include "corecel/math/QuantityIO.json.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/geo/CoreGeoParams.hh"
#include "celeritas/geo/GeoMaterialView.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/track/SimTrackView.hh"

#include "CoreParams.hh"
#include "CoreTrackView.hh"
#include "Debug.hh"

using celeritas::units::NativeTraits;

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Return an opaque ID as an integer value
template<class I, class T>
constexpr int to_int(OpaqueId<I, T> id)
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
struct Labeled
{
    std::string_view label;

    template<class T>
    nlohmann::json operator()(T const& obj) const
    {
        return nlohmann::json{obj, this->label};
    }
};

//---------------------------------------------------------------------------//
struct FromId
{
    CoreParams const* params{nullptr};

    //! Transform an ID to a label if possible
    template<class T, class S>
    nlohmann::json operator()(OpaqueId<T, S> id) const
    {
        if (!id)
        {
            return "<invalid>";
        }
        if (!this->params)
        {
            return to_int(id);
        }
        return this->convert_impl(id);
    }

    //! Transform to an action label if inside the stepping loop
    nlohmann::json convert_impl(ActionId id) const
    {
        auto const& reg = *this->params->action_reg();
        return reg.action(id)->label();
    }

    //! Transform to a particle label if inside the stepping loop
    nlohmann::json convert_impl(ParticleId id) const
    {
        auto const& params = *this->params->particle();
        return params.id_to_label(id);
    }

    //! Transform to a volume label
    nlohmann::json convert_impl(VolumeId id) const
    {
        auto const& params = *this->params->geometry();
        return params.volumes().at(id);
    }

    //! Transform to a material label
    nlohmann::json convert_impl(PhysMatId id) const
    {
        auto const& params = *this->params->material();
        return params.id_to_label(id);
    }
};

//---------------------------------------------------------------------------//
// Helper macros for writing data to JSON

#define ASSIGN_TRANSFORMED(NAME, TRANSFORM) j[#NAME] = TRANSFORM(view.NAME())
#define ASSIGN_TRANSFORMED_IF(NAME, TRANSFORM, COND) \
    if (auto&& temp = view.NAME(); COND(temp))       \
    {                                                \
        j[#NAME] = TRANSFORM(temp);                  \
    }

//---------------------------------------------------------------------------//
// Create JSON from geoetry view, using host metadata if possible
void to_json_impl(nlohmann::json& j, GeoTrackView const& view, FromId from_id)
{
    ASSIGN_TRANSFORMED(pos, Labeled{NativeTraits::Length::label()});
    ASSIGN_TRANSFORMED(dir, passthrough);
    ASSIGN_TRANSFORMED(is_outside, passthrough);
    ASSIGN_TRANSFORMED(is_on_boundary, passthrough);

    if (!view.is_outside())
    {
        ASSIGN_TRANSFORMED(volume_id, from_id);
    }
}

//---------------------------------------------------------------------------//
// Create JSON from particle view, using host metadata if possible
void to_json_impl(nlohmann::json& j,
                  GeoMaterialView const& view,
                  GeoTrackView const& geo,
                  FromId from_id)
{
    if (!geo.is_outside())
    {
        j = from_id(view.material_id(geo.volume_id()));
    }
    else
    {
        j = nullptr;
    }
}

//---------------------------------------------------------------------------//
// Create JSON from particle view, using host metadata if possible
void to_json_impl(nlohmann::json& j,
                  ParticleTrackView const& view,
                  FromId from_id)
{
    ASSIGN_TRANSFORMED(particle_id, from_id);
    ASSIGN_TRANSFORMED(energy, passthrough);
}

//---------------------------------------------------------------------------//
// Create JSON from sim view, using host metadata if possible
void to_json_impl(nlohmann::json& j, SimTrackView const& view, FromId from_id)
{
    ASSIGN_TRANSFORMED(status, to_cstring);
    if (view.status() != TrackStatus::inactive)
    {
        ASSIGN_TRANSFORMED(track_id, to_int);
        ASSIGN_TRANSFORMED(parent_id, to_int);
        ASSIGN_TRANSFORMED(event_id, to_int);
        ASSIGN_TRANSFORMED(num_steps, passthrough);
        ASSIGN_TRANSFORMED(event_id, to_int);
        ASSIGN_TRANSFORMED(time, Labeled{NativeTraits::Time::label()});
        ASSIGN_TRANSFORMED(step_length, Labeled{NativeTraits::Length::label()});

        ASSIGN_TRANSFORMED_IF(num_looping_steps, passthrough, bool);
    }

    ASSIGN_TRANSFORMED_IF(post_step_action, from_id, bool);
    ASSIGN_TRANSFORMED_IF(along_step_action, from_id, bool);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, CoreTrackView const& view)
{
    ASSIGN_TRANSFORMED(thread_id, to_int);
    ASSIGN_TRANSFORMED(track_slot_id, to_int);

    FromId from_id{view.core_scalars().host_core_params.get()};
    to_json_impl(j["sim"], view.sim(), from_id);

    if (view.sim().status() == TrackStatus::inactive)
    {
        // Skip all other output since the track is inactive
        return;
    }

    to_json_impl(j["geo"], view.geometry(), from_id);
    to_json_impl(j["mat"], view.geo_material(), view.geometry(), from_id);
    to_json_impl(j["particle"], view.particle(), from_id);
}

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, GeoTrackView const& view)
{
    return to_json_impl(j, view, {});
}

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, ParticleTrackView const& view)
{
    return to_json_impl(j, view, {});
}

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, SimTrackView const& view)
{
    return to_json_impl(j, view, {});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
