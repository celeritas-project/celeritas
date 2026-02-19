//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/DebugIO.json.cc
//---------------------------------------------------------------------------//
#include "DebugIO.json.hh"

#include "corecel/io/JsonUtils.json.hh"
#include "corecel/io/LabelIO.json.hh"
#include "corecel/math/QuantityIO.json.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/geo/CoreGeoParams.hh"
#include "celeritas/geo/CoreGeoTrackView.hh"
#include "celeritas/geo/GeoMaterialView.hh"
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
struct IdToJson
{
    CoreParams const* params{nullptr};

    //! Transform an ID to a label if possible
    template<class T, class S>
    nlohmann::json operator()(OpaqueId<T, S> id) const
    {
        if (this->params && id)
        {
            return this->convert_impl(id);
        }
        return id;
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
    nlohmann::json convert_impl(ImplVolumeId id) const
    {
        auto const& params = *this->params->geometry();
        return params.impl_volumes().at(id);
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

#define DIO_ASSIGN(NAME, TRANSFORM) j[#NAME] = TRANSFORM(view.NAME())
#define DIO_ASSIGN_IF(NAME, TRANSFORM, COND)   \
    if (auto&& temp = view.NAME(); COND(temp)) \
    {                                          \
        j[#NAME] = TRANSFORM(temp);            \
    }

//---------------------------------------------------------------------------//
// Create JSON from geometry view, using host metadata if possible
void to_json_impl(nlohmann::json& j,
                  GeoTrackView const& view,
                  IdToJson id_to_json)
{
    DIO_ASSIGN(pos, Labeled{NativeTraits::Length::label()});
    DIO_ASSIGN(dir, passthrough);
    DIO_ASSIGN(is_outside, passthrough);
    DIO_ASSIGN(is_on_boundary, passthrough);

    if (!view.is_outside())
    {
        j["volume_id"] = id_to_json(view.impl_volume_id());
    }
}

//---------------------------------------------------------------------------//
// Create JSON from particle view, using host metadata if possible
void to_json_impl(nlohmann::json& j,
                  GeoMaterialView const& view,
                  GeoTrackView const& geo,
                  IdToJson id_to_json)
{
    if (!geo.is_outside())
    {
        j = id_to_json(view.material_id(geo.impl_volume_id()));
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
                  IdToJson id_to_json)
{
    DIO_ASSIGN(particle_id, id_to_json);
    DIO_ASSIGN(energy, passthrough);
}

//---------------------------------------------------------------------------//
// Create JSON from sim view, using host metadata if possible
void to_json_impl(nlohmann::json& j,
                  SimTrackView const& view,
                  IdToJson id_to_json)
{
    DIO_ASSIGN(status, to_cstring);
    if (view.status() != TrackStatus::inactive)
    {
        DIO_ASSIGN(track_id, passthrough);
        DIO_ASSIGN(parent_id, passthrough);
        DIO_ASSIGN(event_id, passthrough);
        DIO_ASSIGN(num_steps, passthrough);
        DIO_ASSIGN(event_id, passthrough);
        DIO_ASSIGN(time, Labeled{NativeTraits::Time::label()});
        DIO_ASSIGN(step_length, Labeled{NativeTraits::Length::label()});

        DIO_ASSIGN_IF(num_looping_steps, passthrough, bool);
    }

    DIO_ASSIGN_IF(post_step_action, id_to_json, bool);
    DIO_ASSIGN_IF(along_step_action, id_to_json, bool);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, CoreTrackView const& view)
{
    DIO_ASSIGN(thread_id, passthrough);
    DIO_ASSIGN(track_slot_id, passthrough);

    IdToJson id_to_json{view.core_scalars().host_core_params.get()};
    to_json_impl(j["sim"], view.sim(), id_to_json);

    if (view.sim().status() == TrackStatus::inactive)
    {
        // Skip all other output since the track is inactive
        return;
    }

    to_json_impl(j["geo"], view.geometry(), id_to_json);
    to_json_impl(j["mat"], view.geo_material(), view.geometry(), id_to_json);
    to_json_impl(j["particle"], view.particle(), id_to_json);
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
