//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/DebugIO.json.cc
//---------------------------------------------------------------------------//
#include "DebugIO.json.hh"

#include "corecel/io/JsonUtils.json.hh"
#include "corecel/io/LabelIO.json.hh"
#include "corecel/math/QuantityIO.json.hh"
#include "geocel/VolumeParams.hh"

#include "LevelStateAccessor.hh"
#include "OrangeParams.hh"
#include "OrangeTrackView.hh"

#include "detail/UniverseIndexer.hh"

namespace celeritas
{
namespace
{

struct IdToJson
{
    detail::UniverseIndexer univ_indexer;
    OrangeParams const* orange{nullptr};
    VolumeParams const* volumes{nullptr};

    nlohmann::json operator()(ImplSurfaceId const& id) const
    {
        if (id && orange)
        {
            return orange->surfaces().at(id);
        }
        return id;
    }

    nlohmann::json operator()(ImplVolumeId const& id) const
    {
        if (id && orange)
        {
            return orange->impl_volumes().at(id);
        }
        return id;
    }

    nlohmann::json operator()(VolumeId const& id) const
    {
        if (id && volumes)
        {
            return volumes->volume_labels().at(id);
        }
        return id;
    }

    nlohmann::json operator()(VolumeInstanceId const& id) const
    {
        if (id && orange)
        {
            return volumes->volume_instance_labels().at(id);
        }
        return id;
    }

    nlohmann::json operator()(UniverseId const& id) const
    {
        if (id && orange)
        {
            return orange->universes().at(id);
        }
        return id;
    }

    nlohmann::json operator()(LevelStateAccessor const& lsa) const
    {
        CELER_EXPECT(orange);

        UniverseId u_id = lsa.universe();

        return {
            {"pos", lsa.pos()},
            {"dir", lsa.dir()},
            {"universe", (*this)(u_id)},
            {"volume",
             [&] {
                 auto local_vol = lsa.vol();
                 ImplVolumeId impl_vol;
                 if (u_id && local_vol)
                 {
                     impl_vol = univ_indexer.global_volume(u_id, local_vol);
                 }

                 nlohmann::json result = {
                     {"local", local_vol},
                     {"impl", (*this)(impl_vol)},
                 };
                 if (impl_vol && orange && volumes)
                 {
                     result["canonical"] = (*this)(orange->volume_id(impl_vol));
                     result["instance"]
                         = (*this)(orange->volume_instance_id(impl_vol));
                 }
                 return result;
             }()},
        };
    }
};

}  // namespace

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, OrangeTrackView const& view)
{
    IdToJson id_to_json{view.make_universe_indexer(),
                        view.scalars().host_geo_params,
                        view.scalars().host_volume_params};

    nlohmann::json levels = nlohmann::json::array();
    for (auto lev_id : range(view.level() + 1))
    {
        levels.push_back(id_to_json(view.make_lsa(lev_id)));
    }

    j = {
        {"levels", std::move(levels)},
        {"surface", id_to_json(view.impl_surface_id())},
    };

    if (auto next = view.next_impl_surface_id())
    {
        j["next_surface"] = id_to_json(next);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
