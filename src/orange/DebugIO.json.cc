//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/DebugIO.json.cc
//---------------------------------------------------------------------------//
#include "DebugIO.json.hh"

#include "corecel/io/JsonUtils.json.hh"  // IWYU pragma: keep
#include "corecel/io/LabelIO.json.hh"  // IWYU pragma: keep
#include "corecel/math/QuantityIO.json.hh"  // IWYU pragma: keep
#include "geocel/VolumeParams.hh"

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

    nlohmann::json operator()(UnivId const& id) const
    {
        if (id && orange)
        {
            return orange->universes().at(id);
        }
        return id;
    }

    nlohmann::json operator()(OrangeTrackView const& view) const
    {
        CELER_EXPECT(orange);

        ImplVolumeId impl_vol = view.impl_volume_id();
        auto local = univ_indexer.local_volume(impl_vol);

        return {
            {"pos", view.pos()},
            {"dir", view.dir()},
            {"universe", (*this)(local.univ)},
            {"volume",
             [&] {
                 auto local_vol = local.volume;

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
    IdToJson id_to_json{view.make_univ_indexer(),
                        view.scalars().host_geo_params,
                        view.scalars().host_volume_params};

    nlohmann::json levels = nlohmann::json::array();
    levels.push_back(id_to_json(view));

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
