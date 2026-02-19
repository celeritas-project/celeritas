//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeParamsOutput.cc
//---------------------------------------------------------------------------//
#include "OrangeParamsOutput.hh"

#include <nlohmann/json.hpp>

#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"
#include "orange/OrangeTypes.hh"

#include "OrangeInputIO.json.hh"  // IWYU pragma: keep
#include "OrangeParams.hh"  // IWYU pragma: keep
#include "OrangeTypesIO.json.hh"  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from shared orange data.
 */
OrangeParamsOutput::OrangeParamsOutput(SPConstOrangeParams orange)
    : orange_(std::move(orange))
{
    CELER_EXPECT(orange_);
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void OrangeParamsOutput::output(JsonPimpl* j) const
{
    using json = nlohmann::json;

    auto obj = json{
        {"tracking_logic", orange_tracking_logic},
    };
    auto const& data = orange_->host_ref();

#define OPO_PAIR(DATA, NAME) {#NAME, DATA.NAME}
#define OPO_SIZE_PAIR(DATA, NAME) {#NAME, DATA.NAME.size()}

    // Save param scalars
    obj["scalars"] = {
        OPO_PAIR(data.scalars, num_univ_levels),
        OPO_PAIR(data.scalars, max_faces),
        OPO_PAIR(data.scalars, max_intersections),
        OPO_PAIR(data.scalars, max_csg_levels),
        OPO_PAIR(data.scalars, tol),
    };

    // Save sizes
    obj["sizes"] = {
        OPO_SIZE_PAIR(data, connectivity_records),
        OPO_SIZE_PAIR(data, daughters),
        OPO_SIZE_PAIR(data, fast_real3s),
        OPO_SIZE_PAIR(data, local_surface_ids),
        OPO_SIZE_PAIR(data, local_volume_ids),
        OPO_SIZE_PAIR(data, logic_ints),
        OPO_SIZE_PAIR(data, obz_records),
        OPO_SIZE_PAIR(data, real_ids),
        OPO_SIZE_PAIR(data, reals),
        OPO_SIZE_PAIR(data, rect_arrays),
        OPO_SIZE_PAIR(data, simple_units),
        OPO_SIZE_PAIR(data, surface_types),
        OPO_SIZE_PAIR(data, transforms),
        OPO_SIZE_PAIR(data, univ_indices),
        OPO_SIZE_PAIR(data, univ_types),
        OPO_SIZE_PAIR(data, volume_ids),
        OPO_SIZE_PAIR(data, volume_instance_ids),
        OPO_SIZE_PAIR(data, volume_records),
    };
    obj["sizes"]["bih"] = [&bihdata = data.bih_tree_data] {
        return json::object({
            OPO_SIZE_PAIR(bihdata, bboxes),
            OPO_SIZE_PAIR(bihdata, inner_nodes),
            OPO_SIZE_PAIR(bihdata, leaf_nodes),
            OPO_SIZE_PAIR(bihdata, local_volume_ids),
        });
    }();
    obj["sizes"]["universe_indexer"] = [&uidata = data.univ_indexer_data] {
        auto ui = json::object({
            OPO_SIZE_PAIR(uidata, surfaces),
            OPO_SIZE_PAIR(uidata, volumes),
        });
        return ui;
    }();

#undef OPO_SIZE_PAIR
#undef OPO_PAIR

    // Write BIH metadata as a struct of arrays
    {
        obj["bih_metadata"] = json::object();
        auto& bih_metadata = obj["bih_metadata"];

        auto make_array = [&](std::string key) -> json& {
            bih_metadata[key] = json::array();
            return bih_metadata[key];
        };

        auto& finite = make_array("num_finite_bboxes");
        auto& infinite = make_array("num_infinite_bboxes");
        auto& depth = make_array("depth");

        for (auto i : range(data.simple_units.size()))
        {
            auto const& mdi
                = data.simple_units[SimpleUnitId{i}].bih_tree.metadata;
            finite.push_back(mdi.num_finite_bboxes);
            infinite.push_back(mdi.num_infinite_bboxes);
            depth.push_back(mdi.depth);
        }
    }

    j->obj = std::move(obj);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
