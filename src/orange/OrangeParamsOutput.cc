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
#include "corecel/sys/Environment.hh"
#include "orange/OrangeTypes.hh"

#include "OrangeInputIO.json.hh"  // IWYU pragma: keep
#include "OrangeParams.hh"  // IWYU pragma: keep
#include "OrangeTypesIO.json.hh"  // IWYU pragma: keep

#include "detail/BIHData.hh"
#include "detail/BIHView.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Create JSON representation of the structure of a BIH tree.
 */
nlohmann::json make_bih_structure_json(detail::BIHTreeRecord const& tree,
                                       NativeCRef<BIHTreeData> const& storage)
{
    using json = nlohmann::json;

    auto out = json::array();

    detail::BIHView view{tree, storage};

    // Handle inner nodes
    for (auto i : range(tree.inner_nodes.size()))
    {
        auto const& inner = view.inner_node(BIHNodeId{i});
        auto const& left = inner.edges[detail::BIHInnerNode::Side::left];
        auto const& right = inner.edges[detail::BIHInnerNode::Side::right];

        out.push_back(json::array({"i",
                                   std::string(1, to_char(inner.axis)),
                                   json::array({left.child.unchecked_get(),
                                                right.child.unchecked_get()}),
                                   json::array({left.bounding_plane_pos,
                                                right.bounding_plane_pos})}));
    }

    // Handle leaf nodes
    size_type offset = tree.inner_nodes.size();
    for (auto i : range(tree.leaf_nodes.size()))
    {
        auto const& leaf = view.leaf_node(BIHNodeId{offset + i});
        auto vols = json::array();
        for (auto id : view.leaf_vol_ids(leaf))
        {
            vols.push_back(id.unchecked_get());
        }
        out.push_back(json::array({"l", std::move(vols)}));
    }

    // Handle inf vols
    auto inf_vols = nlohmann::json::array();
    for (auto id : view.inf_vol_ids())
    {
        inf_vols.push_back(id.unchecked_get());
    }

    return json::object({
        {"tree", std::move(out)},
        {"inf_vol_ids", std::move(inf_vols)},
    });
}

//---------------------------------------------------------------------------//
}  // namespace

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
            auto const& unit = data.simple_units[SimpleUnitId{i}];
            auto const& mdi = unit.bih_tree.metadata;
            finite.push_back(mdi.num_finite_bboxes);
            infinite.push_back(mdi.num_infinite_bboxes);
            depth.push_back(mdi.depth);
        }

        // Include structure information if requested by the user
        if (celeritas::getenv_flag("ORANGE_BIH_STRUCTURE", false).value)
        {
            auto& structure = make_array("structure");
            for (auto i : range(data.simple_units.size()))
            {
                auto const& unit = data.simple_units[SimpleUnitId{i}];
                structure.push_back(make_bih_structure_json(
                    unit.bih_tree, data.bih_tree_data));
            }
        }
    }

    j->obj = std::move(obj);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
