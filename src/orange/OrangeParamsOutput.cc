//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeParamsOutput.cc
//---------------------------------------------------------------------------//

#include "OrangeParamsOutput.hh"

#include <nlohmann/json.hpp>

#include "corecel/cont/LdgSpan.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/io/LabelIO.json.hh"  // IWYU pragma: keep
#include "corecel/sys/Environment.hh"
#include "geocel/BoundingBoxIO.json.hh"
#include "orange/OrangeTypes.hh"

#include "OrangeData.hh"
#include "OrangeInputIO.json.hh"  // IWYU pragma: keep
#include "OrangeParams.hh"  // IWYU pragma: keep
#include "OrangeTypesIO.json.hh"  // IWYU pragma: keep
#include "univ/TrackerVisitor.hh"

#include "detail/BvhData.hh"
#include "detail/BvhView.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Create JSON representation of the structure of a BVH tree.
 */
nlohmann::json make_bvh_structure_json(
    detail::BvhTreeRecord const& tree,
    NativeCRef<detail::BvhTreeData> const& storage)
{
    using json = nlohmann::json;

    auto out = json::array();

    detail::BvhView view{tree, storage};

    // Handle internal nodes
    for (auto i : range(BvhNodeId{view.num_internal_nodes()}))
    {
        auto const& inner = view.inner_node(i);
        using Side = detail::BvhInternalNode::Side;

        out.push_back({
            "i",
            std::string(1, to_char(inner.axis())),
            {*inner.child(Side::left), *inner.child(Side::right)},
            {inner.bbox(Side::left), inner.bbox(Side::right)},
        });
    }

    // Handle leaf nodes
    for (auto i : range(BvhNodeId{view.num_internal_nodes()},
                        BvhNodeId{view.num_nodes()}))
    {
        auto vols = json::array();
        for (LocalVolumeId id : remove_ldg_wrapper(view.leaf_vol_ids(i)))
        {
            vols.push_back(*id);
        }
        out.push_back({"l", std::move(vols)});
    }

    // Handle inf vols
    auto inf_vols = nlohmann::json::array();
    for (auto id : remove_ldg_wrapper(view.inf_vol_ids()))
    {
        inf_vols.push_back(id.unchecked_get());
    }

    return json::object({
        {"tree", std::move(out)},
        {"inf_vol_ids", std::move(inf_vols)},
    });
}

//---------------------------------------------------------------------------//
/*!
 * Create JSON representation of the daughter universes embedded in a universe.
 *
 * Each daughter consists of a local volume ID, daughter universe ID, and
 * transform type.
 */
nlohmann::json make_univ_structure_json(HostCRef<OrangeParamsData> const& data,
                                        UnivId uid)
{
    using json = nlohmann::json;

    auto daughters = json::array();

    TrackerVisitor visit_tracker{data};
    visit_tracker(
        [&](auto const& tracker) {
            for (auto vol_id : range(LocalVolumeId{tracker.num_volumes()}))
            {
                DaughterId daughter_id = tracker.daughter(vol_id);
                if (!daughter_id)
                {
                    continue;
                }
                auto const& daughter = data.daughters[daughter_id];
                auto const& transform = data.transforms[daughter.trans_id];
                daughters.push_back({
                    *vol_id,
                    *daughter.univ_id,
                    to_cstring(transform.type),
                });
            }
        },
        uid);

    return json::object({
        {"daughters", std::move(daughters)},
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
        OPO_PAIR(data.scalars, tol),
    };
    if (orange_tracking_logic != LogicNotation::infix)
    {
        // The max_csg_levels field is only set when pre/postfix, even though
        // it always exists as member data in the scalars
        obj["scalars"]["max_csg_levels"] = data.scalars.max_csg_levels;
    }

    // Save sizes
    obj["sizes"] = {
        OPO_SIZE_PAIR(data, connectivity_records),
        OPO_SIZE_PAIR(data, daughters),
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
    obj["sizes"]["bvh"] = [&bvhdata = data.bvh_tree_data] {
        return json::object({
            OPO_SIZE_PAIR(bvhdata, bboxes),
            OPO_SIZE_PAIR(bvhdata, internal_nodes),
            OPO_SIZE_PAIR(bvhdata, leaf_nodes),
            OPO_SIZE_PAIR(bvhdata, local_volume_ids),
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

    // Write BVH metadata as a struct of arrays
    {
        obj["bvh_metadata"] = json::object();
        auto& bvh_metadata = obj["bvh_metadata"];

        auto insert_array = [&bvh_metadata](std::string key) -> json& {
            bvh_metadata[key] = json::array();
            return bvh_metadata[key];
        };

        auto& finite = insert_array("num_finite_bboxes");
        auto& infinite = insert_array("num_infinite_bboxes");
        auto& depth = insert_array("depth");

        for (auto i : range(data.simple_units.size()))
        {
            auto const& unit = data.simple_units[SimpleUnitId{i}];
            auto const& mdi = unit.bvh_tree.metadata;
            finite.push_back(mdi.num_finite_bboxes);
            infinite.push_back(mdi.num_infinite_bboxes);
            depth.push_back(mdi.depth);
        }

        // Include structure information if requested by the user
        if (celeritas::getenv_flag("ORANGE_BVH_STRUCTURE", false).value)
        {
            auto& structure = insert_array("structure");
            for (auto i : range(data.simple_units.size()))
            {
                auto const& unit = data.simple_units[SimpleUnitId{i}];
                structure.push_back(make_bvh_structure_json(
                    unit.bvh_tree, data.bvh_tree_data));
            }
        }
    }

    // Write universe metadata as a struct of arrays
    {
        obj["univ_metadata"] = json::object();
        auto& univ_metadata = obj["univ_metadata"];

        auto insert_array = [&univ_metadata](std::string key) -> json& {
            univ_metadata[key] = json::array();
            return univ_metadata[key];
        };

        auto& type = insert_array("type");
        auto& label = insert_array("label");
        auto& num_surfaces = insert_array("num_surfaces");
        auto& num_volumes = insert_array("num_volumes");

        // Calculate number of surface/volume using universe indexer offsets
        using SizeId = OpaqueId<size_type>;
        auto const& uidata = data.univ_indexer_data;
        auto calc_local_size = [](auto const& offsets, UnivId uid) {
            CELER_EXPECT(uid && uid.unchecked_get() + 1 < offsets.size());
            return offsets[SizeId{uid.unchecked_get() + 1}]
                   - offsets[SizeId{uid.unchecked_get()}];
        };

        for (auto uid : range(UnivId{data.univ_types.size()}))
        {
            type.push_back(to_cstring(data.univ_types[uid]));
            label.push_back(orange_->universes().at(uid));
            num_surfaces.push_back(calc_local_size(uidata.surfaces, uid));
            num_volumes.push_back(calc_local_size(uidata.volumes, uid));
        }

        // Include structure information if requested by the user
        if (celeritas::getenv_flag("ORANGE_UNIV_STRUCTURE", false).value)
        {
            auto& structure = insert_array("structure");
            for (auto uid : range(UnivId{data.univ_types.size()}))
            {
                structure.push_back(make_univ_structure_json(data, uid));
            }
        }
    }

    j->obj = std::move(obj);
}

//---------------------------------------------------------------------------//
/*!
 * Print a BVH structure to a JSON string for debugging.
 */
std::string dump_bvh_structure(detail::BvhTreeRecord const& tree,
                               NativeCRef<detail::BvhTreeData> const& data)
{
    return make_bvh_structure_json(tree, data).dump();
}

//---------------------------------------------------------------------------//
/*!
 * Print the universe structure to a JSON string for debugging.
 *
 * The result is an array, indexed by universe ID, of the daughters embedded
 * in each universe.
 */
std::string dump_univ_structure(HostCRef<OrangeParamsData> const& data)
{
    auto result = nlohmann::json::array();
    for (auto uid : range(UnivId{data.univ_types.size()}))
    {
        result.push_back(make_univ_structure_json(data, uid));
    }
    return result.dump();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
