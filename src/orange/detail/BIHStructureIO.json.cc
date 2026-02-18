//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHStructureIO.json.cc
//---------------------------------------------------------------------------//
#include "BIHStructureIO.json.hh"

#include <string>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, BIHStructure::Inner const& value)
{
    auto children = nlohmann::json::array();
    for (auto child_id : value.children)
    {
        children.push_back(child_id.unchecked_get());
    }

    auto bounding_plane_pos = nlohmann::json::array();
    for (auto bp_pos : value.bounding_plane_pos)
    {
        bounding_plane_pos.push_back(bp_pos);
    }

    j = nlohmann::json::array({"i",
                               std::string(1, to_char(value.axis)),
                               std::move(children),
                               std::move(bounding_plane_pos)});
}

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, BIHStructure::Leaf const& value)
{
    auto vol_ids = nlohmann::json::array();
    for (auto vol_id : value.vol_ids)
    {
        vol_ids.push_back(vol_id.unchecked_get());
    }
    j = nlohmann::json::array({"l", std::move(vol_ids)});
}

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, BIHStructure::Node const& value)
{
    std::visit([&j](auto const& v) { to_json(j, v); }, value);
}

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, BIHStructure const& value)
{
    auto tree = nlohmann::json::array();
    for (auto const& node : value.tree())
    {
        tree.push_back(node);
    }

    auto inf_vol_ids = nlohmann::json::array();
    for (auto vol_id : value.inf_vol_ids())
    {
        inf_vol_ids.push_back(vol_id.unchecked_get());
    }

    j = nlohmann::json::object({
        {"tree", std::move(tree)},
        {"inf_vol_ids", std::move(inf_vol_ids)},
    });
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
