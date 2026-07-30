//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTreeIO.json.cc
//---------------------------------------------------------------------------//
#include "CsgTreeIO.json.hh"

#include <string>

#include "corecel/cont/Range.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/io/LabelIO.json.hh"
#include "corecel/io/StreamableVariant.hh"
#include "geocel/BoundingBoxIO.json.hh"
#include "orange/surf/SurfaceIO.hh"
#include "orange/transform/TransformIO.hh"

#include "CsgTree.hh"

#include "detail/CsgUnit.hh"

using nlohmann::json;

namespace celeritas
{
namespace orangeinp
{
namespace
{
//---------------------------------------------------------------------------//
struct JsonConverter
{
    json operator()(orangeinp::True const&) const { return "t"; }
    json operator()(orangeinp::False const&) const { return "f"; }
    json operator()(orangeinp::Aliased const& a) const
    {
        return json::array({"=", a.node.unchecked_get()});
    }
    json operator()(orangeinp::Negated const& n) const
    {
        return json::array({"~", n.node.unchecked_get()});
    }
    json operator()(orangeinp::Surface const& s) const
    {
        return json::array({"S", s.id.unchecked_get()});
    }
    json operator()(orangeinp::Joined const& j) const
    {
        auto nodes = json::array();
        for (auto n : j.nodes)
        {
            nodes.push_back(n.unchecked_get());
        }
        return json::array({std::string{to_char(j.op)}, std::move(nodes)});
    }
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
void to_json(json& j, CsgTree const& tree)
{
    ContainerVisitor<CsgTree const&, CsgTree::NodeId> visit_tree{tree};

    j = json::array();
    for (auto n : range(CsgTree::NodeId{tree.size()}))
    {
        j.push_back(visit_tree(JsonConverter{}, n));
    }
}

namespace detail
{
//---------------------------------------------------------------------------//
//! Write CSG unit to JSON
void to_json(nlohmann::json& j, CsgUnit const& unit)
{
    using nlohmann::json;
    j = json::object();

    j["surfaces"] = [&all_s = unit.surfaces] {
        json result = json::array();
        for (auto const& s : all_s)
        {
            result.push_back(to_string(StreamableVariant{s}));
        }
        return result;
    }();

    j["tree"] = unit.tree;

    j["metadata"] = [&all_md = unit.metadata] {
        json result = json::array();
        for (auto const& md : all_md)
        {
            json entry = json::array();
            for (auto const& label : md)
            {
                entry.push_back(label);
            }
            result.push_back(std::move(entry));
        }
        return result;
    }();

    j["regions"] = [&all_regions = unit.regions] {
        json result = json::object();
        for (auto&& [node, reg] : all_regions)
        {
            auto entry = nlohmann::json{
                {"exterior", reg.bounds.exterior},
                {"transform", reg.trans_id.unchecked_get()},
            };
            if (reg.bounds.interior)
            {
                entry["interior"] = reg.bounds.interior;
            }
            if (reg.bounds.negated)
            {
                entry["negated"] = true;
            }

            result[std::to_string(node.unchecked_get())] = std::move(entry);
        }
        return result;
    }();

    j["volumes"] = [&unit] {
        CELER_ASSERT(unit.tree.volumes().size() == unit.fills.size());
        json result = json::array();
        for (auto i : range(unit.tree.volumes().size()))
        {
            auto entry
                = nlohmann::json{{"csg_node", unit.tree.volumes()[i].get()}};
            if (auto* m = std::get_if<GeoMatId>(&unit.fills[i]))
            {
                entry["material"] = m->unchecked_get();
            }
            else if (auto* d = std::get_if<Daughter>(&unit.fills[i]))
            {
                entry["universe"] = d->univ_id.unchecked_get();
                entry["transform"] = d->trans_id.unchecked_get();
            }
            result.push_back(entry);
        }
        return result;
    }();

    j["fills"] = [&volumes = unit.tree.volumes()] {
        json result = json::array();
        for (auto const& v : volumes)
        {
            result.push_back(v.unchecked_get());
        }
        return result;
    }();

    if (unit.background)
    {
        j["background"] = unit.background.get();
    }

    j["transforms"] = [&all_t = unit.transforms] {
        json result = json::array();
        for (auto const& t : all_t)
        {
            result.push_back(to_string(StreamableVariant{t}));
        }
        return result;
    }();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
