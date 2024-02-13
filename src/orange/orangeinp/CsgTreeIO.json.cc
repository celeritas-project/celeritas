//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTreeIO.json.cc
//---------------------------------------------------------------------------//
#include "CsgTreeIO.json.hh"

#include "corecel/cont/Range.hh"
#include "corecel/cont/VariantUtils.hh"

#include "CsgTree.hh"

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

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
