//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTestUtils.cc
//---------------------------------------------------------------------------//
#include "CsgTestUtils.hh"

#include <iomanip>
#include <set>
#include <sstream>
#include <variant>
#include <vector>
#include <nlohmann/json.hpp>

#include "corecel/Assert.hh"
#include "corecel/StringSimplifier.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Repr.hh"
#include "corecel/io/StreamableVariant.hh"
#include "orange/BoundingBoxUtils.hh"
#include "orange/OrangeTypes.hh"
#include "orange/orangeinp/CsgTree.hh"
#include "orange/orangeinp/CsgTreeIO.json.hh"
#include "orange/orangeinp/CsgTreeUtils.hh"
#include "orange/orangeinp/detail/CsgUnit.hh"
#include "orange/orangeinp/detail/IntersectSurfaceState.hh"
#include "orange/surf/SurfaceIO.hh"
#include "orange/transform/TransformIO.hh"

#include "Test.hh"

using namespace celeritas::orangeinp::detail;

namespace celeritas
{
namespace orangeinp
{
namespace test
{

//---------------------------------------------------------------------------//
std::string to_json_string(CsgTree const& tree)
{
    nlohmann::json obj(tree);
    return obj.dump();
}

//---------------------------------------------------------------------------//
std::vector<int> to_vec_int(std::vector<NodeId> const& nodes)
{
    std::vector<int> result;
    for (auto nid : nodes)
    {
        result.push_back(nid ? nid.unchecked_get() : -1);
    }
    return result;
}

//---------------------------------------------------------------------------//
std::vector<std::string> surface_strings(CsgUnit const& u)
{
    // Loop through CSG tree's encountered surfaces
    std::vector<std::string> result;
    for (auto nid : range(NodeId{u.tree.size()}))
    {
        if (auto* surf_node = std::get_if<Surface>(&u.tree[nid]))
        {
            auto lsid = surf_node->id;
            CELER_ASSERT(lsid < u.surfaces.size());
            result.push_back(std::visit(
                [](auto&& surf) {
                    std::ostringstream os;
                    os << std::setprecision(5) << surf;
                    return os.str();
                },
                u.surfaces[lsid.get()]));
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
std::vector<std::string> volume_strings(CsgUnit const& u)
{
    std::vector<std::string> result;

    for (auto const& nid : u.tree.volumes())
    {
        if (nid < u.tree.size())
        {
            result.push_back(build_infix_string(u.tree, nid));
        }
        else
        {
            result.push_back("<INVALID>");
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
std::string tree_string(CsgUnit const& u)
{
    return ::celeritas::orangeinp::test::to_json_string(u.tree);
}

//---------------------------------------------------------------------------//
std::vector<std::string> md_strings(CsgUnit const& u)
{
    std::vector<std::string> result;
    ::celeritas::test::StringSimplifier simplify;
    for (auto const& md_set : u.metadata)
    {
        result.push_back(to_string(
            join_stream(md_set.begin(),
                        md_set.end(),
                        ',',
                        [&simplify](std::ostream& os, Label const& l) {
                            os << simplify(l.name);
                            if (!l.ext.empty())
                            {
                                os << Label::default_sep << l.ext;
                            }
                        })));
    }
    return result;
}

//---------------------------------------------------------------------------//
std::vector<std::string> bound_strings(CsgUnit const& u)
{
    std::vector<std::string> result;
    for (auto&& [node, reg] : u.regions)
    {
        BoundingZone const& bzone = reg.bounds;
        std::ostringstream os;
        os << std::setprecision(3);
        if (bzone.negated)
        {
            os << "~";
        }
        os << node.unchecked_get() << ": {";
        auto print_bb = [&os](BBox const& bb) {
            if (!bb)
            {
                os << "null";
            }
            else if (is_infinite(bb))
            {
                os << "inf";
            }
            else
            {
                os << bb;
            }
        };
        print_bb(bzone.interior);
        os << ", ";
        print_bb(bzone.exterior);
        os << '}';
        result.push_back(std::move(os).str());
    }
    return result;
}

//---------------------------------------------------------------------------//
std::vector<std::string> transform_strings(CsgUnit const& u)
{
    std::vector<std::string> result;
    std::set<TransformId> printed_transform;
    for (auto&& [node, reg] : u.regions)
    {
        std::ostringstream os;
        os << node.unchecked_get() << ": t=";
        if (auto t = reg.trans_id)
        {
            os << t.unchecked_get();
            if (t < u.transforms.size())
            {
                if (printed_transform.insert(t).second)
                {
                    os << " -> " << std::setprecision(3)
                       << StreamableVariant{u.transforms[t.unchecked_get()]};
                }
            }
            else
            {
                os << " -> "
                   << "<INVALID>";
            }
        }
        else
        {
            os << "<MISSING>";
        }

        result.push_back(std::move(os).str());
    }
    return result;
}

//---------------------------------------------------------------------------//
std::vector<int> volume_nodes(CsgUnit const& u)
{
    std::vector<int> result;
    for (auto nid : u.tree.volumes())
    {
        result.push_back(nid ? nid.unchecked_get() : -1);
    }
    return result;
}

//---------------------------------------------------------------------------//
std::vector<std::string> fill_strings(CsgUnit const& u)
{
    std::vector<std::string> result;
    for (auto const& f : u.fills)
    {
        if (std::holds_alternative<std::monostate>(f))
        {
            result.push_back("<UNASSIGNED>");
        }
        else if (auto* mid = std::get_if<GeoMatId>(&f))
        {
            result.push_back("m" + std::to_string(mid->unchecked_get()));
        }
        else if (auto* d = std::get_if<Daughter>(&f))
        {
            std::ostringstream os;
            os << "{u=";
            if (auto u = d->universe_id)
            {
                os << u.unchecked_get();
            }
            else
            {
                os << "<MISSING>";
            }
            os << ", t=";
            if (auto t = d->trans_id)
            {
                os << t.unchecked_get();
            }
            else
            {
                os << "<MISSING>";
            }
            os << '}';
            result.push_back(os.str());
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
std::vector<real_type> flattened(BoundingZone const& bz)
{
    std::vector<real_type> result;
    for (auto const* bb : {&bz.interior, &bz.exterior})
    {
        result.insert(result.end(), bb->lower().begin(), bb->lower().end());
        result.insert(result.end(), bb->upper().begin(), bb->upper().end());
    }
    result.push_back(bz.negated ? -1 : 1);
    return result;
}

//---------------------------------------------------------------------------//
void print_expected(CsgUnit const& u)
{
    std::cout << R"cpp(
/***** EXPECTED UNIT *****/
)cpp"
              << "static char const * const expected_surface_strings[] = "
              << repr(surface_strings(u)) << ";\n"
              << "static char const * const expected_volume_strings[] = "
              << repr(volume_strings(u)) << ";\n"
              << "static char const * const expected_md_strings[] = "
              << repr(md_strings(u)) << ";\n"
              << "static char const * const expected_bound_strings[] = "
              << repr(bound_strings(u)) << ";\n"
              << "static char const * const expected_trans_strings[] = "
              << repr(transform_strings(u)) << ";\n"
              << "static char const * const expected_fill_strings[] = "
              << repr(fill_strings(u)) << ";\n"
              << "static int const expected_volume_nodes[] = "
              << repr(volume_nodes(u)) << ";\n"
              << "static char const expected_tree_string[] = R\"json("
              << tree_string(u) << ")json\";\n"
              << R"cpp(
EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
EXPECT_VEC_EQ(expected_trans_strings, transform_strings(u));
EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
)cpp"
              << "EXPECT_EQ(GeoMatId{";
    if (u.background)
    {
        std::cout << u.background.unchecked_get();
    }
    std::cout << "}, u.background);\n"
              << "/*************************/\n"
              << std::endl;
}

//---------------------------------------------------------------------------//
void print_expected(IntersectSurfaceState const& css)
{
    std::cout << R"cpp(
/***** EXPECTED STATE *****/
// clang-format off
)cpp"
              << "static real_type const expected_local_bz[] = "
              << repr(flattened(css.local_bzone)) << ";\n"
              << "static real_type const expected_global_bz[] = "
              << repr(flattened(css.global_bzone)) << ";\n"
              << "static int const expected_nodes[] = "
              << repr(to_vec_int(css.nodes)) << ";"
              << R"cpp(
// clang-format on

EXPECT_VEC_SOFT_EQ(expected_local_bz, flattened(css.local_bzone));
EXPECT_VEC_SOFT_EQ(expected_global_bz, flattened(css.global_bzone));
EXPECT_VEC_EQ(expected_nodes, to_vec_int(css.nodes));
/*************************/
)cpp" << std::endl;
}

void stream_node_id(std::ostream& os, NodeId n)
{
    os << "N{";
    if (n)
    {
        os << n.unchecked_get();
    }
    os << '}';
}

void stream_logic_int(std::ostream& os, logic_int value)
{
    using namespace logic;
    if (is_operator_token(value))
    {
        os << "logic::";
        switch (static_cast<OperatorToken>(value))
        {
            case lopen:
                os << "lopen";
                return;
            case lclose:
                os << "lclose";
                return;
            case lor:
                os << "lor";
                return;
            case land:
                os << "land";
                return;
            case lnot:
                os << "lnot";
                return;
            case ltrue:
                os << "ltrue";
                return;
            default:
                CELER_ASSERT_UNREACHABLE();
        }
    }
    os << value << 'u';
}

std::ostream& operator<<(std::ostream& os, ReprLogic const& rl)
{
    os << '{'
       << join_stream(
              std::begin(rl.logic), std::end(rl.logic), ", ", stream_logic_int)
       << ",}";
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
