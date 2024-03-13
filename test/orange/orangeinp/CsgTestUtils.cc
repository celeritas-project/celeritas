//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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

#include "celeritas_config.h"
#include "corecel/io/Join.hh"
#include "corecel/io/Repr.hh"
#include "corecel/io/StreamableVariant.hh"
#include "orange/BoundingBoxUtils.hh"
#include "orange/orangeinp/CsgTree.hh"
#include "orange/orangeinp/CsgTreeUtils.hh"
#include "orange/orangeinp/detail/ConvexSurfaceState.hh"
#include "orange/orangeinp/detail/CsgUnit.hh"
#include "orange/surf/SurfaceIO.hh"
#include "orange/transform/TransformIO.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>

#    include "orange/orangeinp/CsgTreeIO.json.hh"
#endif

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
#if CELERITAS_USE_JSON
    nlohmann::json obj(tree);
    return obj.dump();
#else
    CELER_DISCARD(tree);
    return {};
#endif
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
    std::vector<std::string> result;

    for (auto const& vs : u.surfaces)
    {
        result.push_back(std::visit(
            [](auto&& surf) {
                std::ostringstream os;
                os << std::setprecision(5) << surf;
                return os.str();
            },
            vs));
    }
    return result;
}

//---------------------------------------------------------------------------//
std::vector<std::string> volume_strings(CsgUnit const& u)
{
    std::vector<std::string> result;

    for (auto const& nid : u.volumes)
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
    for (auto const& md_set : u.metadata)
    {
        result.push_back(to_string(join(md_set.begin(), md_set.end(), ',')));
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
        if (auto t = reg.transform_id)
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
    for (auto nid : u.volumes)
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
        else if (auto* mid = std::get_if<MaterialId>(&f))
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
            if (auto t = d->transform_id)
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
if (CELERITAS_USE_JSON)
{
    EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
}
)cpp"
              << "EXPECT_EQ(MaterialId{";
    if (u.background)
    {
        std::cout << u.background.unchecked_get();
    }
    std::cout << "}, u.background);\n"
              << "/*************************/\n"
              << std::endl;
}

//---------------------------------------------------------------------------//
void print_expected(ConvexSurfaceState const& css)
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

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
