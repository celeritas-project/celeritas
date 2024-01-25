//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/CsgTestUtils.cc
//---------------------------------------------------------------------------//
#include "CsgTestUtils.hh"

#include <sstream>
#include <variant>

#include "celeritas_config.h"
#include "corecel/io/Join.hh"
#include "corecel/io/Repr.hh"
#include "orange/construct/CsgTree.hh"
#include "orange/orangeinp/detail/CsgUnit.hh"
#include "orange/surf/SurfaceIO.hh"
#include "orange/transform/TransformIO.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>

#    include "orange/construct/CsgTreeIO.json.hh"
#endif

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
std::string to_json_string(CsgTree const& tree)
{
#if CELERITAS_USE_JSON
    nlohmann::json obj{tree};
    return obj.dump();
#else
    CELER_DISCARD(tree);
    return {};
#endif
}

//---------------------------------------------------------------------------//
}  // namespace test

namespace orangeinp
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
std::vector<std::string> surface_strings(CsgUnit const& u)
{
    std::vector<std::string> result;

    for (auto const& vs : u.surfaces)
    {
        result.push_back(std::visit(
            [](auto&& surf) {
                std::ostringstream os;
                os << surf;
                return os.str();
            },
            vs));
    }
    return result;
}

//---------------------------------------------------------------------------//
std::string tree_string(CsgUnit const& u)
{
    return ::celeritas::test::to_json_string(u.tree);
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
std::vector<real_type> flattened_bboxes(CsgUnit const& u)
{
    std::vector<real_type> result;
    for (auto const& bbox : u.bboxes)
    {
        result.insert(result.end(), bbox.lower().begin(), bbox.lower().end());
        result.insert(result.end(), bbox.upper().begin(), bbox.upper().end());
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
            os << '{';
            if (auto u = d->universe_id)
            {
                os << u.unchecked_get();
            }
            else
            {
                os << "<MISSING>";
            }
            os << ',';
            if (auto t = d->transform_id)
            {
                if (t < u.transforms.size())
                {
                    std::visit([&os](auto&& transform) { os << transform; },
                               u.transforms[t.unchecked_get()]);
                }
                else
                {
                    os << "<INVALID: " << t.unchecked_get() << ">";
                }
            }
            else
            {
                os << "<MISSING>";
            }

            result.push_back(os.str());
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
void print_expected(CsgUnit const& u)
{
    std::cout << "/***** EXPECTED UNIT *****/\n"
              << "static char const * const expected_surface_strings[] = "
              << repr(surface_strings(u)) << ";\n"
              << "EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));"
              << R"cpp(
if (CELERITAS_USE_JSON)
{
    EXPECT_JSON_EQ(
        R"json()cpp"
              << tree_string(u) << R"cpp()json",
        tree_string(u));
}
)cpp"
              << "static char const * const expected_md_strings[] = "
              << repr(md_strings(u)) << ";\n"
              << "EXPECT_VEC_EQ(expected_md_strings, md_strings(u));\n"
              << "static real_type const expected_flattened_bboxes[] = "
              << repr(flattened_bboxes(u)) << ";\n"
              << "EXPECT_VEC_SOFT_EQ(expected_flattened_bboxes, "
                 "flattened_bboxes(u));\n"
              << "static int const expected_volume_nodes[] = "
              << repr(volume_nodes(u)) << ";\n"
              << "EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));\n"
              << "static char const * const expected_fill_strings[] = "
              << repr(fill_strings(u)) << ";\n"
              << "EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));\n"
              << "EXPECT_EQ(NodeId{";
    if (u.exterior)
    {
        std::cout << u.exterior.unchecked_get();
    }
    std::cout << "}, u.exterior);\n"
              << "/*************************/\n"
              << std::endl;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
