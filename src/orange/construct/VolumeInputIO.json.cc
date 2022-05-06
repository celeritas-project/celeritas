//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/VolumeInputIO.json.cc
//---------------------------------------------------------------------------//
#include "VolumeInputIO.json.hh"

#include <string>
#include <vector>

#include "orange/Types.hh"

#include "VolumeInput.hh"
#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Build a cell from a C string.
 *
 * This used by JSON input processing and will eventually be used in unit
 * tests. A valid string satisfies the regex "[0-9~!| ]+", but the result may
 * not be a valid logic expression. (The volume inserter will ensure that the
 * logic expression at least is consistent for a CSG region definition.)
 *
 * Example:
 * \code

     parse_logic("4 ~ 5 & 6 &");

   \endcode
 */
std::vector<logic_int> parse_logic(const char* c)
{
    std::vector<logic_int> result;
    logic_int              s = 0;
    while (char v = *c++)
    {
        if (v >= '0' && v <= '9')
        {
            // Parse a surface number. 'Push' this digit onto the surface ID by
            // multiplying the existing ID by 10.
            s = 10 * s + (v - '0');

            const char next = *c;
            if (next == ' ' || next == '\0')
            {
                // Next char is end of word or end of string
                result.push_back(s);
                s = 0;
            }
        }
        else
        {
            // Parse a logic token
            switch (v)
            {
                // clang-format off
                case ' ': break;
                case '*': result.push_back(logic::ltrue); break;
                case '|': result.push_back(logic::lor);   break;
                case '&': result.push_back(logic::land);  break;
                case '~': result.push_back(logic::lnot);  break;
                default:  CELER_ASSERT_UNREACHABLE();
                    // clang-format on
            }
        }
    }
    return result;
}
//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * I/O routine for JSON.
 */
void from_json(const nlohmann::json& j, VolumeInput& value)
{
    // Convert faces to OpaqueId
    std::vector<SurfaceId::size_type> temp_faces;
    j.at("faces").get_to(temp_faces);
    value.faces.reserve(temp_faces.size());
    for (auto surfid : temp_faces)
    {
        CELER_ASSERT(surfid != SurfaceId{}.unchecked_get());
        value.faces.emplace_back(surfid);
    }

    // Convert logic string to vector
    auto temp_logic = j.at("logic").get<std::string>();
    value.logic     = parse_logic(temp_logic.c_str());

    // Read scalars, including optional flags
    j.at("num_intersections").get_to(value.max_intersections);
    auto flag_iter = j.find("flags");
    value.flags    = (flag_iter == j.end() ? 0 : flag_iter->get<int>());
}

//---------------------------------------------------------------------------//
} // namespace celeritas
