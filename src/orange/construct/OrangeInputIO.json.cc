//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/OrangeInputIO.json.cc
//---------------------------------------------------------------------------//
#include "OrangeInputIO.json.hh"

#include <algorithm>
#include <vector>

#include "corecel/cont/Array.json.hh"
#include "corecel/cont/Label.json.hh"
#include "corecel/cont/Range.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Build a vector of strings for each surface type.
 */
std::vector<std::string> make_surface_strings()
{
    std::vector<std::string> result(static_cast<size_type>(SurfaceType::size_));
    for (auto surf_type : range(SurfaceType::size_))
    {
        result[static_cast<size_type>(surf_type)] = to_cstring(surf_type);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a surface type string to an enum for I/O.
 */
SurfaceType to_surface_type(const std::string& s)
{
    // The number of surface types will be short, and presumably each string is
    // small enough to fit inside string's static allocation. Therefore the
    // string search will be on a small-ish, nearly contiguous block of memory,
    // so it's preferable than using unordered_map or a more heavyweight
    // container.
    static const auto surface_string = make_surface_strings();

    auto iter = std::find(surface_string.begin(), surface_string.end(), s);
    CELER_VALIDATE(iter != surface_string.end(),
                   << "invalid surface string '" << s << "'");

    unsigned int result_int = iter - surface_string.begin();
    CELER_EXPECT(result_int < static_cast<size_type>(SurfaceType::size_));
    return static_cast<SurfaceType>(result_int);
}

//---------------------------------------------------------------------------//
/*!
 * Build a volume from a C string.
 *
 * A valid string satisfies the regex "[0-9~!| ]+", but the result may
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

} // namespace

//---------------------------------------------------------------------------//
/*!
 * Read surface data from an ORANGE JSON file.
 */
void from_json(const nlohmann::json& j, SurfaceInput& value)
{
    // Read and convert types
    const auto& type_labels = j.at("types").get<std::vector<std::string>>();
    value.types.resize(type_labels.size());
    std::transform(type_labels.begin(),
                   type_labels.end(),
                   value.types.begin(),
                   &to_surface_type);

    j.at("data").get_to(value.data);
    j.at("sizes").get_to(value.sizes);
}

//---------------------------------------------------------------------------//
/*!
 * Read cell/volume data from an ORANGE JSON file.
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
    const auto& temp_logic = j.at("logic").get<std::string>();
    value.logic            = parse_logic(temp_logic.c_str());

    // Parse bbox
    if (j.contains("bbox"))
    {
        auto bbox  = j.at("bbox").get<Array<Real3, 2>>();
        value.bbox = {bbox[0], bbox[1]};
    }

    // Read scalars, including optional flags
    auto flag_iter = j.find("flags");
    value.flags    = (flag_iter == j.end() ? 0 : flag_iter->get<int>());
    j.at("zorder").get_to(value.zorder);
}

//---------------------------------------------------------------------------//
/*!
 * Read a unit definition from an ORANGE input file.
 */
void from_json(const nlohmann::json& j, UnitInput& value)
{
    using VecLabel = std::vector<Label>;
    j.at("surfaces").get_to(value.surfaces);
    j.at("cells").get_to(value.volumes);
    j.at("md").at("name").get_to(value.label);

    {
        // Move labels into lower-level data structures
        auto labels = j.at("cell_names").get<VecLabel>();
        CELER_VALIDATE(labels.size() == value.volumes.size(),
                       << "incorrect size for volume labels");
        for (auto i : range(labels.size()))
        {
            value.volumes[i].label = std::move(labels[i]);
        }

        j.at("surface_names").get_to(value.surfaces.labels);
    }
    {
        const auto& bbox = j.at("bbox");
        value.bbox       = {bbox.at(0).get<Real3>(), bbox.at(1).get<Real3>()};
    }
}

//---------------------------------------------------------------------------//
/*!
 * Read a partially preprocessed geometry definition from an ORANGE JSON file.
 */
void from_json(const nlohmann::json& j, OrangeInput& value)
{
    const auto& universes = j.at("universes");

    value.units.reserve(universes.size());
    for (const auto& uni : universes)
    {
        CELER_VALIDATE(uni.at("_type").get<std::string>() == "simple unit",
                       << "unsupported universe type '" << uni["_type"] << "'");
        value.units.push_back(uni.get<UnitInput>());
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
