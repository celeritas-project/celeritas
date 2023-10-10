//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/OrangeInputIO.json.cc
//---------------------------------------------------------------------------//
#include "OrangeInputIO.json.hh"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/io/Label.hh"
#include "corecel/io/LabelIO.json.hh"
#include "orange/BoundingBoxIO.json.hh"
#include "orange/OrangeTypes.hh"
#include "orange/construct/OrangeInput.hh"

#include "detail/OrangeInputIOImpl.json.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Get the i'th slice of a span of data.
 */
template<size_type N, class T>
decltype(auto) slice(Span<T> data, size_type i)
{
    CELER_ASSERT(N * (i + 1) <= data.size());
    Array<std::remove_const_t<T>, N> result;
    std::copy_n(data.data() + i * N, N, result.begin());
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Read surface data from an ORANGE JSON file.
 */
void from_json(nlohmann::json const& j, SurfaceInput& value)
{
    // Read and convert types
    auto const& type_labels = j.at("types").get<std::vector<std::string>>();
    value.types.resize(type_labels.size());
    std::transform(type_labels.begin(),
                   type_labels.end(),
                   value.types.begin(),
                   &detail::to_surface_type);

    j.at("data").get_to(value.data);
    j.at("sizes").get_to(value.sizes);
}

//---------------------------------------------------------------------------//
/*!
 * Read cell/volume data from an ORANGE JSON file.
 */
void from_json(nlohmann::json const& j, VolumeInput& value)
{
    // Convert faces to OpaqueId
    std::vector<LocalSurfaceId::size_type> temp_faces;
    j.at("faces").get_to(temp_faces);
    value.faces.reserve(temp_faces.size());
    for (auto surfid : temp_faces)
    {
        CELER_ASSERT(surfid != LocalSurfaceId{}.unchecked_get());
        value.faces.emplace_back(surfid);
    }

    // Convert logic string to vector
    auto const& temp_logic = j.at("logic").get<std::string>();
    value.logic = detail::parse_logic(temp_logic.c_str());

    // Parse bbox
    if (j.contains("bbox"))
    {
        j.at("bbox").get_to(value.bbox);
    }
    else
    {
        value.bbox = BBox::from_infinite();
    }

    // Read scalars, including optional flags
    auto flag_iter = j.find("flags");
    value.flags = (flag_iter == j.end() ? 0 : flag_iter->get<int>());
    j.at("zorder").get_to(value.zorder);
}

//---------------------------------------------------------------------------//
/*!
 * Read a unit definition from an ORANGE input file.
 */
void from_json(nlohmann::json const& j, UnitInput& value)
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
    if (j.contains("bbox"))
    {
        j.at("bbox").get_to(value.bbox);
    }
    else
    {
        value.bbox = BBox::from_infinite();
    }

    if (j.contains("parent_cells"))
    {
        auto const& parent_cells
            = j.at("parent_cells").get<std::vector<size_type>>();

        auto const& daughters = j.at("daughters").get<std::vector<size_type>>();
        CELER_VALIDATE(parent_cells.size() == daughters.size(),
                       << "fields 'parent_cells' and 'daughters' have "
                          "different lengths");

        auto const& translations
            = j.at("translations").get<std::vector<real_type>>();
        CELER_VALIDATE(3 * parent_cells.size() == translations.size(),
                       << "field 'translations' is not 3x length of "
                          "'parent_cells'");

        UnitInput::MapVolumeDaughter daughter_map;
        for (auto i : range(parent_cells.size()))
        {
            daughter_map[LocalVolumeId{parent_cells[i]}] = {
                UniverseId{daughters[i]}, slice<3>(make_span(translations), i)};
        }

        value.daughter_map = std::move(daughter_map);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Read a rectangular array universe definition from an ORANGE input file.
 */
void from_json(nlohmann::json const& j, RectArrayInput& value)
{
    j.at("md").at("name").get_to(value.label);

    for (auto ax : range(Axis::size_))
    {
        value.grid[to_int(ax)]
            = j.at(std::string(1, to_char(ax))).get<std::vector<real_type>>();
        CELER_VALIDATE(value.grid[to_int(ax)].size() >= 2,
                       << "axis " << to_char(ax)
                       << " does must have at least two grid points");
    }

    // Read daughters universes/translations
    {
        auto parents = j.at("parent_cells").get<std::vector<size_type>>();
        auto daughters = j.at("daughters").get<std::vector<size_type>>();
        auto translations = j.at("translations").get<std::vector<real_type>>();

        CELER_VALIDATE(3 * daughters.size() == translations.size(),
                       << "field 'daughters' is not 3x length of "
                          "'parent_cells'");

        value.daughters.resize(daughters.size());

        for (auto i : range(daughters.size()))
        {
            value.daughters[parents[i]] = {
                UniverseId{daughters[i]}, slice<3>(make_span(translations), i)};
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Read tolerances.
 */
void from_json(nlohmann::json const& j, Tolerance<>& value)
{
    j.at("rel").get_to(value.rel);
    CELER_VALIDATE(value.rel > 0 && value.rel < 1,
                   << "tolerance " << value.rel
                   << " is out of range [must be in (0,1)]");

    j.at("abs").get_to(value.abs);
    CELER_VALIDATE(value.abs > 0,
                   << "tolerance " << value.abs
                   << " is out of range [must be greater than zero]");
}

//---------------------------------------------------------------------------//
/*!
 * Read a partially preprocessed geometry definition from an ORANGE JSON file.
 */
void from_json(nlohmann::json const& j, OrangeInput& value)
{
    auto const& universes = j.at("universes");

    value.universes.reserve(universes.size());

    for (auto const& uni : universes)
    {
        auto const& uni_type = uni.at("_type").get<std::string>();
        if (uni_type == "simple unit")
        {
            value.universes.push_back(uni.get<UnitInput>());
        }
        else if (uni_type == "rectangular array")
        {
            value.universes.push_back(uni.get<RectArrayInput>());
        }
        else if (uni_type == "hexagonal array")
        {
            CELER_NOT_IMPLEMENTED("hexagonal array universes");
        }
        else
        {
            CELER_VALIDATE(
                false, << "unsupported universe type '" << uni_type << "'");
        }
    }

    if (j.count("tol"))
    {
        j.at("tol").get_to(value.tol);
    }
}

//---------------------------------------------------------------------------//
// WRITERS
//---------------------------------------------------------------------------//
/*!
 * Write tolerances.
 */
template<class T>
void to_json(nlohmann::json& j, Tolerance<T> const& value)
{
    j = {
        {"rel", value.rel},
        {"abs", value.abs},
    };
}

template void to_json(nlohmann::json&, Tolerance<real_type> const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
