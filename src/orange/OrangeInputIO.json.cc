//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeInputIO.json.cc
//---------------------------------------------------------------------------//
#include "OrangeInputIO.json.hh"

#include <algorithm>
#include <initializer_list>
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
#include "corecel/io/JsonUtils.json.hh"
#include "corecel/io/Label.hh"
#include "corecel/io/LabelIO.json.hh"
#include "corecel/io/Logger.hh"
#include "geocel/BoundingBoxIO.json.hh"

#include "OrangeInput.hh"
#include "OrangeTypes.hh"
#include "surf/SurfaceTypeTraits.hh"

#include "detail/LogicUtils.hh"
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
/*!
 * Construct a transform from a simple translation.
 */
VariantTransform make_transform(Real3 const& translation)
{
    if (CELER_UNLIKELY(translation == (Real3{0, 0, 0})))
    {
        return NoTransformation{};
    }
    return Translation{translation};
}

//---------------------------------------------------------------------------//
/*!
 * Convert a vector of variants to a json array.
 */
template<class T>
nlohmann::json variants_to_json(std::vector<T> const& values)
{
    auto result = nlohmann::json::array();
    for (auto const& var : values)
    {
        auto j = nlohmann::json::object();
        std::visit([&j](auto&& u) { to_json(j, u); }, var);
        result.push_back(std::move(j));
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the bounding box or infinite if not there.
 */
BBox get_bbox(nlohmann::json const& j)
{
    if (auto iter = j.find("bbox"); iter != j.end())
    {
        return iter->get<BBox>();
    }
    return BBox::from_infinite();
}

//---------------------------------------------------------------------------//
}  // namespace

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
    for (auto surf_id : temp_faces)
    {
        CELER_ASSERT(surf_id != LocalSurfaceId{}.unchecked_get());
        value.faces.emplace_back(surf_id);
    }

    // Read scalars, including optional flags
    if (auto iter = j.find("flags"); iter != j.end())
    {
        iter->get_to(value.flags);
    }
    else
    {
        value.flags = 0;
    }

    if (auto iter = j.find("zorder"); iter != j.end())
    {
        if (iter->is_string())
        {
            auto s = iter->get<std::string>();
            CELER_VALIDATE(s.size() == 1, << "invalid zorder '" << s << "'");
            value.zorder = to_zorder(s.front());
        }
        else
        {
            // Backward compatibility: integer value
            iter->get_to(value.zorder);
            constexpr size_type uint16_max = 65536;
            if (static_cast<size_type>(value.zorder) == uint16_max - 2)
            {
                value.zorder = ZOrder::implicit_exterior;
            }
            else if (static_cast<size_type>(value.zorder) == uint16_max - 3)
            {
                value.zorder = ZOrder::exterior;
            }
            char c = to_char(value.zorder);
            CELER_VALIDATE(to_zorder(c) != ZOrder::invalid,
                           << "invalid zorder "
                           << static_cast<int>(value.zorder));
        }
    }
    else
    {
        value.zorder = ZOrder::media;
    }

    if (value.zorder == ZOrder::background)
    {
        // Background volumes should be "nowhere" explicitly using "inside"
        // logic
        value.logic = {logic::ltrue, logic::lnot};
        value.bbox = {};
    }
    else
    {
        // Convert logic string to vector
        value.logic = detail::string_to_logic(j.at("logic").get<std::string>());
        value.bbox = get_bbox(j);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write cell/volume data to an ORANGE JSON file.
 */
void to_json(nlohmann::json& j, VolumeInput const& value)
{
    CELER_EXPECT(value);

    // Convert faces from OpaqueId
    std::vector<LocalSurfaceId::size_type> temp_faces;
    temp_faces.reserve(value.faces.size());
    for (auto surf_id : value.faces)
    {
        temp_faces.emplace_back(surf_id.unchecked_get());
    }
    j["faces"] = std::move(temp_faces);

    // Convert logic string to vector
    if (!value.logic.empty())
    {
        j["logic"] = detail::logic_to_string(value.logic);
    }

    // Write optional values
    if (value.bbox != BBox::from_infinite())
    {
        j["bbox"] = value.bbox;
    }
    if (value.flags != 0)
    {
        j["flags"] = value.flags;
    }
    if (value.zorder != ZOrder::media)
    {
        j["zorder"] = std::string(1, to_char(value.zorder));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Read a unit definition from an ORANGE input file.
 *
 * NOTE: 'cell' nomenclature is from SCALE export (version 0)
 */
void from_json(nlohmann::json const& j, UnitInput& value)
{
    j.at("md").at("name").get_to(value.label);

    value.surfaces = detail::import_zipped_surfaces(j.at("surfaces"));
    for (char const* key : {"volumes", "cells"})
    {
        auto iter = j.find(key);
        if (iter != j.end())
        {
            iter->get_to(value.volumes);
            break;
        }
    }

    {
        // Move labels into lower-level data structures
        std::vector<Label> labels;
        for (char const* key : {"volume_labels", "cell_names"})
        {
            auto iter = j.find(key);
            if (iter != j.end())
            {
                iter->get_to(labels);
                break;
            }
        }
        CELER_VALIDATE(labels.size() == value.volumes.size() || labels.empty(),
                       << "incorrect size for volume labels: got "
                       << labels.size() << ", expected "
                       << value.volumes.size());
        for (auto i : range(labels.size()))
        {
            value.volumes[i].label = std::move(labels[i]);
        }
    }

    for (char const* key : {"surface_labels", "surface_names"})
    {
        auto iter = j.find(key);
        if (iter != j.end())
        {
            iter->get_to(value.surface_labels);
            break;
        }
    }
    CELER_VALIDATE(value.surface_labels.size() == value.surfaces.size()
                       || value.surface_labels.empty(),
                   << "incorrect size for surface labels: got "
                   << value.surface_labels.size() << ", expected "
                   << value.surfaces.size());

    value.bbox = get_bbox(j);

    for (char const* key : {"parent_volumes", "parent_cells"})
    {
        auto iter = j.find(key);
        if (iter == j.end())
        {
            continue;
        }
        auto const& parent_vols = iter->get<std::vector<size_type>>();

        auto const& daughters = j.at("daughters").get<std::vector<size_type>>();
        CELER_VALIDATE(parent_vols.size() == daughters.size(),
                       << "fields '" << key
                       << "' and 'daughters' have different lengths");

        std::vector<VariantTransform> transforms;  // SCALE ORANGE v0
        if (auto iter = j.find("transforms"); iter != j.end())
        {
            for (auto const& t : *iter)
            {
                transforms.push_back(detail::import_transform(t));
            }
        }
        else if (auto iter = j.find("translations"); iter != j.end())
        {
            auto translations = iter->get<std::vector<real_type>>();
            CELER_VALIDATE(3 * parent_vols.size() == translations.size(),
                           << "field 'translations' is not 3x length of '"
                           << key << "'");
            // Convert translations
            for (auto i : range(parent_vols.size()))
            {
                transforms.push_back(
                    make_transform(slice<3>(make_span(translations), i)));
            }
        }
        else
        {
            CELER_VALIDATE(false, << "missing 'transforms' or 'translations'");
        }
        CELER_ASSERT(transforms.size() == parent_vols.size());

        for (auto i : range(parent_vols.size()))
        {
            DaughterInput daughter;
            daughter.universe_id = UniverseId{daughters[i]};
            daughter.transform = std::move(transforms[i]);
            value.daughter_map.emplace(LocalVolumeId{parent_vols[i]},
                                       std::move(daughter));
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write a unit definition to an ORANGE input file.
 */
void to_json(nlohmann::json& j, UnitInput const& value)
{
    if (!value)
    {
        CELER_LOG(warning) << "Unit '" << value.label
                           << "' is not in a valid state";
    }

    j["_type"] = "unit";
    j["md"]["name"] = value.label;
    j["surfaces"] = detail::export_zipped_surfaces(value.surfaces);
    j["volumes"] = value.volumes;

    j["surface_labels"] = value.surface_labels;

    j["volume_labels"] = [&value] {
        auto volume_labels = nlohmann::json::array();
        for (auto const& v : value.volumes)
        {
            volume_labels.push_back(v.label);
        }
        return volume_labels;
    }();

    if (value.bbox && value.bbox != BBox::from_infinite())
    {
        // Write if not null or infinity (TODO: background volume is
        // actually "null",
        // other volumes should not be)
        j["bbox"] = value.bbox;
    }

    if (!value.daughter_map.empty())
    {
        std::vector<size_type> parent_cells;
        auto daughters = nlohmann::json::array();
        auto transforms = nlohmann::json::array();

        for (auto const& [local_vol, daughter_inp] : value.daughter_map)
        {
            parent_cells.push_back(local_vol.unchecked_get());
            daughters.push_back(daughter_inp.universe_id.unchecked_get());
            transforms.push_back(
                detail::export_transform(daughter_inp.transform));
        }
        j["parent_cells"] = std::move(parent_cells);
        j["daughters"] = std::move(daughters);
        j["transforms"] = std::move(transforms);
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
            = j.at(std::string(1, to_char(ax))).get<std::vector<double>>();
        CELER_VALIDATE(value.grid[to_int(ax)].size() >= 2,
                       << "axis " << to_char(ax)
                       << " does must have at least two grid points");
    }

    // Read daughters universes/translations
    {
        if (j.contains("transforms"))
        {
            CELER_NOT_IMPLEMENTED("transforms from JSON I/O");
        }

        std::vector<size_type> parents;
        if (auto iter = j.find("parent_cells"); iter != j.end())
        {
            iter->get_to(parents);
        }
        auto daughters = j.at("daughters").get<std::vector<size_type>>();
        auto translations = j.at("translations").get<std::vector<real_type>>();

        CELER_VALIDATE(3 * daughters.size() == translations.size(),
                       << "field 'translations' is not 3x length of "
                          "'daughters'");

        value.daughters.resize(daughters.size());

        for (auto i : range(daughters.size()))
        {
            DaughterInput daughter;
            daughter.universe_id = UniverseId{daughters[i]};

            // Read and convert transform
            daughter.transform
                = make_transform(slice<3>(make_span(translations), i));

            // Save daughter
            size_type parent = parents.empty() ? i : parents[i];
            value.daughters[parent] = std::move(daughter);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write a rectangular array universe definition to an ORANGE input file.
 */
void to_json(nlohmann::json& j, RectArrayInput const& value)
{
    CELER_EXPECT(value);

    j["_type"] = "rectarray";
    j["md"] = nlohmann::json::object({{"name", value.label}});

    for (auto ax : range(Axis::size_))
    {
        j[std::string(1, to_char(ax))] = value.grid[to_int(ax)];
    }

    // Write daughters universes/translations
    {
        std::vector<size_type> daughters;
        std::vector<real_type> translations;

        for (auto const& d : value.daughters)
        {
            daughters.push_back(d.universe_id.unchecked_get());
            if (auto* tr = std::get_if<Translation>(&d.transform))
            {
                translations.insert(translations.end(),
                                    tr->translation().begin(),
                                    tr->translation().end());
            }
            else if (std::holds_alternative<NoTransformation>(d.transform))
            {
                using R = real_type;
                translations.insert(translations.end(), {R{0}, R{0}, R{0}});
            }
            else
            {
                CELER_NOT_IMPLEMENTED("writing rect arrays with transforms");
            }
        }

        j["daughters"] = daughters;
        j["translations"] = translations;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Read tolerances.
 */
template<class T>
void from_json(nlohmann::json const& j, Tolerance<T>& value)
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

template void from_json(nlohmann::json const&, Tolerance<real_type>&);

//---------------------------------------------------------------------------//
/*!
 * Write tolerances.
 */
template<class T>
void to_json(nlohmann::json& j, Tolerance<T> const& value)
{
    CELER_EXPECT(value);

    j = {
        {"rel", value.rel},
        {"abs", value.abs},
    };
}

template void to_json(nlohmann::json&, Tolerance<real_type> const&);

//---------------------------------------------------------------------------//
/*!
 * Read a partially preprocessed geometry definition from an ORANGE JSON file.
 */
void from_json(nlohmann::json const& j, OrangeInput& value)
{
    CELER_VALIDATE(j.contains("_format"),
                   << "invalid ORANGE JSON input: no '_format' found");
    auto const& fmt = j.at("_format").get<std::string>();
    CELER_VALIDATE(fmt == "orange" || fmt == "ORANGE" || fmt == "SCALE ORANGE",
                   << "invalid ORANGE JSON input: unknown format '" << fmt
                   << "'");
    std::string version{"<unknown>"};
    if (auto iter = j.find("_version"); iter != j.end())
    {
        version = std::to_string(iter->get<int>());
    }
    CELER_LOG(debug) << "Reading '" << fmt << "' input version " << version;

    check_units(j, "ORANGE");

    auto const& universes = j.at("universes");
    value.universes.reserve(universes.size());

    for (auto const& uni : universes)
    {
        auto const& uni_type = uni.at("_type").get<std::string>();
        if (uni_type == "unit" || uni_type == "simple unit")
        {
            value.universes.push_back(uni.get<UnitInput>());
        }
        else if (uni_type == "rectarray" || uni_type == "rectangular array")
        {
            value.universes.push_back(uni.get<RectArrayInput>());
        }
        else
        {
            CELER_VALIDATE(
                false, << "unsupported universe type '" << uni_type << "'");
        }
    }

    if (auto iter = j.find("tol"); iter != j.end())
    {
        iter->get_to(value.tol);
    }
    else
    {
        value.tol = Tolerance<>::from_default();
        CELER_LOG(debug) << "No input tolerance provided: setting default "
                            "tolerance";
    }
    CELER_ENSURE(value);
}

//---------------------------------------------------------------------------//
/*!
 * Write an ORANGE input file.
 */
void to_json(nlohmann::json& j, OrangeInput const& value)
{
    CELER_EXPECT(value);

    j = nlohmann::json::object({
        {"_format", "ORANGE"},
        {"_version", 0},
        {"universes", variants_to_json(value.universes)},
    });
    if (value.tol)
    {
        j["tol"] = value.tol;
    }
    save_units(j);
}

//---------------------------------------------------------------------------//
/*!
 * Helper to read the input from a file or stream.
 *
 * Example to read from a file:
 * \code
   OrangeInput inp;
   std::ifstream("foo.org.json") >> inp;
 * \endcode
 */
std::istream& operator>>(std::istream& is, OrangeInput& inp)
{
    auto j = nlohmann::json::parse(is);
    j.get_to(inp);
    return is;
}

//---------------------------------------------------------------------------//
/*!
 * Helper to write the input to a file or stream.
 */
std::ostream& operator<<(std::ostream& os, OrangeInput const& inp)
{
    nlohmann::json j = inp;
    os << j.dump(0);
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
