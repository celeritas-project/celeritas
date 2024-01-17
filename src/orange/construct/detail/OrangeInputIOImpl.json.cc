//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/detail/OrangeInputIOImpl.json.cc
//---------------------------------------------------------------------------//
#include "OrangeInputIOImpl.json.hh"

#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/StringEnumMapper.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Convert a surface type string to an enum for I/O.
 */
SurfaceType to_surface_type(std::string const& s)
{
    static auto const from_string
        = StringEnumMapper<SurfaceType>::from_cstring_func(to_cstring,
                                                           "surface type");
    return from_string(s);
}

//---------------------------------------------------------------------------//
/*!
 * Create in-place a new variant surface in a vector.
 */
struct SurfaceEmplacer
{
    std::vector<VariantSurface>* surfaces;

    void operator()(SurfaceType st, Span<real_type const> data)
    {
        // Given the surface type, emplace a surface variant using the given
        // data.
        return visit_surface_type(
            [this, data](auto st_constant) {
                using Surface = typename decltype(st_constant)::type;
                using StorageSpan = typename Surface::StorageSpan;

                // Construct the variant on the back of the vector
                surfaces->emplace_back(std::in_place_type<Surface>,
                                       StorageSpan{data.data(), data.size()});
            },
            st);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Convert a logic token to a string.
 */
void logic_to_stream(std::ostream& os, logic_int val)
{
    if (logic::is_operator_token(val))
    {
        os << to_char(static_cast<logic::OperatorToken>(val));
    }
    else
    {
        // Just a face ID
        os << val;
    }
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Read a transform from a JSON object
 */
VariantTransform import_transform(nlohmann::json const& src)
{
    std::vector<real_type> data;
    src.get_to(data);
    if (data.size() == 0)
    {
        return NoTransformation{};
    }
    else if (data.size() == 3)
    {
        return Translation{Translation::StorageSpan{make_span(data)}};
    }
    else if (data.size() == 12)
    {
        return Transformation{Transformation::StorageSpan{make_span(data)}};
    }
    else
    {
        CELER_VALIDATE(false,
                       << "invalid number of elements in transform: "
                       << data.size());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write a transform to arrays suitable for JSON export.
 */
nlohmann::json export_transform(VariantTransform const& t)
{
    // Append the transform data as a single array. Rely on the size to
    // unpack it.
    return std::visit(
        [](auto&& tr) -> nlohmann::json {
            auto result = nlohmann::json::array();
            for (auto v : tr.data())
            {
                result.push_back(v);
            }
            return result;
        },
        t);
}

//---------------------------------------------------------------------------//
/*!
 * Read surface data from an ORANGE JSON file.
 */
std::vector<VariantSurface> import_zipped_surfaces(nlohmann::json const& j)
{
    // Read and convert types
    auto const& type_labels = j.at("types").get<std::vector<std::string>>();
    auto const& data = j.at("data").get<std::vector<real_type>>();
    auto const& sizes = j.at("sizes").get<std::vector<size_type>>();

    // Reserve space and create run-to-compile-to-runtime surface constructor
    std::vector<VariantSurface> result;
    result.reserve(type_labels.size());
    SurfaceEmplacer emplace_surface{&result};

    std::size_t data_idx = 0;
    for (auto i : range(type_labels.size()))
    {
        CELER_ASSERT(data_idx + sizes[i] <= data.size());
        emplace_surface(
            to_surface_type(type_labels[i]),
            Span<real_type const>{data.data() + data_idx, sizes[i]});
        data_idx += sizes[i];
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Write surface data to a JSON object.
 */
nlohmann::json export_zipped_surfaces(std::vector<VariantSurface> const& all_s)
{
    std::vector<std::string> surface_types;
    std::vector<real_type> surface_data;
    std::vector<size_type> surface_sizes;

    for (auto const& var_s : all_s)
    {
        std::visit(
            [&](auto&& s) {
                auto d = s.data();
                surface_types.push_back(to_cstring(s.surface_type()));
                surface_data.insert(surface_data.end(), d.begin(), d.end());
                surface_sizes.push_back(d.size());
            },
            var_s);
    }

    return nlohmann::json::object({
        {"types", std::move(surface_types)},
        {"data", std::move(surface_data)},
        {"sizes", std::move(surface_sizes)},
    });
}

//---------------------------------------------------------------------------//
/*!
 * Build a logic definition from a C string.
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
std::vector<logic_int> string_to_logic(std::string const& s)
{
    std::vector<logic_int> result;

    logic_int surf_id{};
    bool reading_surf{false};
    for (char v : s)
    {
        if (v >= '0' && v <= '9')
        {
            // Parse a surface number. 'Push' this digit onto the surface ID by
            // multiplying the existing ID by 10.
            if (!reading_surf)
            {
                surf_id = 0;
                reading_surf = true;
            }
            surf_id = 10 * surf_id + (v - '0');
            continue;
        }
        else if (reading_surf)
        {
            // Next char is end of word or end of string
            result.push_back(surf_id);
            reading_surf = false;
        }

        // Parse a logic token
        switch (v)
        {
                // clang-format off
            case '*': result.push_back(logic::ltrue); continue;
            case '|': result.push_back(logic::lor);   continue;
            case '&': result.push_back(logic::land);  continue;
            case '~': result.push_back(logic::lnot);  continue;
                // clang-format on
        }
        CELER_VALIDATE(v == ' ',
                       << "unexpected token '" << v
                       << "' while parsing logic string");
    }
    if (reading_surf)
    {
        result.push_back(surf_id);
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a logic vector to a string.
 */
std::string logic_to_string(std::vector<logic_int> const& logic)
{
    return to_string(celeritas::join_stream(
        logic.begin(), logic.end(), ' ', logic_to_stream));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
