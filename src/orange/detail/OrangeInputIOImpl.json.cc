//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/OrangeInputIOImpl.json.cc
//---------------------------------------------------------------------------//
#include "OrangeInputIOImpl.json.hh"

#include <vector>

#include "corecel/cont/Range.hh"
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
    CELER_ASSERT_UNREACHABLE();
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
}  // namespace detail
}  // namespace celeritas
