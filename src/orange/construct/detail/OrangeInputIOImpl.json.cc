//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/detail/OrangeInputIOImpl.json.cc
//---------------------------------------------------------------------------//
#include "OrangeInputIOImpl.json.hh"

#include <vector>

#include "corecel/io/StringEnumMapper.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace detail
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
                using Storage = typename Surface::Storage;

                // Construct the variant on the back of the vector
                surfaces->emplace_back(std::in_place_type<Surface>,
                                       Storage{data.data(), data.size()});
            },
            st);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Read surface data from an ORANGE JSON file.
 */
std::vector<VariantSurface> read_surfaces(nlohmann::json const& j)
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
std::vector<logic_int> parse_logic(char const* c)
{
    std::vector<logic_int> result;
    logic_int s = 0;
    while (char v = *c++)
    {
        if (v >= '0' && v <= '9')
        {
            // Parse a surface number. 'Push' this digit onto the surface ID by
            // multiplying the existing ID by 10.
            s = 10 * s + (v - '0');

            char const next = *c;
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
/*!
 * Construct a transform from a translation.
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
}  // namespace detail
}  // namespace celeritas
