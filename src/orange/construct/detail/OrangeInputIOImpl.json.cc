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
}  // namespace detail
}  // namespace celeritas
