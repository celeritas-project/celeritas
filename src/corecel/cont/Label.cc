//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/Label.cc
//---------------------------------------------------------------------------//
#include "Label.hh"

#include <ostream>
#include <regex>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct a label from a Geant4 pointer-appended name.
 */
Label Label::from_geant(std::string const& name)
{
    static const std::regex final_ptr_regex{"0x[0-9a-f]{4,16}$"};

    // Remove possible Geant uniquifying pointer-address suffix
    // (Geant4 does this automatically, but VGDML does not)
    auto split_point = name.end();
    std::smatch ptr_match;
    if (std::regex_search(name, ptr_match, final_ptr_regex))
    {
        // Copy pointer as 'extension' and delete from name
        split_point = name.begin() + ptr_match.position(0);
    }

    Label result;
    result.name.assign(name.begin(), split_point);
    result.ext.assign(split_point, name.end());
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct a label from by splitting on a separator.
 */
Label Label::from_separator(std::string const& name, char sep)
{
    auto pos = name.rfind(sep);
    if (pos == std::string::npos)
    {
        return Label{name};
    }

    auto iter = name.begin() + pos;
    return Label{std::string(name.begin(), iter),
                 std::string(iter + 1, name.end())};
}

//---------------------------------------------------------------------------//
/*!
 * Write a label to a stream.
 *
 * \todo account for \c os.width .
 */
std::ostream& operator<<(std::ostream& os, Label const& lab)
{
    os << lab.name;

    if (lab.ext.empty())
    {
        // No extension: don't add a separator
        return os;
    }
    os << Label::default_sep << lab.ext;

    return os;
}

//---------------------------------------------------------------------------//
/*!
 * Get the label as a string.
 */
std::string to_string(Label const& lab)
{
    std::ostringstream os;
    os << lab;
    return os.str();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
