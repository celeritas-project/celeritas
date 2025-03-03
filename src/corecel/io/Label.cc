//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/Label.cc
//---------------------------------------------------------------------------//
#include "Label.hh"

#include <ostream>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Create *implicitly* from a name without extension.
 */
Label::Label(std::string_view n) : name{n} {}

//---------------------------------------------------------------------------//
/*!
 * Construct a label from by splitting on a separator.
 */
Label Label::from_separator(std::string_view name, char sep)
{
    auto pos = name.rfind(sep);
    if (pos == std::string_view::npos)
    {
        return Label{std::string{name}};
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
