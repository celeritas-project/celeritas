//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/Label.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <iosfwd>
#include <string>

#include "corecel/math/HashUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Small class for volume and material labels
struct Label
{
    std::string name; //!< Primary readable label component
    std::string ext; //!< Uniquifying component such as a pointer address or ID

    //// STATIC DATA ////

    //! Default separator for output and splitting
    static constexpr char default_sep = '@';

    //// CLASS METHODS ////

    //! Create an empty label
    Label() = default;

    //! Create from just a name
    explicit Label(std::string n) : name{std::move(n)}, ext{} {}

    //! Create from a name and label
    Label(std::string n, std::string e) : name{std::move(n)}, ext{std::move(e)}
    {
    }

    //// STATIC METHODS ////

    // Construct a label from a Geant4 pointer-appended name
    static Label from_geant4(const std::string& name);

    // Construct a label from by splitting on a separator
    static Label
    from_separator(const std::string& name, char sep = default_sep);
};

//---------------------------------------------------------------------------//
//! Test equality
inline constexpr bool operator==(const Label& lhs, const Label& rhs)
{
    return lhs.name == rhs.name && lhs.ext == rhs.ext;
}

//! Test inequality
inline constexpr bool operator!=(const Label& lhs, const Label& rhs)
{
    return !(lhs == rhs);
}

//! Less-than comparison for sorting
inline bool operator<(const Label& lhs, const Label& rhs)
{
    if (lhs.name < rhs.name)
        return true;
    else if (lhs.name > rhs.name)
        return false;
    if (lhs.ext < rhs.ext)
        return true;
    return false;
}

//---------------------------------------------------------------------------//
// Write a label to a stream
std::ostream& operator<<(std::ostream&, const Label&);

//---------------------------------------------------------------------------//
// Get the label as a string
std::string to_string(const Label&);

//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//
//! \cond
namespace std
{
//! Specialization for std::hash for unordered storage.
template<>
struct hash<celeritas::Label>
{
    using argument_type = celeritas::Label;
    using result_type   = std::size_t;
    result_type operator()(const argument_type& label) const noexcept
    {
        return celeritas::hash_combine(label.name, label.ext);
    }
};
} // namespace std
//! \endcond
