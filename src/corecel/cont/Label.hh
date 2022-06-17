//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/Label.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <string>

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Small class for volume and material labels
struct Label
{
    std::string name; //!< Primary readable label component
    std::string ext; //!< Uniquifying component such as a pointer address or ID

    //! Create from just a name
    explicit Label(std::string n) : name{std::move(n)}, ext{} {}

    //! Create from a name and label
    Label(std::string n, std::string e) : name{std::move(n)}, ext{std::move(e)}
    {
    }
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
} // namespace celeritas
