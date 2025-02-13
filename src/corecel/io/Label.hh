//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/Label.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <string>
#include <tuple>
#include <utility>

#include "corecel/Config.hh"

#include "corecel/math/HashUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for managing volume and material labels.
 *
 * This class is needed because names in Geant4/VecGeom can be non-unique. The
 * only way to map between duplicate volume names between VecGeom and Geant4 is
 * to ensure that uniquifying, consistent extensions are written on output (and
 * not cleared on input), and to use those to differentiate the duplicate
 * volumes.
 *
 * Materials likewise can have duplicate names (perhaps because some have
 * different range cutoffs, etc.), so this class can be used to return a range
 * of IDs that match a single material name.
 *
 * \sa corecel/cont/LabelIdMultiMap.hh
 */
struct Label
{
    std::string name;  //!< Primary readable label component
    std::string ext;  //!< Uniquifying component: pointer address or ID

    //// STATIC DATA ////

    //! Default separator for output and splitting
    static constexpr char default_sep = '@';

    //// CLASS METHODS ////

    //! Create an empty label
    Label() = default;

    //! Create *implicitly* from a C string (mostly for testing)
    Label(char const* cstr) : name{cstr} {}

    //! Create *implicitly* from just a string name (capture)
    Label(std::string&& n) : name{std::move(n)} {}

    //! Create *implicitly* from just a string name (copy)
    Label(std::string const& n) : name{n} {}

    //! Create from a name and label
    Label(std::string n, std::string e) : name{std::move(n)}, ext{std::move(e)}
    {
    }

    //! Whether both the label and extension are empty
    bool empty() const { return name.empty() && ext.empty(); }

    //// STATIC METHODS ////

    // Construct a label from by splitting on a separator
    static Label
    from_separator(std::string const& name, char sep = default_sep);
};

//---------------------------------------------------------------------------//
//! Test equality
inline bool operator==(Label const& lhs, Label const& rhs)
{
    return lhs.name == rhs.name && lhs.ext == rhs.ext;
}

//! Test inequality
inline bool operator!=(Label const& lhs, Label const& rhs)
{
    return !(lhs == rhs);
}

//! Less-than comparison for sorting
inline bool operator<(Label const& lhs, Label const& rhs)
{
    return std::tie(lhs.name, lhs.ext) < std::tie(rhs.name, rhs.ext);
}

//---------------------------------------------------------------------------//
// Write a label to a stream
std::ostream& operator<<(std::ostream&, Label const&);

//---------------------------------------------------------------------------//
// Get the label as a string
std::string to_string(Label const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas

//---------------------------------------------------------------------------//
//! \cond
namespace std
{
//! Specialization for std::hash for unordered storage.
template<>
struct hash<celeritas::Label>
{
    using argument_type = celeritas::Label;
    using result_type = std::size_t;
    result_type operator()(argument_type const& label) const
        noexcept(!CELERITAS_DEBUG)
    {
        return celeritas::hash_combine(label.name, label.ext);
    }
};
}  // namespace std
//! \endcond
