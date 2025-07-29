//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeToString.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <variant>

#include "corecel/io/Label.hh"

#include "Types.hh"

namespace celeritas
{
class VolumeParams;
//---------------------------------------------------------------------------//
/*!
 * Visitor class that converts volume-related types to string representations.
 *
 * This class can be used with std::visit to convert variant types to strings.
 * When constructed with VolumeParams reference, it will look up labels for
 * IDs; otherwise it will print just the ID value or a null indicator.
 */
class VolumeToString
{
  public:
    //!@{
    //! \name Type aliases
    using VariantLabel
        = std::variant<Label, VolumeId, VolumeInstanceId, std::string>;
    //!@}

  public:
    // Construct without any labels
    VolumeToString() = default;

    // Construct with volume params reference
    inline explicit VolumeToString(VolumeParams const& params);

    // Overloaded operators for different types
    std::string operator()(Label const& label) const;
    std::string operator()(VolumeId const& id) const;
    std::string operator()(VolumeInstanceId const& id) const;

  private:
    VolumeParams const* params_{nullptr};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with volume params reference.
 */
VolumeToString::VolumeToString(VolumeParams const& params) : params_(&params)
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
