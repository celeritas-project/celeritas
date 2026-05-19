//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/UniqueVolumeToString.hh
//! \sa test/geocel/Volume.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Types.hh"

namespace celeritas
{
class VolumeParams;
//---------------------------------------------------------------------------//
/*!
 * Convert a unique volume instance ID to a slash-separated instance path.
 *
 * The value \c nullid, which corresponds to "outside", returns an empty
 * string. All other strings are slash-separated labels starting with the world
 * instance.
 */
class UniqueVolumeToString
{
  public:
    //!@{
    //! \name Type aliases
    using SPVolumeParams = std::shared_ptr<VolumeParams const>;
    //!@}

  public:
    // Construct with non-owning shared pointer (limited lifetime)
    static UniqueVolumeToString from_ref(VolumeParams const&);

    // Construct with shared volume metadata
    explicit UniqueVolumeToString(SPVolumeParams vols);

    // Apply conversion
    std::string operator()(VolumeUniqueInstanceId uid);

  private:
    SPVolumeParams vols_;
    VolumeInstanceId world_instance_;
    std::vector<VolumeInstanceId> path_buffer_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
