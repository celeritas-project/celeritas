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
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/io/Join.hh"

#include "VolumeParams.hh"
#include "VolumePathFinder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Convert a unique volume instance ID to a slash-separated instance path.
 */
class UniqueVolumeToString
{
  public:
    //!@{
    //! \name Type aliases
    using SPVolumeParams = std::shared_ptr<VolumeParams const>;
    //!@}

  public:
    // Construct with shared volume metadata
    explicit UniqueVolumeToString(SPVolumeParams vols)
        : vols_(std::move(vols))
        , path_buffer_(vols_ ? vols_->num_volume_levels() : 0)
    {
        CELER_EXPECT(vols_);
    }

    // Convert a unique instance ID to slash-joined volume instance labels
    std::string operator()(VolumeUniqueInstanceId uid)
    {
        VolumePathFinder find_path{vols_->host_ref(), make_span(path_buffer_)};
        auto path = find_path(uid);
        return to_string(
            join(path.begin(), path.end(), '/', [this](VolumeInstanceId vi) {
                return to_string(vols_->volume_instance_labels().at(vi));
            }));
    }

  private:
    SPVolumeParams vols_;
    std::vector<VolumeInstanceId> path_buffer_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
