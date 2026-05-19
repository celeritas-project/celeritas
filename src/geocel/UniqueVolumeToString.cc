//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/UniqueVolumeToString.cc
//---------------------------------------------------------------------------//
#include "UniqueVolumeToString.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/io/Join.hh"

#include "VolumeParams.hh"
#include "VolumePathFinder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared volume metadata.
 */
UniqueVolumeToString::UniqueVolumeToString(SPVolumeParams vols)
    : vols_(std::move(vols))
    , path_buffer_(vols_ ? vols_->num_volume_levels() : 0)
{
    CELER_EXPECT(vols_);
}

//---------------------------------------------------------------------------//
/*!
 * Apply conversion.
 */
std::string UniqueVolumeToString::operator()(VolumeUniqueInstanceId uid)
{
    VolumePathFinder find_path{vols_->host_ref(), make_span(path_buffer_)};
    auto path = find_path(uid);
    return to_string(
        join(path.begin(), path.end(), '/', [this](VolumeInstanceId vi) {
            return to_string(vols_->volume_instance_labels().at(vi));
        }));
}

}  // namespace celeritas
