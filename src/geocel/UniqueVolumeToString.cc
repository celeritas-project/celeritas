//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/UniqueVolumeToString.cc
//---------------------------------------------------------------------------//
#include "UniqueVolumeToString.hh"

#include <utility>

#include "corecel/Assert.hh"

#include "VolumeParams.hh"
#include "VolumePathFinder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with non-owning shared pointer.
 *
 * This convenience function is for the case when shared pointers are
 * unavailable (unit testing, non-persistent usage).
 */
UniqueVolumeToString UniqueVolumeToString::from_ref(VolumeParams const& vols)

{
    return UniqueVolumeToString{std::shared_ptr<VolumeParams const>{
        &vols, [](VolumeParams const*) {}}};
}

//---------------------------------------------------------------------------//
/*!
 * Construct with shared volume metadata.
 */
UniqueVolumeToString::UniqueVolumeToString(SPVolumeParams vols)
    : vols_(std::move(vols))
    , path_buffer_(vols_ ? vols_->num_volume_levels() : 0)
{
    CELER_EXPECT(vols_);
    world_instance_ = vols_->world_instance();
}

//---------------------------------------------------------------------------//
/*!
 * Unfold to a string.
 */
std::string UniqueVolumeToString::operator()(VolumeUniqueInstanceId uid)
{
    if (!uid)
    {
        return {};
    }

    // Find the volume, filling the buffer and returning a span
    VolumePathFinder find_path{vols_->host_ref(), make_span(path_buffer_)};
    auto path = find_path(uid);

    // Add the world volume to the output first
    auto const& labels = vols_->volume_instance_labels();
    std::ostringstream os;
    if (world_instance_)
    {
        // Print the world instance: always true if Geant4, but not necessarily
        // for manually constructed geometry graphs.
        os << labels.at(world_instance_);
    }
    else
    {
        os << "[WORLD]";
    }
    // Add all daughter volumes
    for (VolumeInstanceId vi : path)
    {
        CELER_ASSERT(vi);
        os << '/' << labels.at(vi);
    }
    return std::move(os).str();
}

}  // namespace celeritas
