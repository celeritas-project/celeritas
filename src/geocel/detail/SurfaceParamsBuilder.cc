//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/detail/SurfaceParamsBuilder.cc
//---------------------------------------------------------------------------//
#include "SurfaceParamsBuilder.hh"

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/io/Logger.hh"
#include "geocel/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// SurfaceInputInserter INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get the next surface ID to be added.
 */
SurfaceId SurfaceInputInserter::next_surface_id() const
{
    return id_cast<SurfaceId>(labels_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Get the label for a volume ID.
 */
Label const& SurfaceInputInserter::label(VolumeId vol_id) const
{
    CELER_EXPECT(vol_id < volumes_.num_volumes());
    return volumes_.volume_labels().at(vol_id);
}

//---------------------------------------------------------------------------//
/*!
 * Get the label for a volume instance ID.
 */
Label const& SurfaceInputInserter::label(VolumeInstanceId vol_inst_id) const
{
    CELER_EXPECT(vol_inst_id < volumes_.num_volume_instances());
    return volumes_.volume_instance_labels().at(vol_inst_id);
}

//---------------------------------------------------------------------------//
/*!
 * Get the label for a surface ID.
 */
Label const& SurfaceInputInserter::label(SurfaceId surface_id) const
{
    CELER_EXPECT(surface_id < labels_.size());
    return labels_[surface_id.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Construct with pointers to target data and volume params.
 */
SurfaceInputInserter::SurfaceInputInserter(
    VolumeParams const& volumes,
    std::vector<Label>& labels,
    std::vector<VolumeSurfaceData>& volume_surfaces)
    : volumes_(volumes), labels_{labels}, volume_surfaces_(volume_surfaces)
{
    CELER_EXPECT(labels_.empty());

    // Resize the vector to match the number of volumes
    volume_surfaces_.resize(volumes_.num_volumes());
}

//---------------------------------------------------------------------------//
/*!
 * Process an input surface and return its ID.
 */
SurfaceId SurfaceInputInserter::operator()(inp::Surface const& surf)
{
    CELER_ASSUME(!surf.surface.valueless_by_exception());
    SurfaceId result;
    try
    {
        result = std::visit(*this, surf.surface);
    }
    catch (RuntimeError& e)
    {
        CELER_LOG(error) << "While processing surface '" << surf.label << "'";
        throw;
    }
    labels_.push_back(surf.label);
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Process a boundary surface.
 */
SurfaceId SurfaceInputInserter::operator()(inp::Surface::Boundary const& vol_id)
{
    CELER_EXPECT(vol_id < volumes_.num_volumes());

    auto& vol_data = volume_surfaces_[vol_id.get()];

    // Check if a boundary already exists for this volume
    SurfaceId surf_id = this->next_surface_id();
    CELER_VALIDATE(!vol_data.boundary,
                   << "duplicate boundary surface for volume '"
                   << this->label(vol_id) << "': existing surface is'"
                   << this->label(vol_data.boundary) << "'");
    vol_data.boundary = surf_id;
    return surf_id;
}

//---------------------------------------------------------------------------//
/*!
 * Process an interface surface.
 */
SurfaceId
SurfaceInputInserter::operator()(inp::Surface::Interface const& interface)
{
    CELER_EXPECT(interface.first < volumes_.num_volume_instances());
    CELER_EXPECT(interface.second < volumes_.num_volume_instances());

    // Get the volume associated with the pre-step volume instance
    VolumeId pre_vol_id = volumes_.volume(interface.first);
    CELER_ASSERT(pre_vol_id < volume_surfaces_.size());
    auto& vol_data = volume_surfaces_[pre_vol_id.get()];

    // Try to insert the new mapping
    SurfaceId surf_id = this->next_surface_id();
    auto [iter, inserted] = vol_data.interfaces.insert({interface, surf_id});
    CELER_VALIDATE(inserted,
                   << "duplicate interface surface between volume instances '"
                   << this->label(interface.first) << "' and '"
                   << this->label(interface.second)
                   << "': existing surface is '" << this->label(iter->second)
                   << "'");

    return surf_id;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with pointers to target collections.
 */
VolumeSurfaceRecordBuilder::VolumeSurfaceRecordBuilder(
    VolumeItems<VolumeSurfaceRecord>* volume_surfaces,
    Items<VolumeInstanceId>* volume_instance_ids,
    Items<SurfaceId>* surface_ids)
    : volume_surfaces_{volume_surfaces}
    , volume_instance_ids_{volume_instance_ids}
    , surface_ids_{surface_ids}
{
    CELER_EXPECT(volume_surfaces && volume_instance_ids && surface_ids);
}

//---------------------------------------------------------------------------//
/*!
 * Convert VolumeSurfaceData to VolumeSurfaceRecord.
 */
VolumeId VolumeSurfaceRecordBuilder::operator()(VolumeSurfaceData const& data)
{
    VolumeSurfaceRecord record;

    // Set boundary surface ID if it exists
    record.boundary = data.boundary;

    if (!data.interfaces.empty())
    {
        // Flatten the interface map into sorted arrays

        // Save "pre" volume instance IDs
        auto pre_start = volume_instance_ids_.size_id();
        for (auto const& [key, surf_id] : data.interfaces)
        {
            CELER_DISCARD(surf_id);
            volume_instance_ids_.push_back(key.first);
        }

        // Save "post" volume instance IDs and surface IDs
        auto post_start = volume_instance_ids_.size_id();
        auto surf_start = surface_ids_.size_id();
        for (auto const& [key, surf_id] : data.interfaces)
        {
            // Add post volume instance ID and surface ID
            volume_instance_ids_.push_back(key.second);
            surface_ids_.push_back(surf_id);
        }

        // Set up ranges in the record
        record.interface_pre = {pre_start, post_start};
        record.interface_post = {post_start, volume_instance_ids_.size_id()};
        record.surface = {surf_start, surface_ids_.size_id()};

        CELER_ASSERT(record.interface_pre.size()
                     == record.interface_post.size());
        CELER_ASSERT(record.interface_pre.size() == record.surface.size());
    }

    // Add the record to the collection
    return volume_surfaces_.push_back(record);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
