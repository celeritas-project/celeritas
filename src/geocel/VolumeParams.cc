//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeParams.cc
//---------------------------------------------------------------------------//
#include "VolumeParams.hh"

#include "inp/Model.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from input.
 */
VolumeParams::VolumeParams(inp::Volumes const& in)
{
    CELER_VALIDATE(!in.volumes.empty(), << "no volumes were available");
    // Lambda to extract labels from a collection
    auto extract_labels = [](auto const& items) {
        std::vector<Label> labels;
        labels.reserve(items.size());
        for (auto const& item : items)
        {
            labels.push_back(item.label);
        }
        return labels;
    };

    // Build label maps
    v_labels_ = VolumeMap("volume", extract_labels(in.volumes));
    vi_labels_
        = VolInstMap("volume_instance", extract_labels(in.volume_instances));

    // Unzip volume properties
    materials_.resize(in.volumes.size());
    children_.resize(in.volumes.size());
    for (auto vol_idx : range(in.volumes.size()))
    {
        materials_[vol_idx] = in.volumes[vol_idx].material;

        // Reserve space for children
        auto const& vol_children = in.volumes[vol_idx].children;
        // Validate all children are valid volume instance IDs
        CELER_EXPECT(std::all_of(
            vol_children.begin(), vol_children.end(), [&in](auto const& id) {
                return id < in.volume_instances.size();
            }));

        // Set children instances
        children_[vol_idx].assign(vol_children.begin(), vol_children.end());
    }

    // Unzip volume instances and add parent relationships
    volumes_.resize(in.volume_instances.size());
    parents_.resize(in.volumes.size());
    for (auto vi_idx : range(in.volume_instances.size()))
    {
        auto const& vol_inst = in.volume_instances[vi_idx];

        // Store the logical volume that this physical volume instantiates
        CELER_EXPECT(vol_inst.volume < in.volumes.size());
        volumes_[vi_idx] = vol_inst.volume;

        // Add this instance as a parent of its referenced volume
        CELER_EXPECT(vol_inst.volume < parents_.size());
        parents_[vol_inst.volume.unchecked_get()].push_back(
            id_cast<VolumeInstanceId>(vi_idx));
    }

    // Add missing parent-child relationships by cross-referencing
    // Child relationships are defined in the volumes, but we should ensure
    // they're consistent with the volume instances

    // Validate sizes of all containers
    CELER_ENSURE(v_labels_.size() == in.volumes.size());
    CELER_ENSURE(vi_labels_.size() == in.volume_instances.size());
    CELER_ENSURE(materials_.size() == in.volumes.size());
    CELER_ENSURE(parents_.size() == in.volumes.size());
    CELER_ENSURE(children_.size() == in.volumes.size());
    CELER_ENSURE(volumes_.size() == in.volume_instances.size());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
