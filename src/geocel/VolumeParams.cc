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
    CELER_VALIDATE(in, << "no volumes were available");

    // Build label maps
    auto extract_labels = [](auto const& items) {
        std::vector<Label> labels;
        labels.reserve(items.size());
        for (auto const& item : items)
        {
            labels.push_back(item.label);
        }
        return labels;
    };
    v_labels_ = VolumeMap("volume", extract_labels(in.volumes));
    vi_labels_
        = VolInstMap("volume_instance", extract_labels(in.volume_instances));

    // Unzip volume properties
    materials_.resize(this->num_volumes());
    children_.resize(this->num_volumes());
    for (auto vol_idx : range(this->num_volumes()))
    {
        materials_[vol_idx] = in.volumes[vol_idx].material;

        // Set children instances
        auto const& vol_children = in.volumes[vol_idx].children;
        CELER_EXPECT(std::all_of(
            vol_children.begin(), vol_children.end(), [this](auto const& id) {
                return id < this->num_volume_instances();
            }));
        children_[vol_idx].assign(vol_children.begin(), vol_children.end());
    }

    // Unzip volume instances and add parent relationships
    volumes_.resize(this->num_volume_instances());
    parents_.resize(this->num_volumes());
    for (auto vi_idx : range(this->num_volume_instances()))
    {
        auto const& vol_inst = in.volume_instances[vi_idx];
        if (!vol_inst)
        {
            // Skip null volume instance
            continue;
        }

        // Store the logical volume that this physical volume instantiates
        volumes_[vi_idx] = vol_inst.volume;

        // Add this instance as a parent of its referenced volume
        CELER_EXPECT(vol_inst.volume < this->num_volumes());
        parents_[vol_inst.volume.unchecked_get()].push_back(
            VolumeInstanceId{vi_idx});
    }

    CELER_ENSURE(this->num_volumes() == in.volumes.size());
    CELER_ENSURE(this->num_volume_instances() == in.volume_instances.size());
    CELER_ENSURE(v_labels_.size() == this->num_volumes());
    CELER_ENSURE(vi_labels_.size() == this->num_volume_instances());
    CELER_ENSURE(materials_.size() == this->num_volumes());
    CELER_ENSURE(parents_.size() == this->num_volumes());
    CELER_ENSURE(children_.size() == this->num_volumes());
    CELER_ENSURE(volumes_.size() == this->num_volume_instances());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
