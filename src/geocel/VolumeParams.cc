//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeParams.cc
//---------------------------------------------------------------------------//
#include "VolumeParams.hh"

#include "corecel/io/Logger.hh"

#include "VolumeVisitor.hh"
#include "inp/Model.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
int calc_num_volume_levels(VolumeParams const& params)
{
    CELER_EXPECT(params.world());
    int max_level{0};

    VolumeVisitor visit_vol{params};
    visit_vol(
        [&max_level](VolumeId, int level) {
            CELER_ASSERT(level >= 0);
            max_level = std::max(max_level, level);
            return true;
        },
        params.world());
    return max_level + 1;
}

//---------------------------------------------------------------------------//
//! Volumes corresponding to global tracking model
std::weak_ptr<VolumeParams const> g_volumes_;

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Set global geometry instance.
 *
 * This allows many parts of the codebase to independently access Geant4
 * metadata. It should be called during initialization of any Celeritas front
 * end that integrates with Geant4. We can't use shared pointers here because
 * of global initialization order issues (the low-level Geant4 objects may be
 * cleared before a static celeritas::VolumeParams is destroyed).
 *
 * \note This should be done only during setup on the main thread.
 */
void global_volumes(std::shared_ptr<VolumeParams const> const& gv)
{
    CELER_LOG(debug) << (!gv                    ? "Clearing"
                         : g_volumes_.expired() ? "Setting"
                                                : "Updating")
                     << " celeritas::volumes";
    g_volumes_ = gv;
}

//---------------------------------------------------------------------------//
/*!
 * Access the global canonical volume metadata.
 *
 * This can be used by geometry-related helper functions throughout
 * the code base.
 *
 * \return Weak pointer to the global VolumeParams wrapper, which may be null.
 */
std::weak_ptr<VolumeParams const> const& global_volumes()
{
    return g_volumes_;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from input.
 */
VolumeParams::VolumeParams(inp::Volumes const& in)
{
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

    // TODO: warn about duplicate labels (see LabelIdMultiMap::duplicates)

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

    // Save world
    CELER_EXPECT(!in.world || in.world < in.volumes.size());
    world_ = in.world;

    // Calculate additional properties
    if (world_)
    {
        num_volume_levels_ = calc_num_volume_levels(*this);
    }

    CELER_ENSURE(this->num_volumes() == in.volumes.size());
    CELER_ENSURE(this->num_volume_instances() == in.volume_instances.size());
    CELER_ENSURE(v_labels_.size() == this->num_volumes());
    CELER_ENSURE(vi_labels_.size() == this->num_volume_instances());
    CELER_ENSURE(materials_.size() == this->num_volumes());
    CELER_ENSURE(parents_.size() == this->num_volumes());
    CELER_ENSURE(children_.size() == this->num_volumes());
    CELER_ENSURE(volumes_.size() == this->num_volume_instances());
    CELER_ENSURE((this->num_volume_levels() == 0) == this->empty());
}

//---------------------------------------------------------------------------//
//! Construct with no volumes, often for unit testing
VolumeParams::VolumeParams() : VolumeParams{inp::Volumes{}} {}

//---------------------------------------------------------------------------//
}  // namespace celeritas
