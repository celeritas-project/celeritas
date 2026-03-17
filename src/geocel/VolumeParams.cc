//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeParams.cc
//---------------------------------------------------------------------------//
#include "VolumeParams.hh"

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/ParamsDataStore.hh"
#include "corecel/io/Logger.hh"

#include "VolumeVisitor.hh"
#include "inp/Model.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Accessor wrapping a host VolumeParamsData for use with VolumeVisitor.
 */
template<Ownership W>
struct VolumeDataAccessor
{
    using VolumeRef = VolumeId;
    using VolumeInstanceRef = VolumeInstanceId;

    VolumeParamsData<W, MemSpace::host> const& params;

    VolumeId volume(VolumeInstanceId vi) const
    {
        return params.volume_ids[vi];
    }

    Span<VolumeInstanceId const> children(VolumeId v) const
    {
        return params.vi_storage[params.volumes[v].children];
    }
};

//---------------------------------------------------------------------------//
int calc_num_volume_levels(HostVal<VolumeParamsData> const& params)
{
    CELER_EXPECT(params.scalars.world);
    int max_level{0};

    std::vector<bool> visited(params.scalars.num_volumes, false);
    VolumeVisitor visit_vol{VolumeDataAccessor<Ownership::value>{params}};
    visit_vol(
        [&max_level, &visited](VolumeId v, int level) {
            if (visited[v.unchecked_get()])
            {
                return false;
            }
            visited[v.unchecked_get()] = true;
            CELER_ASSERT(level >= 0);
            max_level = std::max(max_level, level);
            return true;
        },
        params.scalars.world);
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

    auto const num_volumes = v_labels_.size();
    auto const num_volume_instances = vi_labels_.size();

    // Aggregate parents: scan all volume instances and record which volumes
    // each instance belongs to
    std::vector<std::vector<VolumeInstanceId>> parent_lists(num_volumes);
    for (auto vi_idx : range(num_volume_instances))
    {
        auto const& vol_inst = in.volume_instances[vi_idx];
        if (!vol_inst)
        {
            continue;
        }
        CELER_EXPECT(vol_inst.volume < num_volumes);
        parent_lists[vol_inst.volume.unchecked_get()].push_back(
            VolumeInstanceId{vi_idx});
    }

    // Build host data
    HostVal<VolumeParamsData> host_data;
    CollectionBuilder vol_builder{&host_data.volumes};
    CollectionBuilder vi_ids_builder{&host_data.volume_ids};
    CollectionBuilder vi_storage_builder{&host_data.vi_storage};

    // Build per-volume records
    for (auto vol_idx : range(num_volumes))
    {
        auto const& vol_children = in.volumes[vol_idx].children;
        CELER_EXPECT(std::all_of(
            vol_children.begin(), vol_children.end(), [&](auto const& id) {
                return id < num_volume_instances;
            }));

        VolumeRecord rec;
        rec.material = in.volumes[vol_idx].material;
        rec.children = vi_storage_builder.insert_back(vol_children.begin(),
                                                      vol_children.end());
        auto const& parents = parent_lists[vol_idx];
        rec.parents
            = vi_storage_builder.insert_back(parents.begin(), parents.end());
        vol_builder.push_back(rec);
    }

    // Build volume_ids: map VolumeInstanceId -> VolumeId
    for (auto vi_idx : range(num_volume_instances))
    {
        auto const& vol_inst = in.volume_instances[vi_idx];
        vi_ids_builder.push_back(vol_inst ? vol_inst.volume : VolumeId{});
    }

    // Set scalars
    CELER_EXPECT(!in.world || in.world < in.volumes.size());
    host_data.scalars.world = in.world;
    host_data.scalars.num_volumes = num_volumes;
    host_data.scalars.num_volume_instances = num_volume_instances;

    // Calculate depth via VolumeVisitor
    if (in.world)
    {
        host_data.scalars.num_volume_levels = calc_num_volume_levels(host_data);
    }

    CELER_ENSURE(host_data);
    data_ = ParamsDataStore<VolumeParamsData>{std::move(host_data)};

    CELER_ENSURE(data_.host_ref().volumes.size() == num_volumes);
    CELER_ENSURE(data_.host_ref().volume_ids.size() == num_volume_instances);
    CELER_ENSURE((this->num_volume_levels() == 0) == this->empty());
}

//---------------------------------------------------------------------------//
//! Construct with no volumes, often for unit testing
VolumeParams::VolumeParams() : VolumeParams{inp::Volumes{}} {}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class ParamsDataStore<VolumeParamsData>;
template class ParamsDataInterface<VolumeParamsData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
