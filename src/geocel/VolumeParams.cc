//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeParams.cc
//! \sa test/geocel/Volume.test.cc
//---------------------------------------------------------------------------//
#include "VolumeParams.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "corecel/data/ParamsDataStore.hh"
#include "corecel/io/Logger.hh"
#include "geocel/Types.hh"

#include "VolumeVisitor.hh"
#include "inp/Model.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// Note: ull_int is used in this file as shorthand
static_assert(std::is_same_v<ull_int, VolumeUniqueInstanceId::size_type>);

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
/*!
 * Compute the number of unique paths ending at any node in each subtree.
 *
 * For a volume V, this equals 1 (the path ending at V itself) plus the sum of
 * the counts for each child volume:
 *   num_desc(V) = 1 + sum(num_desc(volume(vi)) for vi in V.children).
 * The leading 1 accounts for the path that terminates exactly at V without
 * descending further.  Computed bottom-up via iterative post-order DFS so
 * that shared sub-volumes (DAG diamonds) are evaluated only once.
 */
std::vector<ull_int>
calc_num_descendants(HostVal<VolumeParamsData> const& params)
{
    CELER_EXPECT(params.scalars.world);

    auto const num_volumes = params.volumes.size();
    std::vector<ull_int> num_descendents(num_volumes, 0);

    auto num_desc = [&num_descendents](VolumeId v) -> ull_int& {
        CELER_EXPECT(v < num_descendents.size());
        return num_descendents[v.unchecked_get()];
    };

    // Iterative post-order DFS: pair (volume, fully_expanded)
    std::vector<std::pair<VolumeId, bool>> stack
        = {{params.scalars.world, false}};

    while (!stack.empty())
    {
        auto [v, expanded] = stack.back();
        stack.pop_back();

        if (num_desc(v) != 0)
        {
            // Already computed; reachable via multiple ancestor paths (DAG)
            continue;
        }
        auto children = params.vi_storage[params.volumes[v].children];
        if (expanded)
        {
            // All children computed; accumulate self + children
            ull_int n = 1;
            for (VolumeInstanceId vi : children)
            {
                n += num_desc(params.volume_ids[vi]);
            }
            num_desc(v) = n;
        }
        else
        {
            // Push self again for post-processing, then push uncomputed
            // children
            stack.push_back({v, true});
            for (VolumeInstanceId vi : children)
            {
                VolumeId child = params.volume_ids[vi];
                if (num_desc(child) == 0)
                {
                    stack.push_back({child, false});
                }
            }
        }
    }

    return num_descendents;
}

//---------------------------------------------------------------------------//
/*!
 * Precompute unique-instance offsets for all volume instances.
 *
 * For each volume instance \c vi at position \c k in parent volume \c P's
 * children list, the offset is the sum of \c num_desc[volume(vj)] for all
 * preceding siblings \c vj (positions 0..k-1). Volume instances not appearing
 * in any children list receive offset 0.
 */
std::vector<ull_int>
calc_unique_instance_offsets(HostVal<VolumeParamsData> const& params,
                             std::vector<ull_int> const& num_desc)
{
    auto const num_vi = params.volume_ids.size();
    std::vector<ull_int> offsets(num_vi, 0);

    for (auto vol_idx : range(params.volumes.size()))
    {
        ull_int running = 0;
        for (VolumeInstanceId vi :
             params.vi_storage[params.volumes[VolumeId{vol_idx}].children])
        {
            offsets[vi.unchecked_get()] = running;
            running += num_desc[params.volume_ids[vi].unchecked_get()];
        }
    }

    return offsets;
}

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from input.
 */
VolumeParams::VolumeParams(inp::Volumes const& in)
{
    CELER_VALIDATE(in.volumes.empty() != static_cast<bool>(in.world),
                   << "missing world volume (or unused volumes provided)");
    CELER_VALIDATE(!in.world || in.world < in.volumes.size(),
                   << "world volume ID " << in.world
                   << " is out of range (num_volumes = " << in.volumes.size()
                   << ")");

    // Build label maps
    v_labels_ = VolumeMap::from_labeled_items("volume", in.volumes);
    vi_labels_ = VolInstMap::from_labeled_items("volume_instance",
                                                in.volume_instances);

    // TODO: warn about duplicate labels (see LabelIdMultiMap::duplicates)
    auto const num_volumes = v_labels_.size();
    auto const num_volume_instances = vi_labels_.size();

    // Aggregate parents: scan all volume instances and record which volumes
    // each instance belongs to
    std::vector<std::vector<VolumeInstanceId>> parent_lists(num_volumes);
    for (auto vi : range(VolumeInstanceId{num_volume_instances}))
    {
        auto const& vi_inp = in.volume_instances[vi.get()];
        if (!vi_inp)
        {
            // TODO: in practice we should prohibit unused volume instances
            continue;
        }
        CELER_VALIDATE(vi_inp.volume < num_volumes,
                       << "assigned volume (" << vi_inp.volume
                       << ") is out of range (" << num_volumes
                       << ") for volume instance " << vi << "='"
                       << vi_labels_.at(vi) << "'");
        parent_lists[vi_inp.volume.unchecked_get()].push_back(vi);
    }

    // Build host data
    HostVal<VolumeParamsData> host_data;
    CollectionBuilder vol_builder{&host_data.volumes};
    CollectionBuilder vi_ids_builder{&host_data.volume_ids};
    DedupeCollectionBuilder vi_storage_builder{&host_data.vi_storage};

    // Set basic scalars
    host_data.scalars.world = in.world;
    host_data.scalars.num_volumes = num_volumes;
    host_data.scalars.num_volume_instances = num_volume_instances;
    if (in)
    {
        // Set the enclosing instance of the world volume (if any)
        CELER_ASSERT(in.world < parent_lists.size());
        auto const& world_parents = parent_lists[in.world.get()];
        if (!world_parents.empty())
        {
            CELER_ASSERT(world_parents.size() == 1);
            host_data.scalars.world_instance = world_parents.front();
        }
    }

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
        vi_ids_builder.push_back(vol_inst.volume);
    }

    if (in)
    {
        // Calculate depth via VolumeVisitor
        host_data.scalars.num_volume_levels = calc_num_volume_levels(host_data);

        // Compute unique-instance offsets
        CollectionBuilder offsets_builder{&host_data.unique_instance_offsets};
        auto const num_desc = calc_num_descendants(host_data);
        host_data.scalars.num_unique_instances
            = num_desc[host_data.scalars.world.get()];
        auto const offsets = calc_unique_instance_offsets(host_data, num_desc);
        offsets_builder.insert_back(offsets.begin(), offsets.end());
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
