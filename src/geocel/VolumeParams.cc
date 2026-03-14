//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeParams.cc
//! \sa test/geocel/Volume.test.cc
//---------------------------------------------------------------------------//
#include "VolumeParams.hh"

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
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
    auto const num_volumes = params.volumes.size();
    std::vector<ull_int> num_desc(num_volumes, 0);

    // Iterative post-order DFS: pair (volume, fully_expanded)
    std::vector<std::pair<VolumeId, bool>> stack;
    if (params.scalars.world)
    {
        stack.push_back({params.scalars.world, false});
    }

    while (!stack.empty())
    {
        auto [v, expanded] = stack.back();
        stack.pop_back();

        if (num_desc[v.unchecked_get()] != 0)
        {
            // Already computed; reachable via multiple ancestor paths (DAG)
            continue;
        }
        if (expanded)
        {
            // All children computed; accumulate self + children
            ull_int n = 1;
            for (VolumeInstanceId vi :
                 params.vi_storage[params.volumes[v].children])
            {
                n += num_desc[params.volume_ids[vi].unchecked_get()];
            }
            num_desc[v.unchecked_get()] = n;
        }
        else
        {
            // Push self again for post-processing, then push uncomputed
            // children
            stack.push_back({v, true});
            for (VolumeInstanceId vi :
                 params.vi_storage[params.volumes[v].children])
            {
                VolumeId child = params.volume_ids[vi];
                if (num_desc[child.unchecked_get()] == 0)
                {
                    stack.push_back({child, false});
                }
            }
        }
    }

    return num_desc;
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
    for (auto vi : range(VolumeInstanceId{num_volume_instances}))
    {
        auto const& vi_inp = in.volume_instances[vi.get()];
        if (!vi_inp)
        {
            continue;
        }
        CELER_VALIDATE(vi_inp.volume < num_volumes,
                       << "assigned volume (" << vi_inp.volume
                       << ") is out of range (" << num_volumes
                       << ") for volume instance " << vi_inp.volume << "='"
                       << vi_labels_.at(vi) << "'");
        parent_lists[vi_inp.volume.unchecked_get()].push_back(vi);
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

    // Set world_instance: the enclosing instance of the world volume, if any
    if (in.world)
    {
        auto world_parents
            = host_data.vi_storage[host_data.volumes[in.world].parents];
        if (!world_parents.empty())
        {
            host_data.scalars.world_instance = world_parents.front();
        }
    }

    // Calculate depth via VolumeVisitor
    if (in.world)
    {
        host_data.scalars.num_volume_levels = calc_num_volume_levels(host_data);
    }

    // Precompute unique-instance offsets
    {
        CollectionBuilder offsets_builder{&host_data.unique_instance_offsets};
        auto const num_desc = calc_num_descendants(host_data);
        for (ull_int off : calc_unique_instance_offsets(host_data, num_desc))
        {
            offsets_builder.push_back(off);
        }
        host_data.scalars.num_unique_instances
            = host_data.scalars.world
                  ? num_desc[host_data.scalars.world.unchecked_get()]
                  : 0;
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
