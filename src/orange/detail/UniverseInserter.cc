//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UniverseInserter.cc
//---------------------------------------------------------------------------//
#include "UniverseInserter.hh"

#include <algorithm>
#include <iterator>
#include <numeric>

#include "geocel/VolumeParams.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
template<class T, class V>
T get_or_default(V const& vt, T d = {})
{
    if (!std::holds_alternative<T>(vt))
    {
        return d;
    }
    return std::get<T>(vt);
}

//---------------------------------------------------------------------------//
void move_back(std::vector<Label>& dst, std::vector<Label>&& src)
{
    dst.insert(dst.end(),
               std::make_move_iterator(src.begin()),
               std::make_move_iterator(src.end()));
    std::move(src).clear();
}

//---------------------------------------------------------------------------//
void move_back(std::vector<Label>& dst,
               std::vector<VolumeInput::VariantLabel>&& src)
{
    dst.reserve(dst.size() + src.size());
    for (auto& label : src)
    {
        CELER_ASSUME(std::holds_alternative<Label>(label));
        dst.push_back(std::move(std::get<Label>(label)));
    }

    std::move(src).clear();
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Initialize with metadata and data.
 *
 * Push back initial zeros on construction.
 */
UniverseInserter::UniverseInserter(VolumeParams const& volume_params,
                                   VecLabel* universe_labels,
                                   VecLabel* surface_labels,
                                   VecLabel* volume_labels,
                                   Data* data)
    : volume_params_{volume_params}
    , universe_labels_{universe_labels}
    , surface_labels_{surface_labels}
    , volume_labels_{volume_labels}
    , types_(&data->universe_types)
    , indices_(&data->universe_indices)
    , surfaces_{&data->universe_indexer_data.surfaces}
    , volumes_{&data->universe_indexer_data.volumes}
    , volume_ids_{&data->volume_ids}
    , volume_instance_ids_{&data->volume_instance_ids}
{
    CELER_EXPECT(universe_labels_ && surface_labels_ && volume_labels_ && data);
    CELER_EXPECT(types_.size() == 0);
    CELER_EXPECT(surfaces_.size() == 0);

    std::fill(num_universe_types_.begin(), num_universe_types_.end(), 0u);

    // Add initial zero offset for universe indexer
    surfaces_.push_back(accum_surface_);
    volumes_.push_back(accum_volume_);
}

//---------------------------------------------------------------------------//
/*!
 * Accumulate the number of local surfaces and volumes.
 */
UniverseId UniverseInserter::operator()(UniverseType type,
                                        Label univ_label,
                                        VecLabel surface_labels,
                                        VecLabel volume_labels)
{
    CELER_EXPECT(type != UniverseType::size_);
    CELER_EXPECT(!volume_labels.empty());

    UniverseId result = this->update_counters(
        type, surface_labels.size(), volume_labels.size());

    // Append metadata
    universe_labels_->push_back(std::move(univ_label));
    move_back(*surface_labels_, std::move(surface_labels));
    move_back(*volume_labels_, std::move(volume_labels));

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Accumulate the number of local surfaces and volumes.
 */
UniverseId UniverseInserter::operator()(UniverseType type,
                                        Label univ_label,
                                        VecLabel surface_labels,
                                        VecVolLabel volume_labels)
{
    CELER_EXPECT(type != UniverseType::size_);
    CELER_EXPECT(!volume_labels.empty());
    CELER_EXPECT(
        !volume_params_.empty()
        || std::all_of(
            volume_labels.begin(), volume_labels.end(), [](auto& varlabel) {
                return std::holds_alternative<Label>(varlabel);
            }));

    UniverseId result = this->update_counters(
        type, surface_labels.size(), volume_labels.size());

    if (!volume_params_.empty())
    {
        volume_ids_.reserve(volume_ids_.size() + volume_labels.size());
        volume_instance_ids_.reserve(volume_ids_.size());

        // Add volume IDs and instance IDs
        for (auto& var_label : volume_labels)
        {
            if (auto vi_id = get_or_default<VolumeInstanceId>(var_label))
            {
                CELER_ASSERT(vi_id < volume_params_.num_volume_instances());
                auto vol_id = volume_params_.volume(vi_id);
                volume_ids_.push_back(vol_id);
                volume_instance_ids_.push_back(vi_id);

                // Replace with volume label corresponding to it
                // TODO: should be volume instance label or also something else
                // specific to this implementation
                var_label = volume_params_.volume_labels().at(vol_id);
            }
            else
            {
                // No physical volume corresponds to it
                volume_ids_.push_back(VolumeId{});
                volume_instance_ids_.push_back(VolumeInstanceId{});
            }
        }
    }

    // Append metadata
    universe_labels_->push_back(std::move(univ_label));
    move_back(*surface_labels_, std::move(surface_labels));
    move_back(*volume_labels_, std::move(volume_labels));

    return result;
}

UniverseId UniverseInserter::update_counters(UniverseType type,
                                             size_type num_surfaces,
                                             size_type num_volumes)
{
    UniverseId result = types_.size_id();

    // Add universe type and index
    types_.push_back(type);
    indices_.push_back(num_universe_types_[type]++);

    // Accumulate and append surface/volumes to universe indexer
    accum_surface_ += num_surfaces;
    accum_volume_ += num_volumes;
    surfaces_.push_back(accum_surface_);
    volumes_.push_back(accum_volume_);

    CELER_ENSURE(std::accumulate(num_universe_types_.begin(),
                                 num_universe_types_.end(),
                                 size_type(0))
                 == types_.size());
    CELER_ENSURE(surfaces_.size() == types_.size() + 1);
    CELER_ENSURE(volumes_.size() == surfaces_.size());
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
