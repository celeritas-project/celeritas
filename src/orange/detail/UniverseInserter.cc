//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UniverseInserter.cc
//---------------------------------------------------------------------------//
#include "UniverseInserter.hh"

#include <algorithm>
#include <iterator>
#include <numeric>

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
template<class T>
void move_back(std::vector<T>& dst, std::vector<T>&& src)
{
    dst.insert(dst.end(),
               std::make_move_iterator(src.begin()),
               std::make_move_iterator(src.end()));
    src.clear();
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Initialize with metadata and data.
 *
 * Push back initial zeros on construction.
 */
UniverseInserter::UniverseInserter(VecLabel* universe_labels,
                                   VecLabel* surface_labels,
                                   VecLabel* volume_labels,
                                   Data* data)
    : universe_labels_{universe_labels}
    , surface_labels_{surface_labels}
    , volume_labels_{volume_labels}
    , types_(&data->universe_types)
    , indices_(&data->universe_indices)
    , surfaces_{&data->universe_indexer_data.surfaces}
    , volumes_{&data->universe_indexer_data.volumes}
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

    UniverseId result = types_.size_id();

    // Add universe type and index
    types_.push_back(type);
    indices_.push_back(num_universe_types_[type]++);

    // Accumulate and append surface/volumes to universe indexer
    accum_surface_ += surface_labels.size();
    accum_volume_ += volume_labels.size();
    surfaces_.push_back(accum_surface_);
    volumes_.push_back(accum_volume_);

    // Append metadata
    universe_labels_->push_back(std::move(univ_label));
    move_back(*surface_labels_, std::move(surface_labels));
    move_back(*volume_labels_, std::move(volume_labels));

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
