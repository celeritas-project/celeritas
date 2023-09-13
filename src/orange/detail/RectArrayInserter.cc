//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/RectArrayInserter.cc
//---------------------------------------------------------------------------//
#include "RectArrayInserter.hh"

#include <algorithm>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "orange/construct/OrangeInput.hh"

namespace celeritas
{
namespace detail
{

//---------------------------------------------------------------------------//
/*!
 * Construct from full parameter data.
 */
RectArrayInserter::RectArrayInserter(Data* orange_data)
    : orange_data_(orange_data)
    , insert_transform_{&orange_data_->transforms, &orange_data_->reals}
{
    CELER_EXPECT(orange_data);
}

//---------------------------------------------------------------------------//
/*!
 * Create a rect array unit and return its ID.
 */
RectArrayId RectArrayInserter::operator()(RectArrayInput const& inp)
{
    RectArrayRecord record;
    RectArrayRecord::SurfaceIndexerData::Sizes sizes;

    auto reals_builder = make_builder(&orange_data_->reals);
    size_type num_volumes = 1;

    for (auto ax : range(Axis::size_))
    {
        std::vector<double> const& grid = inp.grid[to_int(ax)];
        CELER_VALIDATE(grid.size() >= 2,
                       << "grid for " << to_char(ax) << " axis in '"
                       << inp.label << "' is too small (size " << grid.size()
                       << ")");
        CELER_VALIDATE(std::is_sorted(grid.begin(), grid.end()),
                       << "grid for " << to_char(ax) << " axis in '"
                       << inp.label << "' is not monotonically increasing");

        sizes[to_int(ax)] = grid.size();
        record.dims[to_int(ax)] = grid.size() - 1;
        num_volumes *= grid.size() - 1;

        record.grid[to_int(ax)]
            = reals_builder.insert_back(grid.begin(), grid.end());
    }

    record.surface_indexer_data
        = RectArrayRecord::SurfaceIndexerData::from_sizes(sizes);

    CELER_VALIDATE(inp.daughters.size() == num_volumes,
                   << "number of input daughters (" << inp.daughters.size()
                   << ") in '" << inp.label
                   << "' does not match number of volumes (" << num_volumes
                   << ")");

    std::vector<Daughter> daughters;
    for (auto const& daughter_input : inp.daughters)
    {
        Daughter d;
        d.universe_id = daughter_input.universe_id;
        d.transform_id = insert_transform_(daughter_input.translation);
        daughters.push_back(d);
    }

    record.daughters = ItemMap<LocalVolumeId, DaughterId>(
        make_builder(&orange_data_->daughters)
            .insert_back(daughters.begin(), daughters.end()));

    CELER_ASSERT(record);
    return RectArrayId(
        make_builder(&orange_data_->rect_arrays).push_back(record).get());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
