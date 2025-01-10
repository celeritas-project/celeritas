//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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

#include "UniverseInserter.hh"
#include "../OrangeInput.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
//! Return correctly sized volume labels
std::vector<Label> make_volume_labels(RectArrayInput const& inp)
{
    std::vector<Label> result;
    for (auto i : range(inp.grid[to_int(Axis::x)].size() - 1))
    {
        for (auto j : range(inp.grid[to_int(Axis::y)].size() - 1))
        {
            for (auto k : range(inp.grid[to_int(Axis::z)].size() - 1))
            {
                Label vl;
                vl.name = std::string("{" + std::to_string(i) + ","
                                      + std::to_string(j) + ","
                                      + std::to_string(k) + "}");
                vl.ext = inp.label.name;
                result.push_back(std::move(vl));
            }
        }
    }

    CELER_ENSURE(result.size() == inp.daughters.size());
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from full parameter data.
 */
RectArrayInserter::RectArrayInserter(UniverseInserter* insert_universe,
                                     Data* orange_data)
    : orange_data_(orange_data)
    , insert_transform_{&orange_data_->transforms, &orange_data_->reals}
    , insert_universe_{insert_universe}
    , rect_arrays_{&orange_data_->rect_arrays}
    , reals_{&orange_data_->reals}
    , daughters_{&orange_data_->daughters}
{
    CELER_EXPECT(orange_data);
}

//---------------------------------------------------------------------------//
/*!
 * Create a rect array unit and return its ID.
 */
UniverseId RectArrayInserter::operator()(RectArrayInput const& inp)
{
    CELER_VALIDATE(
        inp, << "rect array '" << inp.label << "' is not properly constructed");

    RectArrayRecord record;
    RectArrayRecord::SurfaceIndexerData::Sizes sizes;

    std::vector<Label> surface_labels;
    size_type num_volumes = 1;

    for (auto ax : range(Axis::size_))
    {
        std::vector<double> grid = inp.grid[to_int(ax)];
        CELER_VALIDATE(grid.size() >= 2,
                       << "grid for " << to_char(ax) << " axis in '"
                       << inp.label << "' is too small (size " << grid.size()
                       << ")");
        CELER_VALIDATE(std::is_sorted(grid.begin(), grid.end()),
                       << "grid for " << to_char(ax) << " axis in '"
                       << inp.label << "' is not monotonically increasing");

        // Suppress the outer grid boundaries to avoid coincident surfaces with
        // other universes
        grid.front() = -std::numeric_limits<real_type>::infinity();
        grid.back() = std::numeric_limits<real_type>::infinity();

        sizes[to_int(ax)] = grid.size();
        record.dims[to_int(ax)] = grid.size() - 1;
        num_volumes *= grid.size() - 1;

        record.grid[to_int(ax)] = reals_.insert_back(grid.begin(), grid.end());

        // Create surface labels
        for (auto i : range(inp.grid[to_int(ax)].size()))
        {
            Label sl;
            sl.name = std::string("{" + std::string(1, to_char(ax)) + ","
                                  + std::to_string(i) + "}");
            sl.ext = inp.label.name;
            surface_labels.push_back(std::move(sl));
        }
    }

    record.surface_indexer_data
        = RectArrayRecord::SurfaceIndexerData::from_sizes(sizes);

    CELER_VALIDATE(inp.daughters.size() == num_volumes,
                   << "number of input daughters (" << inp.daughters.size()
                   << ") in '" << inp.label
                   << "' does not match number of volumes (" << num_volumes
                   << ")");

    // Construct daughters
    std::vector<Daughter> daughters;
    for (auto const& daughter_input : inp.daughters)
    {
        Daughter d;
        d.universe_id = daughter_input.universe_id;
        d.trans_id = insert_transform_(daughter_input.transform);
        daughters.push_back(d);
    }
    record.daughters = ItemMap<LocalVolumeId, DaughterId>(
        daughters_.insert_back(daughters.begin(), daughters.end()));

    // Add rect array record
    CELER_ASSERT(record);
    rect_arrays_.push_back(record);

    // Construct universe
    return (*insert_universe_)(UniverseType::rect_array,
                               inp.label,
                               std::move(surface_labels),
                               make_volume_labels(inp));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
