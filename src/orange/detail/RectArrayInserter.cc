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
    CELER_EXPECT(std::all_of(inp.grid.begin(),
                             inp.grid.end(),
                             [](auto const& g) { return g.size() >= 2; }));
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
 * Number of surfaces created by the input.
 */
std::size_t RectArrayInserter::num_surfaces(Input const& i)
{
    std::size_t result{0};
    for (auto ax : range(Axis::size_))
    {
        result += i.grid[to_int(ax)].size();
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Number of volumes created by the input.
 */
std::size_t RectArrayInserter::num_volumes(Input const& i)
{
    std::size_t result{1};
    for (auto ax : range(Axis::size_))
    {
        result *= i.grid[to_int(ax)].size() - 1;
    }
    return result;
}

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
UnivId RectArrayInserter::operator()(RectArrayInput&& inp)
{
    CELER_VALIDATE(
        inp, << "rect array '" << inp.label << "' is not properly constructed");

    RectArrayRecord record;
    RectArrayRecord::SurfaceIndexerData::Sizes sizes;

    std::vector<Label> surface_labels;
    size_type num_volumes = 1;

    for (auto ax : range(Axis::size_))
    {
        auto& grid = inp.grid[to_int(ax)];
        CELER_VALIDATE(grid.size() >= 2,
                       << "grid for " << to_char(ax) << " axis in '"
                       << inp.label << "' is too small (size " << grid.size()
                       << ")");
        CELER_VALIDATE(std::is_sorted(grid.begin(), grid.end()),
                       << "grid for " << to_char(ax) << " axis in '"
                       << inp.label << "' is not monotonically increasing");

        // Suppress the outer grid boundaries to avoid coincident surfaces with
        // other universes
        // FIXME: replace with bump using orange_data.scalars.tol
        grid.front() = -std::numeric_limits<real_type>::infinity();
        grid.back() = std::numeric_limits<real_type>::infinity();

        sizes[to_int(ax)] = grid.size();
        record.dims[to_int(ax)] = grid.size() - 1;
        num_volumes *= grid.size() - 1;

        record.grid[to_int(ax)] = reals_.insert_back(grid.begin(), grid.end());

        // Create surface labels
        for (auto i : range(grid.size()))
        {
            auto name = std::string("{" + std::string(1, to_char(ax)) + ","
                                    + std::to_string(i) + "}");
            surface_labels.emplace_back(std::move(name), inp.label.name);
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
        d.univ_id = daughter_input.univ_id;
        d.trans_id = insert_transform_(daughter_input.transform);
        daughters.push_back(d);
    }
    record.daughters = ItemMap<LocalVolumeId, DaughterId>(
        daughters_.insert_back(daughters.begin(), daughters.end()));

    // Add rect array record
    CELER_ASSERT(record);
    rect_arrays_.push_back(record);

    // Save labels and clear input
    auto vol_labels = make_volume_labels(inp);
    Label unit_label{std::move(inp.label)};
    std::move(inp) = {};

    // Construct universe
    return (*insert_universe_)(UnivType::rect_array,
                               std::move(unit_label),
                               std::move(surface_labels),
                               std::move(vol_labels));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
