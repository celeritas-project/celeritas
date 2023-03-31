//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/RectArrayInserter.cc
//---------------------------------------------------------------------------//
#include "RectArrayInserter.hh"

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
{
    CELER_EXPECT(orange_data);
}

//---------------------------------------------------------------------------//
/*!
 * Create a simple unit and return its ID.
 */
RectArrayId RectArrayInserter::operator()(RectArrayInput const& inp)
{
    RectArrayRecord rect_array;

    for (auto i : range(3))
    {
        rect_array.grid[i]
            = make_builder(&orange_data_->reals)
                  .insert_back(inp.grid[i].begin(), inp.grid[i].end());
    }

    std::vector<Daughter> daughters;
    for (auto const& daughter_input : inp.daughters)
    {
        Daughter d;
        d.universe_id = daughter_input.universe_id;
        d.translation_id = make_builder(&orange_data_->translations)
                               .push_back(daughter_input.translation);
        daughters.push_back(d);
    }

    rect_array.daughters = ItemMap<LocalVolumeId, DaughterId>(
        make_builder(&orange_data_->daughters)
            .insert_back(daughters.begin(), daughters.end()));

    CELER_ASSERT(rect_array);
    return RectArrayId(
        make_builder(&orange_data_->rect_arrays).push_back(rect_array).get());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
