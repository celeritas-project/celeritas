//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VolumeInserter.cc
//---------------------------------------------------------------------------//
#include "VolumeInserter.hh"

#include <algorithm>
#include <vector>

#include "base/Assert.hh"
#include "base/Collection.hh"
#include "base/CollectionBuilder.hh"
#include "base/OpaqueId.hh"
#include "base/Span.hh"
#include "orange/Data.hh"
#include "orange/Types.hh"
#include "orange/construct/VolumeInput.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the maximum logic depth of a volume definition.
 *
 * Return 0 if the definition is invalid so that we can raise an assertion in
 * the caller with more context.
 */
int calc_max_depth(Span<const logic_int> logic)
{
    CELER_EXPECT(!logic.empty());
    // Calculate max depth
    int max_depth = 1;
    int cur_depth = 0;

    for (auto id : logic)
    {
        if (!logic::is_operator_token(id) || id == logic::ltrue)
        {
            ++cur_depth;
        }
        else if (id == logic::land || id == logic::lor)
        {
            max_depth = std::max(cur_depth, max_depth);
            --cur_depth;
        }
    }
    if (cur_depth != 1)
    {
        // Input definition is invalid; return a sentinel value
        max_depth = 0;
    }
    return max_depth;
}
//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to empty volume data.
 */
VolumeInserter::VolumeInserter(Data* volumes) : volume_data_(volumes)
{
    CELER_EXPECT(volume_data_ && volume_data_->defs.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Insert a volume.
 *
 * TODO: add consistancy checks with number of surfaces?
 */
VolumeId VolumeInserter::operator()(const VolumeInput& input)
{
    CELER_EXPECT(input);
    CELER_EXPECT(std::is_sorted(input.faces.begin(), input.faces.end()));

    VolumeId::size_type new_id = volume_data_->defs.size();

    // Calculate the maximum stack depth of the volume definition
    int this_max_depth = calc_max_depth(make_span(input.logic));
    CELER_VALIDATE(this_max_depth > 0,
                   << "invalid logic definition in volume " << new_id
                   << ": operators do not balance");
    max_logic_depth_ = std::max(max_logic_depth_, this_max_depth);

    auto defs  = make_builder(&volume_data_->defs);
    auto faces = make_builder(&volume_data_->faces);
    auto logic = make_builder(&volume_data_->logic);

    VolumeRecord output;
    output.faces = faces.insert_back(input.faces.begin(), input.faces.end());
    output.logic = logic.insert_back(input.logic.begin(), input.logic.end());
    output.num_intersections = input.num_intersections;
    output.flags             = input.flags;
    defs.push_back(output);

    CELER_ENSURE(defs.size() == new_id + 1);
    return VolumeId{new_id};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
