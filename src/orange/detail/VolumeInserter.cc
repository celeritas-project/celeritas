//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/VolumeInserter.cc
//---------------------------------------------------------------------------//
#include "VolumeInserter.hh"

#include <algorithm>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/Ref.hh"
#include "orange/Data.hh"
#include "orange/Types.hh"
#include "orange/construct/OrangeInput.hh"
#include "orange/surf/SurfaceAction.hh"
#include "orange/surf/Surfaces.hh"

namespace celeritas
{
namespace detail
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
//! Return a surface's "simple" flag
struct SimpleSafetyGetter
{
    template<class S>
    constexpr bool operator()(const S&) const noexcept
    {
        return S::simple_safety();
    }
};

//---------------------------------------------------------------------------//
//! Return the number of intersections for a surface
struct NumIntersectionGetter
{
    template<class S>
    constexpr size_type operator()(const S&) const noexcept
    {
        using Intersections = typename S::Intersections;
        return Intersections{}.size();
    }
};

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to empty volume data.
 */
VolumeInserter::VolumeInserter(Data* orange_data) : orange_data_(orange_data)
{
    CELER_EXPECT(orange_data_ && orange_data_->volumes.defs.empty());
    surfaces_ = make_const_ref(orange_data->surfaces);
}

//---------------------------------------------------------------------------//
/*!
 * Insert a volume.
 *
 * TODO: build surface connectivity
 */
VolumeId VolumeInserter::operator()(const VolumeInput& input)
{
    CELER_EXPECT(input);
    CELER_EXPECT(std::is_sorted(input.faces.begin(), input.faces.end()));
    CELER_EXPECT(input.faces.empty()
                 || input.faces.back() < orange_data_->surfaces.size());

    VolumeId::size_type new_id = orange_data_->volumes.defs.size();

    // Calculate the maximum stack depth of the volume definition
    int this_max_depth = calc_max_depth(make_span(input.logic));
    CELER_VALIDATE(this_max_depth > 0,
                   << "invalid logic definition in volume " << new_id
                   << ": operators do not balance");
    max_logic_depth_ = std::max(max_logic_depth_, this_max_depth);

    // Mark as 'simple safety' if all the surfaces are simple
    bool      simple_safety     = true;
    logic_int max_intersections = 0;

    Surfaces::Reals reals;
    reals = orange_data_->reals;
    Surfaces surfaces{surfaces_, reals};
    auto     get_simple_safety
        = make_surface_action(surfaces, SimpleSafetyGetter{});
    auto get_num_intersections
        = make_surface_action(surfaces, NumIntersectionGetter{});

    for (SurfaceId sid : input.faces)
    {
        CELER_ASSERT(sid < surfaces.num_surfaces());
        simple_safety = simple_safety && get_simple_safety(sid);
        max_intersections += get_num_intersections(sid);
    }

    auto defs  = make_builder(&orange_data_->volumes.defs);
    auto faces = make_builder(&orange_data_->volumes.faces);
    auto logic = make_builder(&orange_data_->volumes.logic);

    VolumeRecord output;
    output.faces = faces.insert_back(input.faces.begin(), input.faces.end());
    output.logic = logic.insert_back(input.logic.begin(), input.logic.end());
    output.max_intersections = max_intersections;
    output.flags             = input.flags;
    if (simple_safety)
    {
        output.flags |= VolumeRecord::Flags::simple_safety;
    }
    defs.push_back(output);

    // NOTE: input max intersections are checked but might be removed later
    CELER_ENSURE(output.max_intersections == input.max_intersections
                 || input.max_intersections == 0);
    CELER_ENSURE(defs.size() == new_id + 1);
    return VolumeId{new_id};
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
