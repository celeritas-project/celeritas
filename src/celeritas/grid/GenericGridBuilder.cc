//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GenericGridBuilder.cc
//---------------------------------------------------------------------------//
#include "GenericGridBuilder.hh"

#include "celeritas/io/ImportPhysicsVector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with pointers to data that will be modified.
 */
GenericGridBuilder::GenericGridBuilder(Items<real_type>* reals) : reals_{reals}
{
    CELER_EXPECT(reals);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of generic data with linear interpolation.
 */
auto GenericGridBuilder::operator()(SpanConstReal grid, SpanConstReal values)
    -> Grid
{
    CELER_EXPECT(grid.size() >= 2);
    CELER_EXPECT(grid.front() <= grid.back());
    CELER_EXPECT(values.size() == grid.size());

    Grid result;
    result.grid = reals_.insert_back(grid.begin(), grid.end());
    result.value = reals_.insert_back(values.begin(), values.end());

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid from an imported physics vector.
 */
auto GenericGridBuilder::operator()(ImportPhysicsVector const& pvec) -> Grid
{
    CELER_EXPECT(pvec.vector_type == ImportPhysicsVectorType::free);
    return (*this)(make_span(pvec.x), make_span(pvec.y));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
