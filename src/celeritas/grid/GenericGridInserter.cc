//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GenericGridInserter.cc
//---------------------------------------------------------------------------//
#include "GenericGridInserter.hh"

#include "celeritas/io/ImportPhysicsVector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
GenericGridInserter::GenericGridInserter(RealCollection* real_data,
                                         GenericGridCollection* grid)
    : grid_builder_(real_data), grids_(grid)
{
    CELER_EXPECT(real_data && grid);
}

auto GenericGridInserter::operator()(ImportedPhysicsVector const& vec) -> GenericIndex
{
    if (vec.x.empty())
        return GenericIndex{};

    return grids_.push_back(grid_builder_(vec));
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
