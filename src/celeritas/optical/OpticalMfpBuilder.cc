//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalMfpBuilder.cc
//---------------------------------------------------------------------------//
#include "OpticalMfpBuilder.hh"

#include "celeritas/grid/GenericGridBuilder.hh"
#include "celeritas/io/ImportPhysicsVector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the builder with an existing GenericGridBuilder for a specific
 * optical material.
 */
OpticalModelMfpBuilder::OpticalModelMfpBuilder(
    GenericGridBuilder* build_grid, OpticalMaterialId optical_material)
    : build_grid_(*build_grid), optical_material_(optical_material)
{
    CELER_EXPECT(build_grid);
    CELER_EXPECT(optical_material_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the physics vector as a grid with the GenericGridBuilder.
 *
 * Adds the built grid to the list of grids.
 */
void OpticalModelMfpBuilder::operator()(ImportPhysicsVector const& mfp)
{
    grids_.push_back(build_grid_(mfp));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
