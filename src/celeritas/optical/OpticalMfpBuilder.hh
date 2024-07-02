//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalMfpBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Builder used by optical models to construct mean free path grids in the
 * OpticalPhysicsParams data.
 *
 * This builder is used by OpticalMfpBuilder and is meant to store all of the
 * MFP grids of all optical models for a specific optical material. 
 */
class OpticalModelMfpBuilder
{
  public:
    OpticalModelMfpBuilder(GenericGridBuilder* build_grid, OpticalMaterialId optical_material)
        : build_grid_(*build_grid), optical_material_(optical_material)
    {
        CELER_EXPECT(build_grid_);
        CELER_EXPECT(optical_material_);
    }

    void operator()(ImportPhysicsVector const& mfp)
    {
        grids_.push_back((*build_grid_)(mfp));
    }

    OpticalMaterialId optical_material() const
    {
        return optical_material_;
    }

    std::vector<OpticalValueGrid> const& grids() const
    {
        return grids_;
    }

  private:
    GenericGridBuilder& build_grid_;
    OpticalMaterialId optical_material_;
    std::vector<OpticalValueGrid> grids_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
