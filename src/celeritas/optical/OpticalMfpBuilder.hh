//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalMfpBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "celeritas/grid/GenericGridData.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
class GenericGridBuilder;
class ImportPhysicsVector;
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
    //! Construct builder for a given optical material
    OpticalModelMfpBuilder(GenericGridBuilder* build_grid,
                           OpticalMaterialId optical_material);

    //! Build a grid from the given physics vector
    void operator()(ImportPhysicsVector const& mfp);

    //! Optical material this builder makes grids for
    inline OpticalMaterialId optical_material() const;

    //! Grids built by the builder
    inline std::vector<GenericGridData> const& grids() const;

  private:
    GenericGridBuilder& build_grid_;
    OpticalMaterialId optical_material_;
    std::vector<GenericGridData> grids_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Retrieve the optical material the grids should be built for.
 */
OpticalMaterialId OpticalModelMfpBuilder::optical_material() const
{
    return optical_material_;
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve all of the grids built by this builder so far.
 */
std::vector<GenericGridData> const& OpticalModelMfpBuilder::grids() const
{
    return grids_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
