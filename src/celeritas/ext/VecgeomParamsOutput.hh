//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/VecgeomParamsOutput.hh
//---------------------------------------------------------------------------//
#pragma once
#include <memory>

#include "corecel/io/OutputInterface.hh"

namespace celeritas
{
class VecgeomParams;
//---------------------------------------------------------------------------//
/*!
 * Save extra debugging information about the VecGeom geometry.
 *
 * This is to be used in *addition* to the standard bbox/volume/surface data
 * saved by GeoParamsOutput.
 *
 * \sa celeritas/geo/GeoParamsOutput.hh
 */
class VecgeomParamsOutput final : public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstVecgeomParams = std::shared_ptr<VecgeomParams const>;
    //!@}

  public:
    // Construct from shared geometry data
    explicit VecgeomParamsOutput(SPConstVecgeomParams vecgeom);

    //! Category of data to write
    Category category() const final { return Category::internal; }

    //! Name of the entry inside the category.
    std::string label() const final { return "vecgeom"; }

    // Write output to the given JSON object
    void output(JsonPimpl*) const final;

  private:
    SPConstVecgeomParams vecgeom_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
