//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/MaterialParamsOutput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/io/OutputInterface.hh"

namespace celeritas
{
class MaterialParams;
//---------------------------------------------------------------------------//
/*!
 * Save material diagnostic data.
 */
class MaterialParamsOutput final : public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstMaterialParams = std::shared_ptr<MaterialParams const>;
    //!@}

  public:
    // Construct from shared material data
    explicit MaterialParamsOutput(SPConstMaterialParams material);

    //! Category of data to write
    Category category() const final { return Category::internal; }

    //! Name of the entry inside the category.
    std::string label() const final { return "material"; }

    // Write output to the given JSON object
    void output(JsonPimpl*) const final;

  private:
    SPConstMaterialParams material_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
