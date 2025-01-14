//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoParamsOutput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/io/OutputInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class GeoParamsInterface;

//---------------------------------------------------------------------------//
/*!
 * Save geometry diagnostic data.
 */
class GeoParamsOutput final : public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstGeoParams = std::shared_ptr<GeoParamsInterface const>;
    //!@}

  public:
    // Construct from shared geometry data
    explicit GeoParamsOutput(SPConstGeoParams geo);

    //! Category of data to write
    Category category() const final { return Category::internal; }

    //! Name of the entry inside the category.
    std::string_view label() const final { return "geometry"; }

    // Write output to the given JSON object
    void output(JsonPimpl*) const final;

  private:
    SPConstGeoParams geo_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
