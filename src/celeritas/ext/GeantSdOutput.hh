//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantSdOutput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/io/OutputInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class GeantSd;

//---------------------------------------------------------------------------//
/*!
 * Save debugging information about sensitive detector mappings.
 */
class GeantSdOutput final : public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstGeantSd = std::shared_ptr<GeantSd const>;
    //!@}

  public:
    // Construct from shared hit manager
    explicit GeantSdOutput(SPConstGeantSd hits);

    //! Category of data to write
    Category category() const final { return Category::internal; }

    //! Name of the entry inside the category.
    std::string_view label() const final { return "hit-manager"; }

    // Write output to the given JSON object
    void output(JsonPimpl*) const final;

  private:
    SPConstGeantSd hits_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
