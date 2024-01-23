//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionRegistryOutput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/io/OutputInterface.hh"

namespace celeritas
{
class ActionRegistry;
//---------------------------------------------------------------------------//
/*!
 * Save action manager data.
 */
class ActionRegistryOutput final : public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstActionRegistry = std::shared_ptr<ActionRegistry const>;
    //!@}

  public:
    // Construct from a shared action manager
    explicit ActionRegistryOutput(SPConstActionRegistry actions);

    //! Category of data to write
    Category category() const final { return Category::internal; }

    //! Name of the entry inside the category.
    std::string label() const final { return "actions"; }

    // Write output to the given JSON object
    void output(JsonPimpl*) const final;

  private:
    SPConstActionRegistry actions_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
