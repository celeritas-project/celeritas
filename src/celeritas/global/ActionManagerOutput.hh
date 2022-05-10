//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionManagerOutput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/io/OutputInterface.hh"

namespace celeritas
{
class ActionManager;
//---------------------------------------------------------------------------//
/*!
 * Save action manager data.
 */
class ActionManagerOutput final : public OutputInterface
{
  public:
    //!@{
    //! Type aliases
    using SPConstActionManager = std::shared_ptr<const ActionManager>;
    //!@}

  public:
    // Construct from a shared action manager
    explicit ActionManagerOutput(SPConstActionManager actions);

    //! Category of data to write
    Category category() const final { return Category::internal; }

    //! Name of the entry inside the category.
    std::string label() const final { return "actions"; }

    // Write output to the given JSON object
    void output(JsonPimpl*) const final;

  private:
    SPConstActionManager actions_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
