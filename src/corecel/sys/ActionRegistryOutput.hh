//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ActionRegistryOutput.hh
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

    // Construct from a shared action manager and label
    explicit ActionRegistryOutput(SPConstActionRegistry, std::string);

    //! Category of data to write
    Category category() const final { return Category::internal; }

    //! Name of the entry inside the category.
    std::string_view label() const final { return label_; }

    // Write output to the given JSON object
    void output(JsonPimpl*) const final;

  private:
    SPConstActionRegistry actions_;
    std::string label_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
