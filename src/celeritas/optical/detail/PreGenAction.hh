//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/PreGenAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Macros.hh"
#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
namespace detail
{
struct GenStorage;
//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data at the end of a step.
 */
class PreGenAction final : public ExplicitCoreActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPGenStorage = std::shared_ptr<detail::GenStorage>;
    //!@}

  public:
    // Construct with action ID and storage
    PreGenAction(ActionId id, SPGenStorage storage);

    // Launch kernel with host data
    void execute(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void execute(CoreParams const&, CoreStateDevice&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    //! Short name for the action
    std::string label() const final { return "pre-generator"; }

    // Name of the action (for user output)
    std::string description() const final;

    //! Dependency ordering of the action
    ActionOrder order() const final { return ActionOrder::post_post; }

  private:
    //// DATA ////

    ActionId id_;
    SPGenStorage storage_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
