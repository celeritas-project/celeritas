//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/ScintPreGenAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Macros.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/optical/OpticalDistributionData.hh"

#include "OpticalGenParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ScintillationParams;

namespace detail
{
struct OpticalGenStorage;
//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 */
class ScintPreGenAction final : public ExplicitCoreActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstScintillation = std::shared_ptr<ScintillationParams const>;
    using SPGenStorage = std::shared_ptr<detail::OpticalGenStorage>;
    //!@}

  public:
    // Construct with action ID, optical properties, and storage
    ScintPreGenAction(ActionId id,
                      AuxId data_id,
                      SPConstScintillation scintillation);

    // Launch kernel with host data
    void execute(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void execute(CoreParams const&, CoreStateDevice&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    //! Short name for the action
    std::string_view label() const final
    {
        return "scintillation-pre-generator";
    }

    // Name of the action (for user output)
    std::string_view description() const final;

    //! Dependency ordering of the action
    ActionOrder order() const final { return ActionOrder::post_post; }

  private:
    //// DATA ////

    ActionId id_;
    AuxId data_id_;
    SPConstScintillation scintillation_;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    void execute_impl(CoreParams const&, CoreState<M>&) const;

    void pre_generate(CoreParams const&, CoreStateHost&) const;
    void pre_generate(CoreParams const&, CoreStateDevice&) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
