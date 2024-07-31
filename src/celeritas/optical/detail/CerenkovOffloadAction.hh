//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/CerenkovOffloadAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Macros.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/optical/GeneratorDistributionData.hh"

#include "OffloadParams.hh"

namespace celeritas
{
namespace optical
{
class CerenkovParams;
class MaterialPropertyParams;
}  // namespace optical

namespace detail
{
struct OpticalGenStorage;
//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 */
class CerenkovOffloadAction final : public ExplicitCoreActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstCerenkov
        = std::shared_ptr<celeritas::optical::CerenkovParams const>;
    using SPConstProperties
        = std::shared_ptr<celeritas::optical::MaterialPropertyParams const>;
    using SPGenStorage = std::shared_ptr<detail::OpticalGenStorage>;
    //!@}

  public:
    // Construct with action ID, optical properties, and storage
    CerenkovOffloadAction(ActionId id,
                          AuxId data_id,
                          SPConstProperties properties,
                          SPConstCerenkov cerenkov);

    // Launch kernel with host data
    void execute(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void execute(CoreParams const&, CoreStateDevice&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    //! Short name for the action
    std::string_view label() const final { return "cerenkov-offload"; }

    // Name of the action (for user output)
    std::string_view description() const final;

    //! Dependency ordering of the action
    ActionOrder order() const final { return ActionOrder::user_post; }

  private:
    //// DATA ////

    ActionId id_;
    AuxId data_id_;
    SPConstProperties properties_;
    SPConstCerenkov cerenkov_;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    void execute_impl(CoreParams const&, CoreState<M>&) const;

    void pre_generate(CoreParams const&, CoreStateHost&) const;
    void pre_generate(CoreParams const&, CoreStateDevice&) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
