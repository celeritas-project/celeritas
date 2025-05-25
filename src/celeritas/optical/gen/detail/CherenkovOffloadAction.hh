//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/CherenkovOffloadAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Macros.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/global/ActionInterface.hh"

#include "../GeneratorDistributionData.hh"

namespace celeritas
{
class CherenkovParams;

namespace optical
{
class MaterialParams;
}  // namespace optical

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 */
class CherenkovOffloadAction final : public CoreStepActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstCherenkov = std::shared_ptr<CherenkovParams const>;
    using SPConstMaterial
        = std::shared_ptr<celeritas::optical::MaterialParams const>;
    //!@}

  public:
    // Construct with action ID, optical material, and storage
    CherenkovOffloadAction(ActionId id,
                           AuxId data_id,
                           SPConstMaterial material,
                           SPConstCherenkov cherenkov);

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    //! Short name for the action
    std::string_view label() const final { return "cherenkov-offload"; }

    // Name of the action (for user output)
    std::string_view description() const final;

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::user_post; }

  private:
    //// DATA ////

    ActionId id_;
    AuxId data_id_;
    SPConstMaterial material_;
    SPConstCherenkov cherenkov_;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    void step_impl(CoreParams const&, CoreState<M>&) const;

    void pre_generate(CoreParams const&, CoreStateHost&) const;
    void pre_generate(CoreParams const&, CoreStateDevice&) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
