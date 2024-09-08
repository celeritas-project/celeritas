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
class MaterialParams;
}  // namespace optical

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 */
class CerenkovOffloadAction final : public CoreStepActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstCerenkov
        = std::shared_ptr<celeritas::optical::CerenkovParams const>;
    using SPConstMaterial
        = std::shared_ptr<celeritas::optical::MaterialParams const>;
    //!@}

  public:
    // Construct with action ID, optical material, and storage
    CerenkovOffloadAction(ActionId id,
                          AuxId data_id,
                          SPConstMaterial material,
                          SPConstCerenkov cerenkov);

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    //! Short name for the action
    std::string_view label() const final { return "cerenkov-offload"; }

    // Name of the action (for user output)
    std::string_view description() const final;

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::user_post; }

  private:
    //// DATA ////

    ActionId id_;
    AuxId data_id_;
    SPConstMaterial material_;
    SPConstCerenkov cerenkov_;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    void step_impl(CoreParams const&, CoreState<M>&) const;

    void pre_generate(CoreParams const&, CoreStateHost&) const;
    void pre_generate(CoreParams const&, CoreStateDevice&) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
