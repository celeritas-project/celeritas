//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/CherenkovGeneratorAction.hh
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
namespace optical
{
class CherenkovParams;
class MaterialParams;
}  // namespace optical

namespace detail
{
class OffloadParams;
//---------------------------------------------------------------------------//
/*!
 * Generate Cherenkov photons from optical distribution data.
 *
 * This samples and buffers new optical track initializers in a reproducible
 * way. Rather than let each thread generate all initializers from one
 * distribution, the work is split as evenly as possible among threads:
 * multiple threads may generate initializers from a single distribution.
 */
class CherenkovGeneratorAction final : public CoreStepActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstCherenkov
        = std::shared_ptr<celeritas::optical::CherenkovParams const>;
    using SPConstMaterial
        = std::shared_ptr<celeritas::optical::MaterialParams const>;
    using SPOffloadParams = std::shared_ptr<OffloadParams>;
    //!@}

  public:
    // Construct with action ID, data IDs, and optical properties
    CherenkovGeneratorAction(ActionId id,
                             AuxId offload_id,
                             AuxId optical_id,
                             SPConstMaterial material,
                             SPConstCherenkov cherenkov,
                             size_type auto_flush);

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    //! Short name for the action
    std::string_view label() const final
    {
        return "generate-cherenkov-photons";
    }

    // Name of the action (for user output)
    std::string_view description() const final;

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::user_post; }

  private:
    //// DATA ////

    ActionId id_;
    AuxId offload_id_;
    AuxId optical_id_;
    SPConstMaterial material_;
    SPConstCherenkov cherenkov_;
    size_type auto_flush_;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    void step_impl(CoreParams const&, CoreState<M>&) const;

    void generate(CoreParams const&, CoreStateHost&) const;
    void generate(CoreParams const&, CoreStateDevice&) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
