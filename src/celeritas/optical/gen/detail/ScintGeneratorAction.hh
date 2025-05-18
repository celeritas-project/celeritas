//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/ScintGeneratorAction.hh
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
class OffloadParams;
class ScintillationParams;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Generate scintillation photons from optical distribution data.
 *
 * This samples and buffers new optical track initializers in a reproducible
 * way. Rather than let each thread generate all initializers from one
 * distribution, the work is split as evenly as possible among threads:
 * multiple threads may generate initializers from a single distribution.
 */
class ScintGeneratorAction final : public CoreStepActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstScintillation = std::shared_ptr<ScintillationParams const>;
    using SPOffloadParams = std::shared_ptr<OffloadParams>;
    //!@}

  public:
    // Construct with action ID, data IDs, and optical properties
    ScintGeneratorAction(ActionId id,
                         AuxId offload_id,
                         AuxId optical_id,
                         SPConstScintillation scintillation,
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
        return "generate-scintillation-photons";
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
    SPConstScintillation scintillation_;
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
