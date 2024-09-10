//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/CerenkovGeneratorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Macros.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/optical/GeneratorDistributionData.hh"

namespace celeritas
{
namespace optical
{
class CerenkovParams;
class MaterialParams;
}  // namespace optical

namespace detail
{
class OffloadParams;
//---------------------------------------------------------------------------//
/*!
 * Generate Cerenkov photons from optical distribution data.
 *
 * This samples and buffers new optical primaries in a reproducible way. Rather
 * than let each thread generate all primaries from one distribution, the work
 * is split as evenly as possible among threads: multiple threads may generate
 * primaries from a single distribution.
 */
class CerenkovGeneratorAction final : public CoreStepActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstCerenkov
        = std::shared_ptr<celeritas::optical::CerenkovParams const>;
    using SPConstMaterial
        = std::shared_ptr<celeritas::optical::MaterialParams const>;
    using SPOffloadParams = std::shared_ptr<OffloadParams>;
    //!@}

  public:
    // Construct with action ID, data IDs, and optical properties
    CerenkovGeneratorAction(ActionId id,
                            AuxId offload_id,
                            AuxId optical_id,
                            SPConstMaterial material,
                            SPConstCerenkov cerenkov,
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
        return "generate-cerenkov-photons";
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
    SPConstCerenkov cerenkov_;
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
