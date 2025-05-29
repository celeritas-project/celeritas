//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/GeneratorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Macros.hh"
#include "corecel/data/AuxInterface.hh"
#include "celeritas/global/ActionInterface.hh"

#include "GeneratorTraits.hh"
#include "../GeneratorData.hh"
#include "../OffloadData.hh"

namespace celeritas
{
class CoreParams;

namespace optical
{
class MaterialParams;
}  // namespace optical

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Generate photons from optical distribution data.
 *
 * This samples and buffers new optical track initializers in a reproducible
 * way. Rather than let each thread generate all initializers from one
 * distribution, the work is split as evenly as possible among threads:
 * multiple threads may generate initializers from a single distribution.
 */
template<GeneratorType G>
class GeneratorAction final : public CoreStepActionInterface,
                              public AuxParamsInterface
{
  public:
    //!@{
    //! \name Type aliases
    using TraitsT = GeneratorTraits<G>;
    template<Ownership W, MemSpace M>
    using Data = typename TraitsT::template Data<W, M>;
    using SPConstParams = std::shared_ptr<typename TraitsT::Params const>;
    using SPConstMaterial = std::shared_ptr<optical::MaterialParams const>;
    //!@}

    //! Generator input data
    struct Input
    {
        AuxId optical_id;
        SPConstMaterial material;
        SPConstParams shared;
        size_type auto_flush{};
        size_type capacity{};

        explicit operator bool() const
        {
            return optical_id && material && shared && auto_flush && capacity;
        }
    };

  public:
    // Construct and add to core params
    static std::shared_ptr<GeneratorAction>
    make_and_insert(CoreParams const&, Input&&);

    // Construct with action ID, data IDs, and optical properties
    GeneratorAction(ActionId, AuxId, Input&&);

    //!@{
    //! \name Aux interface

    //! Index of this class instance in its registry
    AuxId aux_id() const final { return aux_id_; }
    // Build state data for a stream
    UPState create_state(MemSpace, StreamId, size_type) const final;
    //!@}

    //!@{
    //! \name Action interface

    //! ID of the action
    ActionId action_id() const final { return action_id_; }
    //! Short name for the action
    std::string_view label() const final { return TraitsT::label; }
    //! Description of the action
    std::string_view description() const final { return TraitsT::description; }
    //!@}

    //!@{
    //! \name StepAction interface

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::user_post; }
    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;
    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;
    //!@}

  private:
    //// DATA ////

    ActionId action_id_;
    AuxId aux_id_;
    Input data_;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    void step_impl(CoreParams const&, CoreState<M>&) const;

    void generate(CoreParams const&, CoreStateHost&) const;
    void generate(CoreParams const&, CoreStateDevice&) const;
};

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

extern template class GeneratorAction<GeneratorType::cherenkov>;
extern template class GeneratorAction<GeneratorType::scintillation>;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
