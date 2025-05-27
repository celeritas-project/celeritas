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
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/global/ActionInterface.hh"

#include "../CherenkovData.hh"
#include "../CherenkovGenerator.hh"
#include "../GeneratorData.hh"
#include "../OffloadData.hh"
#include "../ScintillationData.hh"
#include "../ScintillationGenerator.hh"

namespace celeritas
{
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
template<template<Ownership, MemSpace> class Data, class Generator>
class GeneratorAction final : public CoreStepActionInterface,
                              public AuxParamsInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParams = std::shared_ptr<ParamsDataInterface<Data> const>;
    using SPConstMaterial
        = std::shared_ptr<celeritas::optical::MaterialParams const>;
    //!@}

    //! Input data
    struct Input
    {
        ActionId action;
        AuxId aux;
        AuxId optical;
        SPConstMaterial material;
        SPConstParams shared;
        size_type auto_flush{};
        size_type buffer_capacity{};
        std::string label;

        explicit operator bool() const
        {
            return action && aux && optical && material && shared
                   && auto_flush > 0 && buffer_capacity > 0;
        }
    };

  public:
    // Construct with action ID, data IDs, and optical properties
    explicit GeneratorAction(Input&&);

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
    std::string_view label() const final { return label_; }
    // Name of the action (for user output)
    std::string_view description() const final;
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
    AuxId optical_id_;
    SPConstMaterial material_;
    SPConstParams shared_;
    size_type auto_flush_;
    size_type buffer_capacity_;
    std::string label_;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    void step_impl(CoreParams const&, CoreState<M>&) const;

    void generate(CoreParams const&, CoreStateHost&) const;
    void generate(CoreParams const&, CoreStateDevice&) const;
};

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

extern template class GeneratorAction<CherenkovData, CherenkovGenerator>;
extern template class GeneratorAction<ScintillationData, ScintillationGenerator>;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
