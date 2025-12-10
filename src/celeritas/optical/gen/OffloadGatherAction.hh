//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/OffloadGatherAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Macros.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/global/ActionInterface.hh"

#include "OffloadData.hh"

#include "detail/OffloadGatherTraits.hh"

namespace celeritas
{
class CoreParams;

//---------------------------------------------------------------------------//
/*!
 * Collect state data needed to create optical distribution data.
 *
 * This action is templated on the step action order so that it can be used to
 * collect both pre-step data and pre-post-step data.
 *
 * The pre-step action stores the optical material ID and other
 * beginning-of-step properties so that optical photons can be generated
 * between the start and end points of the step.
 *
 * The pre-post-step action stores the particle's speed following the
 * continuous energy loss but before the particle undergoes a discrete
 * interaction. These must be cached for scintillation, which offloads the
 * distribution data post-step rather than pre-post-step because it also
 * requires the local energy deposition from the interaction.
 *
 * \sa OffloadGatherExecutor
 */
template<StepActionOrder S>
class OffloadGatherAction final : public CoreStepActionInterface,
                                  public AuxParamsInterface
{
  public:
    //!@{
    //! \name Type aliases
    using TraitsT = detail::OffloadGatherTraits<S>;
    //!@}

  public:
    // Construct and add to core params
    static std::shared_ptr<OffloadGatherAction>
    make_and_insert(CoreParams const&);

    // Construct with action ID and storage
    OffloadGatherAction(ActionId id, AuxId aux_id);

    //!@{
    //! \name Aux interface

    //! Index of this class instance in its registry
    AuxId aux_id() const final { return aux_id_; }
    // Build state data for a stream
    UPState create_state(MemSpace, StreamId, size_type) const final;
    //!@}

    //!@{
    //! \name Action interface

    //! ID of the model
    ActionId action_id() const final { return action_id_; }
    //! Short name for the action
    std::string_view label() const final { return TraitsT::label; }
    //! Description of the action
    std::string_view description() const final { return TraitsT::description; }
    //!@}

    //!@{
    //! \name StepAction interface

    //! Dependency ordering of the action
    StepActionOrder order() const final { return S; }
    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;
    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;
    //!@}

  private:
    //// TYPES ////

    using Executor = typename TraitsT::Executor;

    //// DATA ////

    ActionId action_id_;
    AuxId aux_id_;
};

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

extern template class OffloadGatherAction<StepActionOrder::pre>;
extern template class OffloadGatherAction<StepActionOrder::pre_post>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
