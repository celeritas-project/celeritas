//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/PrimaryGeneratorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Macros.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxStateVec.hh"
#include "celeritas/inp/Events.hh"
#include "celeritas/optical/action/ActionInterface.hh"
#include "celeritas/phys/GeneratorInterface.hh"

#include "../GeneratorData.hh"
#include "../OffloadData.hh"

namespace celeritas
{
class CoreParams;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Generate optical primaries from user-configurable distributions.
 *
 * This reproducibly samples and initializes optical photons directly in track
 * slots.
 */
class PrimaryGeneratorAction final
    : public optical::OpticalStepActionInterface,
      public AuxParamsInterface,
      public GeneratorInterface
{
  public:
    //!@{
    //! \name Type aliases
    using Input = inp::OpticalPrimaryGenerator;
    //!@}

  public:
    // Construct and add to core params
    static std::shared_ptr<PrimaryGeneratorAction>
    make_and_insert(::celeritas::CoreParams const&,
                    optical::CoreParams const&,
                    Input&&);

    // Construct with IDs and distributions
    PrimaryGeneratorAction(ActionId, AuxId, GeneratorId, Input);

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
    std::string_view label() const final { return "primary-generate"; }
    //! Description of the action
    std::string_view description() const final;
    //!@}

    //!@{
    //! \name StepAction interface

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::generate; }
    // Launch kernel with host data
    void step(optical::CoreParams const&, CoreStateHost&) const final;
    // Launch kernel with device data
    void step(optical::CoreParams const&, CoreStateDevice&) const final;
    //!@}

    //!@{
    //! \name Generator interface

    //! ID of the generator
    GeneratorId generator_id() const final { return gen_id_; }
    // Get generator counters (mutable)
    GeneratorStateBase& counters(AuxStateVec&) const final;
    // Get generator counters
    GeneratorStateBase const& counters(AuxStateVec const&) const final;
    //!@}

    // Set the number of pending tracks
    template<MemSpace M>
    inline void queue_primaries(optical::CoreState<M>&) const;

  private:
    //// DATA ////

    ActionId action_id_;
    AuxId aux_id_;
    GeneratorId gen_id_;
    PrimaryDistributionData data_;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    void step_impl(optical::CoreParams const&, optical::CoreState<M>&) const;

    void generate(optical::CoreParams const&, CoreStateHost&) const;
    void generate(optical::CoreParams const&, CoreStateDevice&) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Set the number of pending tracks.
 *
 * The number of tracks to generate must be set at the beginning of each event
 * before the optical loop is launched.
 *
 * \todo Currently this is only called during testing, but it *must* be done at
 * the beginning of each event once this action is integrated into the stepping
 * loop. Refactor/replace this.
 */
template<MemSpace M>
void PrimaryGeneratorAction::queue_primaries(optical::CoreState<M>& state) const
{
    CELER_EXPECT(state.aux());
    auto& aux_state = this->counters(*state.aux());
    aux_state.counters.num_pending = data_.num_photons;
    state.counters().num_pending = data_.num_photons;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
