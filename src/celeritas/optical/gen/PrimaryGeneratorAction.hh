//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/PrimaryGeneratorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Macros.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxStateVec.hh"
#include "celeritas/inp/Events.hh"
#include "celeritas/optical/action/ActionInterface.hh"
#include "celeritas/phys/GeneratorInterface.hh"

#include "GeneratorBase.hh"
#include "GeneratorData.hh"
#include "OffloadData.hh"

namespace celeritas
{
class CoreParams;

namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Generate optical primaries from user-configurable distributions.
 *
 * This reproducibly samples and initializes optical photons directly in track
 * slots.
 */
class PrimaryGeneratorAction final : public GeneratorBase
{
  public:
    //!@{
    //! \name Type aliases
    using Input = inp::OpticalPrimaryGenerator;
    //!@}

  public:
    // Construct and add to core params
    static std::shared_ptr<PrimaryGeneratorAction>
    make_and_insert(::celeritas::CoreParams const&, CoreParams const&, Input&&);

    // Construct with IDs and distributions
    PrimaryGeneratorAction(ActionId, AuxId, GeneratorId, Input);

    //!@{
    //! \name Aux interface

    // Build state data for a stream
    UPState create_state(MemSpace, StreamId, size_type) const final;
    //!@}

    //!@{
    //! \name StepAction interface

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;
    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;
    //!@}

    // Set the number of pending tracks
    template<MemSpace M>
    inline void queue_primaries(CoreState<M>&) const;

  private:
    //// DATA ////

    PrimaryDistributionData data_;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    void step_impl(CoreParams const&, CoreState<M>&) const;

    void generate(CoreParams const&, CoreStateHost&) const;
    void generate(CoreParams const&, CoreStateDevice&) const;
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
void PrimaryGeneratorAction::queue_primaries(CoreState<M>& state) const
{
    CELER_EXPECT(state.aux());
    auto& aux_state = this->counters(*state.aux());
    aux_state.counters.num_pending = data_.num_photons;
    state.counters().num_pending = data_.num_photons;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
