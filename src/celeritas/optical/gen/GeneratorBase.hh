//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/GeneratorBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/sys/ActionInterface.hh"
#include "celeritas/optical/action/ActionInterface.hh"
#include "celeritas/phys/GeneratorInterface.hh"

#include "GeneratorCounterState.hh"

#include "detail/GeneratorCounterAlgorithms.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Common interface for generating optical photons.
 */
class GeneratorBase : virtual public optical::OpticalStepActionInterface,
                      virtual public AuxParamsInterface,
                      virtual public GeneratorInterface
{
  public:
    // Construct from IDs, unique label, and description
    GeneratorBase(ActionId id,
                  AuxId aux_id,
                  GeneratorId gen_id,
                  std::string_view label,
                  std::string_view description) noexcept(!CELERITAS_DEBUG);

    //!@{
    //! \name Aux interface

    //! Index of this class instance in its registry
    AuxId aux_id() const final { return aux_id_; }
    //!@}

    //!@{
    //! \name Action interface

    //! ID of the action
    ActionId action_id() const final { return sad_.action_id(); }
    //! Short name for the action
    std::string_view label() const final { return sad_.label(); }
    //! Description of the action
    std::string_view description() const final { return sad_.description(); }
    //!@}

    //!@{
    //! \name StepAction interface

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::generate; }
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

    // Get the generator counter state for a memory space
    template<MemSpace M>
    inline GeneratorCounterState<M>& counter_state(AuxStateVec&) const;

    // Update the generator and state counters
    template<MemSpace M>
    inline void update_counters(optical::CoreState<M>&) const;

  private:
    StaticActionData sad_;
    AuxId aux_id_;
    GeneratorId gen_id_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get the generator counter state for a memory space.
 */
template<MemSpace M>
GeneratorCounterState<M>& GeneratorBase::counter_state(AuxStateVec& aux) const
{
    return dynamic_cast<GeneratorCounterState<M>&>(this->counters(aux));
}

//---------------------------------------------------------------------------//
/*!
 * Update the generator and state counters.
 *
 * This enqueues a single-thread kernel that moves the photons generated in
 * this step from the pending to the generated counts, so it does not
 * synchronize.
 */
template<MemSpace M>
void GeneratorBase::update_counters(optical::CoreState<M>& state) const
{
    CELER_EXPECT(state.aux());

    auto& gen_state = this->counter_state<M>(*state.aux());
    detail::update_counters(state.ref().init.counters,
                            gen_state.device_counters(),
                            state.size(),
                            state.stream_id());
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
