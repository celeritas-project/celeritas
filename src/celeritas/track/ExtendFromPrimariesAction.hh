//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/ExtendFromPrimariesAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Span.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/phys/Primary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct Primary;

template<MemSpace M>
struct PrimaryStateData;
class CoreStateInterface;

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from queued host primary particles.
 *
 * \todo Change "generate" step order to be at the end of the loop
 * alongside create secondaries, and execute the action immediately after
 * adding primaries.
 */
class ExtendFromPrimariesAction final : public CoreStepActionInterface,
                                        public AuxParamsInterface

{
  public:
    // Construct and add to core params
    static std::shared_ptr<ExtendFromPrimariesAction>
    make_and_insert(CoreParams const& core);

    // Hacky helper function (DEPRECATED) to get the primary action from core
    // params
    static std::shared_ptr<ExtendFromPrimariesAction const>
    find_action(CoreParams const& core);

    // Construct with explicit ids
    ExtendFromPrimariesAction(ActionId action_id, AuxId aux_id);

    // Add user-provided primaries on host
    void insert(CoreParams const& params,
                CoreStateInterface& state,
                Span<Primary const> host_primaries) const;

    //!@{
    //! \name Metadata interface

    //! Label for the auxiliary data and action
    std::string_view label() const final { return sad_.label(); }
    // Description of the action
    std::string_view description() const final { return sad_.description(); }
    //!@}

    //!@{
    //! \name Step action interface

    //! ID of the action
    ActionId action_id() const final { return sad_.action_id(); }
    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::generate; }
    // Perform a host action within a step
    void step(CoreParams const& params, CoreStateHost& state) const final;
    // Perform a device action within a step
    void step(CoreParams const& params, CoreStateDevice& state) const final;
    //!@}

    //!@{
    //! \name Aux params interface

    //! Index of this class instance in its registry
    AuxId aux_id() const final { return aux_id_; }
    // Build state data for a stream
    UPState create_state(MemSpace m, StreamId id, size_type size) const final;
    //!@}

  private:
    StaticActionData sad_;
    AuxId aux_id_;

    template<MemSpace M>
    void
    insert_impl(CoreState<M>& state, Span<Primary const> host_primaries) const;

    template<MemSpace M>
    void step_impl(CoreParams const&, CoreState<M>&) const;

    void process_primaries(CoreParams const&,
                           CoreStateHost&,
                           PrimaryStateData<MemSpace::host> const&) const;
    void process_primaries(CoreParams const&,
                           CoreStateDevice&,
                           PrimaryStateData<MemSpace::device> const&) const;
};

template<MemSpace M>
struct PrimaryStateData : public AuxStateInterface
{
    // "Resizable" storage
    Collection<Primary, Ownership::value, M> storage;
    size_type count{0};

    //! Access valid range of primaries
    auto primaries()
    {
        return this->storage[ItemRange<Primary>{ItemId<Primary>{this->count}}];
    }

    //! Access valid range of primaries (const)
    auto primaries() const
    {
        return this->storage[ItemRange<Primary>{ItemId<Primary>{this->count}}];
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
