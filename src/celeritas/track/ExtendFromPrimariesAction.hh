//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/ExtendFromPrimariesAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Span.hh"
#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct Primary;

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from queued host primary particles.
 *
 * This will append to the queued TrackInitializer vector any primaries added
 * with \c CoreState::insert_primaries .
 */
class ExtendFromPrimariesAction final : public ExplicitActionInterface
{
  public:
    //! Construct with explicit Id
    explicit ExtendFromPrimariesAction(ActionId id) : id_(id) {}

    // Execute the action with host data
    void execute(CoreParams const& params, CoreStateHost& state) const final;

    // Execute the action with device data
    void execute(CoreParams const& params, CoreStateDevice& state) const final;

    //! ID of the action
    ActionId action_id() const final { return id_; }

    //! Short name for the action
    std::string label() const final { return "extend-from-primaries"; }

    //! Description of the action for user interaction
    std::string description() const final
    {
        return "create track initializers from primaries";
    }

    //! Dependency ordering of the action
    ActionOrder order() const final { return ActionOrder::start; }

  private:
    ActionId id_;

    template<MemSpace M>
    void execute_impl(CoreParams const&, CoreState<M>&) const;

    void process_primaries(CoreParams const&,
                           CoreStateHost&,
                           Span<Primary const>) const;
    void process_primaries(CoreParams const&,
                           CoreStateDevice&,
                           Span<Primary const>) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
