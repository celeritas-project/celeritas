//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/TrackingCutAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Kill tracks and deposit energy locally.
 *
 * \sa CoreTrackView::tracking_cut_action
 */
class TrackingCutAction final : public ExplicitCoreActionInterface,
                                public ConcreteAction
{
  public:
    // Construct with ID
    explicit TrackingCutAction(ActionId);

    // Launch kernel with host data
    void execute(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void execute(CoreParams const&, CoreStateDevice&) const final;

    //! Dependency ordering of the action
    ActionOrder order() const final { return ActionOrder::post; }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
