//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/detail/BoundaryAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Move a track across a boundary.
 */
class BoundaryAction final : public ExplicitActionInterface,
                             public ConcreteAction
{
  public:
    // Construct with ID
    explicit BoundaryAction(ActionId);

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
