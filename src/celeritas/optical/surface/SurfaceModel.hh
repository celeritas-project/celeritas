//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfaceModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/Types.hh"
#include "celeritas/optical/action/ActionInterface.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
template<SurfacePhysicsStep S>
class SurfaceModel : public OpticalStepActionInterface, public ConcreteAction
{
  public:
    //!@{
    //! \name Type aliases
    using ModelBuilder
        = std::function<std::shared_ptr<SurfaceModel<S>>(ActionId)>;
    //!@}

  public:
    using ConcreteAction::ConcreteAction;

    //! Action order for surface models is always post-step
    StepActionOrder order() const override { return StepActionOrder::post; }

    //! Surface physics step for this model
    static constexpr SurfacePhysicsStep surface_step() { return S; }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
