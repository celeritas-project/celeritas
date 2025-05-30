//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfaceModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Types.hh"
#include "celeritas/optical/action/ActionInterface.hh"

#include <memory>
#include <functional>

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
class SurfaceModel : public OpticalStepActionInterface, public ConcreteAction
{
  public:
    //!@{
    //! \name Type aliases
    
    //! Function to build optical models with a given action id
    using ModelBuilder = std::function<std::shared_ptr<SurfaceModel>(ActionId)>;

    //!@}

  public:
    using ConcreteAction::ConcreteAction;

    //! Action order for surface models is always post-step
    StepActionOrder order() const override { return StepActionOrder::post; }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
