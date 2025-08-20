//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfaceModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>

#include "celeritas/optical/Types.hh"
#include "celeritas/optical/action/ActionInterface.hh"
#include "celeritas/phys/SurfaceModel.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    SurfaceModel ...;
   \endcode
 */
class SurfaceModel : public ::celeritas::SurfaceModel,
                     public OpticalStepActionInterface,
                     public ConcreteAction
{
  public:
    //!@{
    //! \name Type aliases
    using SPModel = std::shared_ptr<SurfaceModel>;
    using ModelBuilder = std::function<SPModel(ActionId)>;
    //!@}

  public:
    using ConcreteAction::ConcreteAction;

    //! Action order for optical surface models is always post-step
    StepActionOrder order() const override { return StepActionOrder::post; }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
