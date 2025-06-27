//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfaceRoughnessModel.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
class SurfaceRoughnessModel : public OpticalStepActionInterface,
                              public ConcreteAction
{
  public:
    //!@{
    //! \name Type aliases
    //! Function to build a surface roughness model with a given action ID
    using ModelBuilder
        = std::function<std::shared_ptr<SurfaceRoughnessModel>(ActionId)>;
    //!@}

  public:
    using ConcreteAction::ConcreteAction;

    StepActionOrder order() const override { return StepActionOrder::post; }
};
//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
