//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfaceModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

#include "celeritas/optical/action/ActionInterface.hh"
#include "celeritas/phys/SurfaceModel.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
class SurfaceModel : public celeritas::SurfaceModel,
                     public OpticalStepActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SurfaceModelBuilder
        = std::function<std::shared_ptr<SurfaceModel>(ActionId)>;
    //!@}

  public:
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
