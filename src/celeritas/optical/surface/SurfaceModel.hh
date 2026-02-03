//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfaceModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/action/ActionInterface.hh"
#include "celeritas/phys/SurfaceModel.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Optical surface physics model.
 *
 * Base class for implementing an optical surface physics model for a given
 * physics sub-step. Adds \c step functions to the core \c phys::SurfaceModel.
 */
class SurfaceModel : public ::celeritas::SurfaceModel
{
  public:
    //!@{
    //! \name Type aliases
    using CoreStateHost = CoreState<MemSpace::host>;
    using CoreStateDevice = CoreState<MemSpace::device>;
    //!@}

  public:
    using ::celeritas::SurfaceModel::SurfaceModel;

    //! Execute the model with host data
    virtual void step(CoreParams const&, CoreStateHost&) const = 0;

    //! Execute the model with device data
    virtual void step(CoreParams const&, CoreStateDevice&) const = 0;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
