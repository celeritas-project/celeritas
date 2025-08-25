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
class SurfaceModel : public ::celeritas::SurfaceModel
{
  public:
    //!@{
    //! \name Type aliases
    using SPModel = std::shared_ptr<SurfaceModel>;
    using ModelBuilder = std::function<SPModel(SurfaceModelId)>;

    using CoreStateHost = CoreState<MemSpace::host>;
    using CoreStateDevice = CoreState<MemSpace::device>;
    //!@}

  public:
    using ::celeritas::SurfaceModel::SurfaceModel;

    virtual void step(CoreParams const&, CoreStateHost&) const = 0;
    virtual void step(CoreParams const&, CoreStateDevice&) const = 0;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
