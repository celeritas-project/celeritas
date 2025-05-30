//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SimpleReflectionModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "SurfaceModel.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
class SimpleReflectionModel : public SurfaceModel
{
  public:
    //!@{
    //! \name Type aliases
    //!@}

  public:
    // Create a model builder
    static ModelBuilder make_builder();

    // Construct with action id
    SimpleReflectionModel(ActionId id);

    // Execute the model with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Execute the model with device data
    void step(CoreParams const&, CoreStateDevice&) const final;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
