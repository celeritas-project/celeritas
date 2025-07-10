//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/simple/SimpleModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/surface/SurfaceModel.hh"

namespace celeritas
{
namespace optical
{

namespace
{

std::string simple_model_name(SurfacePhysicsStep s)
{
    switch (s)
    {
        case SurfacePhysicsStep::roughness:
            return "roughness";
        case SurfacePhysicsStep::reflectivity:
            return "reflectivity";
        case SurfacePhysicsStep::interaction:
            return "interaction";
        default:
            return "invalid";
    };
}

}  // namespace

//---------------------------------------------------------------------------//
/*!
 */
template<SurfacePhysicsStep S>
class SimpleModel : public SurfaceModel<S>
{
  public:
    SimpleModel(ActionId aid)
        : SurfaceModel<S>(aid,
                          "simple-" + simple_model_name(S) + "-model",
                          "simple " + simple_model_nams(S) + " description")
    {
    }

    void step(CoreParams const& params, CoreHostState& state) const final {}

    void step(CoreParams const& params, CoreDeviceState& state) const final {}
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
