//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/PolishedRoughnessModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "BuiltinSurfaceModel.hh"

namespace celeritas
{
namespace inp
{
struct NoRoughness;
}
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
class PolishedRoughnessModel
    : public BuiltinSurfaceModel<SurfacePhysicsOrder::roughness>
{
  public:
    using InputT = inp::NoRoughness;

    PolishedRoughnessModel(SurfaceModelId model,
                           std::vector<PhysSurfaceId> surfaces,
                           std::vector<inp::NoRoughness> const&);

    void step(CoreParams const& params, CoreStateHost& state) const final;
    void step(CoreParams const&, CoreStateDevice&) const final;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
