//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/GaussianRoughnessModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"

#include "BuiltinSurfaceModel.hh"
#include "GaussianRoughnessData.hh"

namespace celeritas
{
namespace inp
{
struct GaussianRoughness;
}
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
class GaussianRoughnessModel
    : public BuiltinSurfaceModel<SurfacePhysicsOrder::roughness>
{
  public:
    using InputT = inp::GaussianRoughness;

    GaussianRoughnessModel(SurfaceModelId model,
                           std::vector<PhysSurfaceId> surfaces,
                           std::vector<InputT> const& inputs);

    void step(CoreParams const& params, CoreStateHost& state) const final;
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    CollectionMirror<GaussianRoughnessData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
