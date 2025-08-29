//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/SmearRoughnessModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"

#include "BuiltinSurfaceModel.hh"
#include "SmearRoughnessData.hh"

namespace celeritas
{
namespace inp
{
struct SmearRoughness;
}
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
class SmearRoughnessModel
    : public BuiltinSurfaceModel<SurfacePhysicsOrder::roughness>
{
  public:
    using InputT = inp::SmearRoughness;

    SmearRoughnessModel(SurfaceModelId model,
                        std::vector<PhysSurfaceId> surfaces,
                        std::vector<InputT> const& inputs);

    void step(CoreParams const& params, CoreStateHost& state) const final;
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    CollectionMirror<SmearRoughnessData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
