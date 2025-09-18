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
 * Gaussian roughness surface model.
 *
 * Approximates the surface roughness of an optical surface with the UNIFIED
 * Gaussian roughness model.
 */
class GaussianRoughnessModel : public BuiltinRoughnessModel
{
  public:
    //!@{
    //! \name Type aliases
    using InputT = inp::GaussianRoughness;
    //!@}

  public:
    // Construct model from surfaces and inputs
    GaussianRoughnessModel(SurfaceModelId model,
                           std::map<PhysSurfaceId, InputT> const& inputs);

    // Launch kernel on host
    void step(CoreParams const& params, CoreStateHost& state) const final;

    // Launch kernel on device
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    CollectionMirror<GaussianRoughnessData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
