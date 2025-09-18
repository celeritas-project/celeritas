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
 * Smear roughness model.
 *
 * Approximates the surface roughness of an optical surface with the GliSur3
 * uniform smear roughness model.
 */
class SmearRoughnessModel : public BuiltinRoughnessModel
{
  public:
    //!@{
    //! \name Type aliases
    using InputT = inp::SmearRoughness;
    //!@}

  public:
    // Construct model from surfaces and inputs
    SmearRoughnessModel(SurfaceModelId model,
                        std::map<PhysSurfaceId, InputT> const& inputs);

    // Launch kernel on host
    void step(CoreParams const& params, CoreStateHost& state) const final;

    // Launch kernel on device
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    CollectionMirror<SmearRoughnessData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
