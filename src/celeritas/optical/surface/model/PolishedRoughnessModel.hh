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
 * Polished roughness surface model.
 *
 * Trivial roughness model that just uses the global surface normal as the
 * local facet normal.
 */
class PolishedRoughnessModel : public BuiltinRoughnessModel
{
  public:
    //!@{
    //! \name Type aliases
    using InputT = inp::NoRoughness;
    //!@}

  public:
    // Construct model from surfaces and inputs
    PolishedRoughnessModel(SurfaceModelId model,
                           std::map<PhysSurfaceId, InputT> const& inputs);

    // Launch kernel on host
    void step(CoreParams const& params, CoreStateHost& state) const final;

    // Launch kernel on device
    void step(CoreParams const&, CoreStateDevice&) const final;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
