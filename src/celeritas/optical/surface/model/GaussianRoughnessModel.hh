//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/GaussianRoughnessModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionMirror.hh"
#include "celeritas/inp/SurfacePhysics.hh"

#include "GaussianRoughnessData.hh"
#include "GaussianRoughnessExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
class GaussianRoughnessModelController
{
  public:
    constexpr static std::string_view label = "gaussian";

    GaussianRoughnessModelController(
        std::vector<inp::GaussianRoughness> const& input);

    template<MemSpace M>
    GaussianRoughnessExecutorBuilder make_builder() const
    {
        return GaussianRoughnessExecutorBuilder{data_.ref<M>()};
    }

  private:
    CollectionMirror<GaussianRoughnessData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
