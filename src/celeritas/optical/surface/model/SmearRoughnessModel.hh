//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/SmearRoughnessModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/ParamsDataStore.hh"
#include "celeritas/inp/SurfacePhysics.hh"
#include "celeritas/optical/surface/SurfaceModel.hh"

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
class SmearRoughnessModel : public SurfaceModel
{
  public:
    //!@{
    //! \name Type aliases
    using InputT = inp::SmearRoughness;
    //!@}

  public:
    // Construct the model from an ID and layer map
    SmearRoughnessModel(SurfaceModelId, std::map<PhysSurfaceId, InputT> const&);

    //! Get the list of physical surfaces this model applies to
    VecSurfaceLayer const& get_surfaces() const final { return surfaces_; }

    // Execute the model with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Execute the model with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    VecSurfaceLayer surfaces_;
    ParamsDataStore<SmearRoughnessData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
