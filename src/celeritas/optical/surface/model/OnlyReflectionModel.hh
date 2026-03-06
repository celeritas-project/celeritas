//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/OnlyReflectionModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/ParamsDataStore.hh"
#include "celeritas/optical/surface/SurfaceModel.hh"

#include "OnlyReflectionData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Only reflection model for optical surface physics.
 *
 * Applies only one \c ReflectionMode for a given surface. Used by Geant4's
 * painted finishes in the UNIFIED model which are only specular-spike
 * reflecting (polished-painted) or diffuse-lobe reflecting (ground-painted).
 */
class OnlyReflectionModel : public SurfaceModel
{
  public:
    //!@{
    //! \name Type aliases
    using InputT = ReflectionMode;
    //!@}

  public:
    //! Construct the model from an ID and layer map
    OnlyReflectionModel(SurfaceModelId, std::map<PhysSurfaceId, InputT> const&);

    //! Get the list of physical surfaces this model applies to
    VecSurfaceLayer const& get_surfaces() const final { return surfaces_; }

    // Execute the model with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Execute the model with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    VecSurfaceLayer surfaces_;
    ParamsDataStore<OnlyReflectionData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
