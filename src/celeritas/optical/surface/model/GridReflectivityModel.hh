//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/GridReflectivityModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/ParamsDataStore.hh"
#include "celeritas/inp/SurfacePhysics.hh"
#include "celeritas/optical/surface/SurfaceModel.hh"

#include "GridReflectivityData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * User-defined grid reflectivity model.
 *
 * Allows user-defined grids to override the usual surface physics logic.
 * Following Geant4's conventions, reflectivity is defined as the probability a
 * track continues with the usual surface interaction (not necessarily just
 * reflects). Transmittance is the probability the track moves to the next
 * surface layer without any changes. If the reflectivity and transmittance do
 * not sum to 1, then the remaining probability is the chance the track is
 * absorbed on the surface.
 *
 * If a track is absorbed on the surface and there's a non-zero efficiency
 * grid, it is sampled as the probability the track is "detected" on the
 * surface. Because this is a hold-over from Geant4 integration, if the track
 * is sampled to pass the efficiency then it is changed from absorbed to
 * transmitted. If the next volume is indeed a detector volume, then it is
 * detected and killed at the surface which matches Geant4's expectation for
 * detection on a surface.
 */
class GridReflectivityModel : public SurfaceModel
{
  public:
    //!@{
    //! \name Type aliases
    using InputT = inp::GridReflection;
    //!@}

  public:
    // Construct the model from an ID and layer map
    GridReflectivityModel(SurfaceModelId,
                          std::map<PhysSurfaceId, InputT> const&);

    //! Get the list of physical surfaces this model applies to
    VecSurfaceLayer const& get_surfaces() const final { return surfaces_; }

    // Execute the model with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Execute the model with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    VecSurfaceLayer surfaces_;
    ParamsDataStore<GridReflectivityData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
