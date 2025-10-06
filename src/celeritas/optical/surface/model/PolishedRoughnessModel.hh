//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/PolishedRoughnessModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/inp/SurfacePhysics.hh"
#include "celeritas/optical/surface/SurfaceModel.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Trivial roughness model for a perfectly polished surface.
 *
 * Sets the facet normal equal to the surface's global normal for each track.
 */
class PolishedRoughnessModel : public SurfaceModel
{
  public:
    //!@{
    //! \name Type aliases
    using InputT = inp::NoRoughness;
    //!@}

  public:
    // Construct the model from an ID and layer map
    PolishedRoughnessModel(SurfaceModelId,
                           std::map<PhysSurfaceId, InputT> const&);

    //! Get the list of physical surfaces this model applies to
    VecSurfaceLayer const& get_surfaces() const final { return surfaces_; }

    // Execute the model with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Execute the model with state data
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    VecSurfaceLayer surfaces_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
