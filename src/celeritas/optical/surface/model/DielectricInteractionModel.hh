//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/DielectricInteractionModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/inp/SurfacePhysics.hh"
#include "celeritas/optical/surface/SurfaceModel.hh"

#include "DielectricInteractionData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
class DielectricInteractionModel : public SurfaceModel
{
  public:
    //!@{
    //! \name Type aliases
    using InputT = inp::DielectricInteraction;
    //!@}

  public:
    // Construct the model from an ID and layer map
    DielectricInteractionModel(SurfaceModelId,
                               std::map<PhysSurfaceId, InputT> const&);

    //! Get the list of physical surfaces this model applies to
    VecSurfaceLayer const& get_surfaces() const final { return surfaces_; }

    // Execute the model with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Execute the model with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    VecSurfaceLayer surfaces_;
    CollectionMirror<DielectricData> dielectric_data_;
    CollectionMirror<UnifiedReflectionData> reflection_data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
