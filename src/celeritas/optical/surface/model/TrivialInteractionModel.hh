//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/TrivialInteractionModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/optical/surface/SurfaceModel.hh"

#include "TrivialInteractionData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Trivial interaction model for optical surface physics.
 *
 * Calls precisely one interactor with no random sampling for a given surface.
 * Mainly useful for testing and very simple simulations.
 */
class TrivialInteractionModel : public SurfaceModel
{
  public:
    //!@{
    //! \name Type aliases
    using InputT = TrivialInteractionMode;
    //!@}

  public:
    //! Construct the model from an ID and layer map
    TrivialInteractionModel(SurfaceModelId,
                            std::map<PhysSurfaceId, InputT> const&);

    //! Get the list of physical surfaces this model applies to
    VecSurfaceLayer const& get_surfaces() const final { return surfaces_; }

    // Execute the model with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Execute the model with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    VecSurfaceLayer surfaces_;
    CollectionMirror<TrivialInteractionData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
