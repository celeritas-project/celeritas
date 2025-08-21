//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/SurfaceModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "corecel/OpaqueId.hh"
#include "geocel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Physics to be applied during a surface crossing.
 *
 * Each surface model is constructed independently given some \c inp data. It
 * internally maps a sequence of "global" \c SurfaceId to a "local"
 * \c InternalSurfaceId. It additionally allows an empty surface returned by \c
 * get_surfaces to indicate a default model to be applied when the user does
 * not specify surface properties.
 *
 * This is currently only used by optical physics classes.
 */
class SurfaceModel
{
  public:
    //!@{
    //! \name Type aliases

    //! Eventually to be a pair of surface+layer
    using PhysSurfaceId = SurfaceId;
    //! Vector of surfaces
    using VecSurfaceLayer = std::vector<PhysSurfaceId>;
    //! Opaque ID of this surface model
    using SurfaceModelId = OpaqueId<SurfaceModel>;
    //! Opaque index of surface data in the list for a particular surface model
    using InternalSurfaceId = OpaqueId<struct InternalModelSurface_>;
    //!@}

  public:
    // Anchored default destructor
    virtual ~SurfaceModel() = 0;

    //! Get the list of surfaces/layers this applies to
    virtual VecSurfaceLayer get_surfaces() const = 0;

    //// ID/LABEL INTERFACE ////

    //! Opaque ID of this surface model
    SurfaceModelId surface_model_id() const { return id_; }

    //! Short descriptive name of this model
    std::string_view label() const { return label_; }

  protected:
    // Construct with label and model ID
    SurfaceModel(SurfaceModelId, std::string_view);

    //!@{
    //! Prevent assignment through base class
    CELER_DEFAULT_COPY_MOVE(SurfaceModel);
    //!@}

  private:
    SurfaceModelId id_;
    std::string_view label_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
