//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/SurfaceModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"
#include "corecel/sys/ActionInterface.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Physics to be applied during a surface crossing.
 *
 * Each surface model is constructed independently given some \c inp data. It
 * internally maps a sequence of "global" \c SurfaceId to a "local" \c
 * ModelSurfaceId. If a \c SurfaceModel with action ID 10 returns a list of
 * surfaces {3, 1, 5} and another with ID 11 returns {0, 4}, then the \c
 * SurfacePhysicsMap class will store
 * \code
 * [{11, 0}, {10, 1}, <null>, {10, 0}, {11, 1}, {10, 2}]
 * \endcode
 * Since neither model specified surface 2, it is null.
 *
 * With this setup, \c Collection data can be accessed locally by indexing on
 * \c ModelSurfaceId .
 *
 * This is currently only used by optical physics classes. Daughters will also
 * inherit from \c OpticalStepActionInterface, \c  ConcreteAction .
 */
class SurfaceModel : virtual public ActionInterface
{
  public:
    //!@{
    //! \name Type aliases

    //! Eventually to be a pair of surface+layer
    using SurfaceLayer = SurfaceId;
    //! Vector of surfaces
    using VecSurfaceLayer = std::vector<SurfaceLayer>;
    //! Opaque index of surface data in the list for a particular surface model
    using ModelSurfaceId = OpaqueId<struct ModelSurface_>;

    //!@}

  public:
    // Get the list of surfaces/layers this applies to
    virtual VecSurfaceLayer get_surfaces() const = 0;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
