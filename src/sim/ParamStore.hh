//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParamStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geometry/GeoParams.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/material/MaterialParams.hh"
#include "ParamPointers.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage constant shared device data.
 */
class ParamStore
{
  public:
    //@{
    //! Type aliases
    using SPConstGeo      = std::shared_ptr<const GeoParams>;
    using SPConstMaterial = std::shared_ptr<const MaterialParams>;
    using SPConstParticle = std::shared_ptr<const ParticleParams>;
    //@}

  public:
    // Construct with no data
    ParamStore() = default;

    // Construct with the shared problem data
    ParamStore(SPConstGeo geo, SPConstMaterial mat, SPConstParticle particle);

    // Get a view to the managed data
    ParamPointers device_pointers();

  private:
    SPConstGeo      geo_params_;
    SPConstMaterial material_params_;
    SPConstParticle particle_params_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
