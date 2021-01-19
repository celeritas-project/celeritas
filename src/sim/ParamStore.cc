//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParamStore.cc
//---------------------------------------------------------------------------//
#include "ParamStore.hh"

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with the shared problem data.
 */
ParamStore::ParamStore(SPConstGeo      geo,
                       SPConstMaterial mat,
                       SPConstParticle particle)
    : geo_params_(std::move(geo))
    , material_params_(std::move(mat))
    , particle_params_(std::move(particle))
{
    CELER_EXPECT(geo_params_);
    CELER_EXPECT(material_params_);
    CELER_EXPECT(particle_params_);
}

//---------------------------------------------------------------------------//
/*!
 * Get a view to the managed data.
 */
ParamPointers ParamStore::device_pointers()
{
    ParamPointers result;
    result.geo      = geo_params_->device_pointers();
    result.material = material_params_->device_pointers();
    result.particle = particle_params_->device_pointers();
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
