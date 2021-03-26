//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geometry/GeoMaterialParams.hh"
#include "geometry/GeoParams.hh"
#include "physics/base/CutoffParams.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/PhysicsParams.hh"
#include "physics/material/MaterialParams.hh"

namespace demo_loop
{
struct LDemoArgs;

//---------------------------------------------------------------------------//
/*!
 * Storage for problem data.
 */
struct LDemoParams
{
    // Geometry and materials
    std::shared_ptr<const celeritas::GeoParams>         geometry;
    std::shared_ptr<const celeritas::MaterialParams>    materials;
    std::shared_ptr<const celeritas::GeoMaterialParams> geo_mat;

    // Physics
    std::shared_ptr<const celeritas::ParticleParams> particles;
    std::shared_ptr<const celeritas::CutoffParams>   cutoffs;
    std::shared_ptr<const celeritas::PhysicsParams>  physics;

    //! True if all params are assigned
    explicit operator bool() const
    {
        return geometry && materials && geo_mat && particles && cutoffs
               && physics;
    }
};

//---------------------------------------------------------------------------//
// Load params from input arguments
LDemoParams load_params(const LDemoArgs& args);

//---------------------------------------------------------------------------//
} // namespace demo_loop
