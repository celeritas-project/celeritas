//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParamPointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"
#include "geometry/GeoParamsPointers.hh"
#include "physics/base/ParticleParamsPointers.hh"
#include "physics/material/MaterialParamsPointers.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * View to the immutable problem data
 */
struct ParamPointers
{
    GeoParamsPointers      geo;
    MaterialParamsPointers material;
    ParticleParamsPointers particle;

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geo && material && particle;
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//
