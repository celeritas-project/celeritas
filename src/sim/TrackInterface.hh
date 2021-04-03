//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "geometry/GeoInterface.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleInterface.hh"
#include "physics/material/MaterialInterface.hh"
#include "random/RngInterface.hh"
#include "SimInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Immutable problem data.
 */
struct ParamPointers
{
    template<template<Ownership, MemSpace> class S>
    using DeviceCRef = S<Ownership::const_reference, MemSpace::device>;

    DeviceCRef<GeoParamsData>      geo;
    DeviceCRef<MaterialParamsData> material;
    DeviceCRef<ParticleParamsData> particle;

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geo && material && particle;
    }
};

//---------------------------------------------------------------------------//
// STATE
//---------------------------------------------------------------------------//
/*!
 * Thread-local state data.
 */
struct StatePointers
{
    template<template<Ownership, MemSpace> class S>
    using DeviceRef = S<Ownership::reference, MemSpace::device>;

    DeviceRef<ParticleStateData> particle;
    DeviceRef<RngStateData>      rng;
    DeviceRef<GeoStateData>      geo;

    SimStatePointers  sim;
    Span<Interaction> interactions;

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return particle && geo && sim && rng && !interactions.empty();
    }

    //! Number of tracks
    CELER_FUNCTION size_type size() const { return particle.size(); }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
