//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleParamsPointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Span.hh"
#include "ParticleDef.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access particle definitions on the device.
 *
 * This view is created from \c ParticleParams. The size of the \c defs data
 * member is the number of particle types (accessed by \c ParticleDefId).
 *
 * \sa ParticleParams (owns the pointed-to data)
 * \sa ParticleTrackView (uses the pointed-to data in a kernel)
 */
struct ParticleParamsPointers
{
    span<const ParticleDef> defs;

    //! Check whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const { return !defs.empty(); }
};

//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//
