//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Types.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

//! Opaque index to a scintillation particle id
using ScintillationParticleId = OpaqueId<struct ScintillationParticle_>;

//! Opaque index to a scintillation spectrum
using ParticleScintSpectrumId = OpaqueId<struct ParScintSpectrumRecord_>;

//---------------------------------------------------------------------------//
/*!
 * Physics classes used inside the optical physics loop.
 *
 * Interface classes that integrate with the main Celeritas stepping loop are
 * in the main namespace.
 */
namespace optical
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
