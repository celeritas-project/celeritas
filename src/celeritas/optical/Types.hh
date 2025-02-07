//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Types.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Types.hh"

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
//---------------------------------------------------------------------------//
//! Alias for MaterialId in core Celeritas namespace
using CoreMaterialId = ::celeritas::MaterialId;

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
