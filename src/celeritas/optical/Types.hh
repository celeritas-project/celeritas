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

//! Opaque index to a material with optical properties
using OpticalMaterialId = OpaqueId<struct OpticalMaterial_>;

//! Opaque index to a scintillation particle id
using ScintillationParticleId = OpaqueId<struct ScintillationParticle_>;

//! Opaque index to a scintillation spectrum
using ParticleScintSpectrumId = OpaqueId<struct ParticleScintillationSpectrum>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
