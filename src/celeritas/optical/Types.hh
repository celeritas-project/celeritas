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
using ScintParticleId = OpaqueId<struct ScintParticle_>;

//! Opaque index to a scintillation spectrum
using ParScintSpectrumId = OpaqueId<struct ParScintSpectrum>;

//---------------------------------------------------------------------------//
// ENUMERATIONS
//---------------------------------------------------------------------------//

//! Process used to generate optical photons
enum class GeneratorType
{
    cherenkov,
    scintillation,
};

namespace optical
{

//! Ordering of surface physics boundary crossing models
enum class SurfacePhysicsOrder
{
    roughness,
    reflectivity,
    interaction,
    size_
};

//! Traversal direction of a sub-subsurface
enum class SubsurfaceDirection : bool
{
    reverse = false,
    forward = true
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

char const* to_cstring(SurfacePhysicsOrder);

//! Convert sub-surface direction to a sign (+1/-1 for forward/reverse resp.)
CELER_FORCEINLINE_FUNCTION int to_signed_offset(SubsurfaceDirection d)
{
    return 2 * static_cast<int>(d) - 1;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
