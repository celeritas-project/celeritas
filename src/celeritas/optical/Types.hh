//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Types.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

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

namespace optical
{

using ModelId = OpaqueId<class Model>;

using SubsurfaceMaterialId = OpaqueId<struct SubsurfaceMaterial>;
using SubsurfaceInterfaceId = OpaqueId<struct SubsurfaceInterface>;

enum class SubsurfaceDirection : int
{
    forward = 1,
    reverse = -1
};

}  // namespace optical

//---------------------------------------------------------------------------//
// ENUMERATIONS
//---------------------------------------------------------------------------//
//! Process used to generate optical photons
enum class GeneratorType
{
    cherenkov,
    scintillation,
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
