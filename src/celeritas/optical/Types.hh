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

enum class SurfacePhysicsStep
{
    Roughness,
    Reflectivity,
    Interaction,
    size_
};

template<SurfacePhysicsStep S>
class SurfaceModel;

using SurfaceRoughnessModel = SurfaceModel<SurfacePhysicsStep::Roughness>;
using SurfaceReflectivityModel = SurfaceModel<SurfacePhysicsStep::Reflectivity>;
using SurfaceInteractionModel = SurfaceModel<SurfacePhysicsStep::Interaction>;

using RoughnessModelId = OpaqueId<SurfaceRoughnessModel>;
using ReflectivityModelId = OpaqueId<SurfaceReflectivityModel>;
using InteractionModelId = OpaqueId<SurfaceInteractionModel>;

}  // namespace optical

//---------------------------------------------------------------------------//
}  // namespace celeritas
