//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Types.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

#include "celeritas/Types.hh"

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

namespace celeritas
{

//! Opaque index to a scintillation particle id
using ScintParticleId = OpaqueId<struct ScintParticle_>;

//! Opaque index to a scintillation spectrum
using ParScintSpectrumId = OpaqueId<struct ParScintSpectrum>;

namespace optical
{

//! Opaque index to a volumetric optical model
using ModelId = OpaqueId<class Model>;

//! Opaque index to an optical surface model
using SurfaceModelId = OpaqueId<class SurfaceModel>;

//!@{
//! \name Surface physics indices

//! Opaque index to a surface defined by the canonical geometry
using GeometricSurfaceId = ::celeritas::SurfaceId;

//! Opaque index to the optical physics parameters of a surface for a specific
//! model
using SurfacePhysicsId = OpaqueId<struct PerModelSurfacePhysics_>;

//! Opaque index to a sub-surface material of a geometric surface
using SubsurfaceMaterialId = OpaqueId<struct SubsurfaceMaterial_>;

//! Opaque index to a sub-surface interface of a geometric surface
using SubsurfaceInterfaceId = OpaqueId<struct SubsurfaceInterface_>;

//!@}

}  // namespace optical
}  // namespace celeritas

//---------------------------------------------------------------------------//
// ENUMERATIONS
//---------------------------------------------------------------------------//

namespace celeritas
{

//! Process used to generate optical photons
enum class GeneratorType
{
    cherenkov,
    scintillation,
};

namespace optical
{

//! Direction to traverse the sub-surfaces of a geometric surface
enum class SubsurfaceDirection : bool
{
    reverse = false,
    forward = true
};

//! Sufrace physics substeps
enum class SurfacePhysicsStep
{
    roughness, reflectivity, interaction, size_;
};

}  // namespace optical
}  // namespace celeritas

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Convert a direction into a signed integer.
 *
 * The forward direction is positive (+1) and the reverse direction is negative
 * (-1).
 */
CELER_FORCEINLINE_FUNCTION int to_signed_offset(SubsurfaceDirection d)
{
    return 2 * static_cast<int>(d) - 1;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
