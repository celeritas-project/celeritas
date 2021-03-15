//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/OpaqueId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

//! Opaque index to ParticleDef in a vector: represents a particle type
using ParticleId = OpaqueId<struct ParticleDef>;

//! Opaque index of physics model
using ModelId = OpaqueId<class Model>;

//! Opaque index of physics process
using ProcessId = OpaqueId<class Process>;

//! Opaque index of a process applicable to a single particle type
using ParticleProcessId = OpaqueId<ProcessId>;

//! Opaque index of electron subshell
using SubshellId = OpaqueId<struct Subshell>;

//---------------------------------------------------------------------------//
//! Whether an interpolation is linear or logarithmic
enum class Interp
{
    linear,
    log
};

//---------------------------------------------------------------------------//
} // namespace celeritas
