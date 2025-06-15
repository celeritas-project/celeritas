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
}  // namespace celeritas
