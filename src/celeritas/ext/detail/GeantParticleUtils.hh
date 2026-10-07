//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantParticleUtils.hh
//! \sa test/celeritas/ext/GeantSd.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

class G4ParticleDefinition;

namespace celeritas
{
class ParticleParams;

namespace detail
{
//---------------------------------------------------------------------------//
// Map every Celeritas particle ID to a Geant4 particle definition
std::vector<G4ParticleDefinition const*> make_geant_particles(
    ParticleParams const& par);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
