//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantParticleUtils.cc
//---------------------------------------------------------------------------//
#include "GeantParticleUtils.hh"

#include <G4ParticleDefinition.hh>

#include "corecel/cont/Range.hh"
#include "celeritas/ext/GeantParticleView.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Map every Celeritas particle ID to a Geant4 particle definition.
 *
 * The result is indexed by \c ParticleId. An exception is thrown if any
 * Celeritas particle has no Geant4 equivalent.
 */
std::vector<G4ParticleDefinition const*> make_geant_particles(
    ParticleParams const& par)
{
    std::vector<G4ParticleDefinition const*> result(par.size());
    for (auto pid : range(ParticleId{par.size()}))
    {
        result[pid.get()]
            = &GeantParticleView::from_pdg(par.id_to_pdg(pid)).definition();
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
