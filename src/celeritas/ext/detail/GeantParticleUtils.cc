//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantParticleUtils.cc
//---------------------------------------------------------------------------//
#include "GeantParticleUtils.hh"

#include <G4ParticleDefinition.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Join.hh"
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
    std::vector<ParticleId> missing;
    for (auto pid : range(ParticleId{par.size()}))
    {
        if (auto const* particle = GeantParticleView::find(par.id_to_pdg(pid)))
        {
            result[pid.get()] = particle;
        }
        else
        {
            missing.push_back(pid);
        }
    }

    CELER_VALIDATE(missing.empty(),
                   << "failed to map Celeritas particles to Geant4: missing "
                   << join_stream(missing.begin(),
                                  missing.end(),
                                  ", ",
                                  [&par](std::ostream& os, ParticleId pid) {
                                      os << '"' << par.id_to_label(pid)
                                         << "\" (ID=" << pid.unchecked_get()
                                         << ", PDG="
                                         << par.id_to_pdg(pid).unchecked_get()
                                         << ")";
                                  }));
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
