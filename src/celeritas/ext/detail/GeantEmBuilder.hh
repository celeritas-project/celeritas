//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantEmBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include <G4NuclearStopping.hh>
#include <G4ParticleDefinition.hh>
#include <G4hMultipleScattering.hh>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class replacing G4EmBuilder for Geant4 versions < 10.7.0.
 */
class GeantEmBuilder
{
  public:
    // Set up minimal EM particle list
    void construct_minimal_em_set();
    // Construct lepton and hadron EM standard physics
    void construct_charged(G4hMultipleScattering* ion_msc,
                           G4NuclearStopping* nuclear_stopping,
                           bool is_wvi);

  private:
    void construct_light_hadrons(G4ParticleDefinition* p1,
                                 G4ParticleDefinition* p2,
                                 bool is_hep,
                                 bool is_proton,
                                 bool is_wvi);
    void construct_ion_em_physics(G4hMultipleScattering* ion_msc,
                                  G4NuclearStopping* nuclear_stopping);
    void construct_basic_em_physics(G4hMultipleScattering* ion_msc,
                                    std::vector<int> const& hadrons);
    void construct_muon_em_physics(G4ParticleDefinition* particle,
                                   bool is_hep,
                                   bool is_wvi);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
