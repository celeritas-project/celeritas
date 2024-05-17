//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/CelerFTFPBert.cc
//---------------------------------------------------------------------------//
#include "CelerFTFPBert.hh"

#include <memory>
#include <G4DecayPhysics.hh>
#include <G4EmExtraPhysics.hh>
#include <G4EmStandardPhysics.hh>
#include <G4HadronElasticPhysics.hh>
#include <G4HadronPhysicsFTFP_BERT.hh>
#include <G4IonPhysics.hh>
#include <G4NeutronTrackingCut.hh>
#include <G4StoppingPhysics.hh>
#include <G4ios.hh>

#include "celeritas/Quantities.hh"

#include "CelerEmStandardPhysics.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct the FTFP BERT physics list with modified EM standard physics.
 */
CelerFTFPBert::CelerFTFPBert(Options const& options)
{
    using ClhepLen = Quantity<units::ClhepTraits::Length, double>;

    int verbosity = options.verbose;
    this->SetVerboseLevel(verbosity);
    this->SetDefaultCutValue(
        native_value_to<ClhepLen>(options.default_cutoff).value());

    // Celeritas-supported EM Physics
    auto em_standard = std::make_unique<CelerEmStandardPhysics>(options);
    RegisterPhysics(em_standard.release());

    // Synchroton Radiation & GN Physics
    auto em_extra = std::make_unique<G4EmExtraPhysics>(verbosity);
    RegisterPhysics(em_extra.release());

    // Decays
    auto decay = std::make_unique<G4DecayPhysics>(verbosity);
    RegisterPhysics(decay.release());

    // Hadron Elastic scattering
    auto hadron_elastic = std::make_unique<G4HadronElasticPhysics>(verbosity);
    RegisterPhysics(hadron_elastic.release());

    // Hadron Physics
    auto hadron = std::make_unique<G4HadronPhysicsFTFP_BERT>(verbosity);
    RegisterPhysics(hadron.release());

    // Stopping Physics
    auto stopping = std::make_unique<G4StoppingPhysics>(verbosity);
    RegisterPhysics(stopping.release());

    // Ion Physics
    auto ion = std::make_unique<G4IonPhysics>(verbosity);
    RegisterPhysics(ion.release());

    // Neutron tracking cut
    auto neutron_cut = std::make_unique<G4NeutronTrackingCut>(verbosity);
    RegisterPhysics(neutron_cut.release());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
