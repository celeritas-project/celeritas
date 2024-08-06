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
#include "CelerOpticalPhysics.hh"
#include "MuHadEmStandardPhysics.hh"

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

    // Celeritas-supported EM physics
    auto celer_em = std::make_unique<CelerEmStandardPhysics>(options);
    RegisterPhysics(celer_em.release());

    // Celeritas-supported Optical Physics
    if (options.optical_options)
    {
        auto optical_physics = std::make_unique<CelerOpticalPhysics>(
            options.optical_options.value());
        RegisterPhysics(optical_physics.release());
    }

    // Muon and hadrom EM standard physics not supported in Celeritas
    auto muhad_em = std::make_unique<MuHadEmStandardPhysics>(verbosity);
    RegisterPhysics(muhad_em.release());

    // Synchroton radiation & GN physics
    auto em_extra = std::make_unique<G4EmExtraPhysics>(verbosity);
    RegisterPhysics(em_extra.release());

    // Decays
    auto decay = std::make_unique<G4DecayPhysics>(verbosity);
    RegisterPhysics(decay.release());

    // Hadron elastic scattering
    auto hadron_elastic = std::make_unique<G4HadronElasticPhysics>(verbosity);
    RegisterPhysics(hadron_elastic.release());

    // Hadron physics
    auto hadron = std::make_unique<G4HadronPhysicsFTFP_BERT>(verbosity);
    RegisterPhysics(hadron.release());

    // Stopping physics
    auto stopping = std::make_unique<G4StoppingPhysics>(verbosity);
    RegisterPhysics(stopping.release());

    // Ion physics
    auto ion = std::make_unique<G4IonPhysics>(verbosity);
    RegisterPhysics(ion.release());

    // Neutron tracking cut
    auto neutron_cut = std::make_unique<G4NeutronTrackingCut>(verbosity);
    RegisterPhysics(neutron_cut.release());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
