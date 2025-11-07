//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/FtfpBertPhysicsList.cc
//---------------------------------------------------------------------------//
#include "FtfpBertPhysicsList.hh"

#include <memory>
#include <G4DecayPhysics.hh>
#include <G4EmStandardPhysics.hh>
#include <G4HadronElasticPhysics.hh>
#include <G4HadronPhysicsFTFP_BERT.hh>
#include <G4IonPhysics.hh>
#include <G4NeutronTrackingCut.hh>
#include <G4StoppingPhysics.hh>
#include <G4ios.hh>

#include "corecel/io/ScopedStreamRedirect.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/g4/SupportedEmStandardPhysics.hh"
#include "celeritas/g4/SupportedOpticalPhysics.hh"

#include "detail/EmStandardPhysics.hh"
#include "detail/PhysicsListUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the FTFP_BERT physics list with modified EM standard physics.
 */
FtfpBertPhysicsList::FtfpBertPhysicsList(Options const& options)
{
    using ClhepLen = Quantity<units::ClhepTraits::Length, double>;

    ScopedStreamRedirect scoped_log(&std::cout);

    int verbosity = options.verbose;
    this->SetVerboseLevel(verbosity);
    this->SetDefaultCutValue(
        native_value_to<ClhepLen>(options.default_cutoff).value());

    // Add celeritas EM physics plus additional mu/hadron
    detail::emplace_physics<detail::EmStandardPhysics>(*this, options);

    if (options.optical)
    {
        // Celeritas-supported Optical Physics
        detail::emplace_physics<SupportedOpticalPhysics>(*this,
                                                         options.optical);
    }

    // TODO: Add a physics constructor equivalent to G4EmExtraPhysics

    // Decays
    detail::emplace_physics<G4DecayPhysics>(*this, verbosity);

    // Hadron elastic scattering
    detail::emplace_physics<G4HadronElasticPhysics>(*this, verbosity);

    // Hadron physics
    detail::emplace_physics<G4HadronPhysicsFTFP_BERT>(*this, verbosity);

    // Stopping physics
    detail::emplace_physics<G4StoppingPhysics>(*this, verbosity);

    // Ion physics
    detail::emplace_physics<G4IonPhysics>(*this, verbosity);

    // Neutron tracking cut
    detail::emplace_physics<G4NeutronTrackingCut>(*this, verbosity);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
