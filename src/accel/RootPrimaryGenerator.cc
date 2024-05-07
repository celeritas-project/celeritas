//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/RootPrimaryGenerator.cc
//---------------------------------------------------------------------------//
#include "RootPrimaryGenerator.hh"

#include <G4Event.hh>
#include <G4ParticleGun.hh>
#include <G4ParticleTable.hh>
#include <TFile.h>
#include <TLeaf.h>
#include <TTree.h>

#include "corecel/cont/Range.hh"
#include "geocel/g4/Convert.geant.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a shared generator.
 */
RootPrimaryGenerator::RootPrimaryGenerator(std::string offloaded_filename,
                                           size_type num_events,
                                           size_type primaries_per_event)
    : num_events_{num_events}, primaries_per_event_(primaries_per_event)
{
    CELER_EXPECT(num_events > 0 && primaries_per_event > 0);

    root_input_.reset(TFile::Open(offloaded_filename.c_str(), "read"));
    CELER_ENSURE(root_input_->IsOpen());

    primaries_tree_.reset(root_input_->Get<TTree>("primaries"));
    CELER_ENSURE(primaries_tree_);
    CELER_VALIDATE(primaries_tree_->GetEntries() > 0,
                   << "TTree `primaries` in '" << root_input_->GetName()
                   << "' has zero entries");

    entry_selector_ = std::uniform_int_distribution<>(0, primaries_per_event);
}

//---------------------------------------------------------------------------//
/*!
 * Generate primaries from ROOT input file with offloaded primary data.
 */
void RootPrimaryGenerator::GeneratePrimaryVertex(G4Event* event)
{
    // Fetch ROOT leaves
    // TODO: Make these private members?
    auto lpid = primaries_tree_->GetLeaf("particle");
    CELER_ASSERT(lpid);
    auto lenergy = primaries_tree_->GetLeaf("energy");
    CELER_ASSERT(lenergy);
    auto lpos = primaries_tree_->GetLeaf("pos");
    CELER_ASSERT(lpos);
    auto ldir = primaries_tree_->GetLeaf("dir");
    CELER_ASSERT(ldir);
    auto ltime = primaries_tree_->GetLeaf("time");
    CELER_ASSERT(ltime);

    std::lock_guard scoped_lock{read_mutex_};

    // Reset the seed for every new event to ensure reproducibility
    rng_.seed(seed_ + event->GetEventID());

    G4ParticleGun gun;
    for ([[maybe_unused]] auto p : range(primaries_per_event_))
    {
        primaries_tree_->GetEntry(entry_selector_(rng_));

        auto part_def = G4ParticleTable::GetParticleTable()->FindParticle(
            lpid->GetValue());
        CELER_ENSURE(part_def);
        gun.SetParticleDefinition(part_def);
        gun.SetParticlePosition(
            convert_to_geant(this->to_array(*lpos), clhep_length));
        gun.SetParticleMomentumDirection(
            convert_to_geant(this->to_array(*ldir), 1));
        gun.SetParticleEnergy(
            convert_to_geant(lenergy->GetValue(), CLHEP::MeV));
        gun.SetParticleTime(convert_to_geant(ltime->GetValue(), CLHEP::s));
        gun.GeneratePrimaryVertex(event);

        if (CELERITAS_DEBUG)
        {
            CELER_ASSERT(G4VPrimaryGenerator::CheckVertexInsideWorld(
                gun.GetParticlePosition()));
        }
    }

    CELER_ENSURE(event->GetNumberOfPrimaryVertex()
                 == static_cast<int>(primaries_per_event_));
}

//---------------------------------------------------------------------------//
/*!
 * Convert TLeaf to Array.
 */
Array<real_type, 3> RootPrimaryGenerator::to_array(TLeaf const& leaf)
{
    return {leaf.GetValue(0), leaf.GetValue(1), leaf.GetValue(2)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
