//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/SensitiveDetector.cc
//---------------------------------------------------------------------------//
#include "SensitiveDetector.hh"

#include <memory>
#include <G4Step.hh>

#include "corecel/Assert.hh"

#include "Threading.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a step callback, or return null if no callback given.
 */
std::unique_ptr<G4VSensitiveDetector>
SensitiveDetector::from_hit_function(std::string sd_name, FuncLocalStep f)
{
    if (!f)
        return nullptr;
    return std::make_unique<SensitiveDetector>(std::move(sd_name),
                                               std::move(f));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a detector name and step callback.
 */
SensitiveDetector::SensitiveDetector(std::string const& name, FuncLocalStep f)
    : G4VSensitiveDetector(name), hit_func_{std::move(f)}
{
    CELER_EXPECT(hit_func_);
}

//---------------------------------------------------------------------------//
/*!
 * Clear the hit collection at the start of each event.
 */
void SensitiveDetector::Initialize(G4HCofThisEvent*)
{
    this->clear();
}

//---------------------------------------------------------------------------//
/*!
 * Call the step function for each step depositing energy in this volume.
 */
G4bool SensitiveDetector::ProcessHits(G4Step* step, G4TouchableHistory*)
{
    CELER_EXPECT(step);
    hit_func_(geant_stream(), *step);
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
