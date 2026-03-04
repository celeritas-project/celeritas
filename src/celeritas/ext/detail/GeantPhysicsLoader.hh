//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantPhysicsLoader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_set>
#include <G4Version.hh>

#include "corecel/Config.hh"

#include "GeantOpticalModelImporter.hh"

class G4VProcess;

namespace celeritas
{
struct ImportData;
class GeoOpticalIdMap;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Load data from Geant4 physics processes and models.
 *
 * One process can be shared by multiple particles. Only handles special cases
 * (non-EM, non-MSC processes): optical processes, Cerenkov, scintillation,
 * and muon-catalyzed fusion capture. Returns \c true if the process was
 * recognized (even if no data required import), \c false otherwise.
 *
 * \todo When import data is refactored this will replace \em all G4 process
 * importing.
 */
class GeantPhysicsLoader
{
  public:
    // Construct with reference to data being loaded
    GeantPhysicsLoader(ImportData& imported,
                       GeoOpticalIdMap const& optical_ids);

    // Load if possible, returning whether it was recognized
    bool operator()(G4VProcess const& p);

  private:
    ImportData& imported_;
    GeantOpticalModelImporter import_optical_model_;
    std::unordered_set<G4VProcess const*> visited_;

    //// PHYSICS LOADERS ////

    // EM particles
    void cerenkov(G4VProcess const& p);
    void muon_minus_atomic_capture(G4VProcess const& p);
    void scintillation(G4VProcess const& p);

    // Optical photons
    void op_absorption(G4VProcess const& p);
    void op_boundary(G4VProcess const& p);
    void op_mie_hg(G4VProcess const& p);
    void op_rayleigh(G4VProcess const& p);
    void op_wls(G4VProcess const& p);
    void op_wls2(G4VProcess const& p);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
