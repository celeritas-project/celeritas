//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantPhysicsLoader.hh
//! \sa test/celeritas/ext/GeantImporter.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "geocel/GeoOpticalIdMap.hh"
#include "celeritas/Types.hh"
#include "celeritas/inp/OpticalPhysics.hh"

class G4VProcess;
class G4MaterialPropertiesTable;

namespace celeritas
{
struct ImportData;
class GeantParticleView;

namespace detail
{
class GeantMaterialPropertyGetter;
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

    // Load if possible, returning whether we handled it
    bool operator()(G4VProcess const& p);

    // Load per-particle data if possible, returning whether we handled it
    bool operator()(GeantParticleView const& particle, G4VProcess const& p);

  private:
    //// TYPES ////

    using PropGetter = GeantMaterialPropertyGetter;

    //// DATA ////

    ImportData& imported_;
    GeoOpticalIdMap const& optical_ids_;
    std::vector<G4Material const*> optical_g4mat_;
    std::unordered_set<G4VProcess const*> visited_;

    //// PHYSICS LOADERS ////

    // EM particles (once per process)
    size_type cerenkov(G4VProcess const& p);
    size_type muon_minus_atomic_capture(G4VProcess const& p);
    size_type scintillation(G4VProcess const& p);

    // Optical photons (once per process)
    size_type op_absorption(G4VProcess const& p);
    size_type op_boundary(G4VProcess const& p);
    size_type op_mie_hg(G4VProcess const& p);
    size_type op_rayleigh(G4VProcess const& p);
    size_type op_wls(G4VProcess const& p);
    size_type op_wls2(G4VProcess const& p);

    // Per-particle loaders
    size_type mu_pair_production(GeantParticleView const&, G4VProcess const&);

    //// HELPERS ////

    // Make a material table accessor for an optical material
    PropGetter property_getter(OptMatId) const;

    inp::Grid load_mfp(OptMatId opt_id, std::string const& prop_name) const;

    template<class MM, optical::ImportModelClass IMC>
    void load_mfps(inp::OpticalBulkModel<MM, IMC>& model,
                   std::string const& prop_name) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
