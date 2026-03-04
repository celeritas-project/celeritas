//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/CelerPhysics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>
#include <CeleritasG4.hh>
#include <DDG4/Geant4Action.h>
#include <DDG4/Geant4PhysicsList.h>
#include <G4VModularPhysicsList.hh>

namespace celeritas
{
namespace dd
{
//---------------------------------------------------------------------------//
/*!
 * DDG4 action plugin for Celeritas tracking manager integration (TMI).
 *
 * Two field modes are supported:
 *  - Uniform field (default): reads a \c ConstantField from the DD4hep
 *    detector description and creates a \c UniformAlongStepFactory.
 *  - Covfie field map: when \c FieldMapFile is set to a non-empty path, loads
 *    a binary covfie file (affine → nearest_neighbour → strided → array
 *    pipeline, coordinates in cm, field in T) and creates a
 *    \c CartMapFieldAlongStepFactory.  Requires \c CELERITAS_USE_covfie.
 *
 * Steering-file properties:
 *  - \c MaxNumTracks  (int)         : maximum tracks in flight
 *  - \c InitCapacity  (int)         : initial state-vector capacity
 *  - \c IgnoreProcesses (string[])  : physics processes to bypass
 *  - \c FieldMapFile  (string)      : path to a covfie field-map binary
 */
class CelerPhysics final : public dd4hep::sim::Geant4PhysicsList
{
  public:
    // Standard constructor
    CelerPhysics(dd4hep::sim::Geant4Context* ctxt, std::string const& name);

    // Delete copy/move
    DDG4_DEFINE_ACTION_CONSTRUCTORS(CelerPhysics);

    // constructPhysics callback
    virtual void constructPhysics(G4VModularPhysicsList* physics) final;

  private:
    int max_num_tracks_{0};
    int init_capacity_{0};
    std::vector<std::string> ignore_processes_;
    std::string field_map_file_;  //!< Path to covfie binary field map (optional)

    // Make options for Celeritas tracking manager
    SetupOptions make_options();
};

//---------------------------------------------------------------------------//
}  // namespace dd
}  // namespace celeritas
