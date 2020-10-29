//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeantPhysicsTableWriter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iostream>
#include <vector>
#include <memory>

#include "io/ImportPhysicsTable.hh"

class TFile;
class TTree;

class G4VProcess;
class G4VEmProcess;
class G4VEnergyLossProcess;
class G4VMultipleScattering;
class G4ParticleDefinition;
class G4PhysicsTable;

using celeritas::ImportPhysicsTable;
using celeritas::ImportPhysicsVector;

namespace geant_exporter
{
//---------------------------------------------------------------------------//
/*!
 * Use an existing TFile address as input to create a new "tables" TTree used
 * store Geant physics tables.
 *
 * TFile passed to the constructor must be open with the "recreate" flag so the
 * class has writing privileges.
 */
class GeantPhysicsTableWriter
{
  public:
    // Constructor adds a new "tables" TTree to the existing ROOT TFile
    GeantPhysicsTableWriter(TFile* root_file);
    // Default destructor
    ~GeantPhysicsTableWriter() = default;

    // Write the physics tables from a given particle and physics process
    // Expected to be called within a G4ParticleTable iterator loop
    void add_physics_tables(const G4VProcess&           process,
                            const G4ParticleDefinition& particle);

  private:
    // Loop over EM processes and write tables to the ROOT file
    void fill_em_tables(const G4VEmProcess& em_process);
    // Loop over energy loss processes and write tables to the ROOT file
    void fill_energy_loss_tables(const G4VEnergyLossProcess& eloss_process);
    // Loop over multiple scattering processes and write tables to the ROOT
    // file
    void
    fill_multiple_scattering_tables(const G4VMultipleScattering& msc_process);
    // Write the physics vectors from a given G4PhysicsTable to this->table_
    void fill_physics_vectors(const G4PhysicsTable&         table,
                              ImportPhysicsVector::DataType xs_or_eloss);
    // Write the remaining elements of this->table_ and fill the tables TTree
    void fill_tables_tree(const G4PhysicsTable&         table,
                          std::string                   table_type_name,
                          std::string                   table_name,
                          ImportPhysicsVector::DataType xs_or_eloss);

  private:
    // TTree created by the constructor
    std::unique_ptr<TTree> tree_tables_;
    // Object written in the TTree. Each ImportPhysicsTable is a new TTree
    // entry
    ImportPhysicsTable table_;
};

//---------------------------------------------------------------------------//
} // namespace geant_exporter
