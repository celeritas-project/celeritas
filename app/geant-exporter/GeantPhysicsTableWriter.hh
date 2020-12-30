//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeantPhysicsTableWriter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iostream>
#include <unordered_map>
#include <vector>
#include <memory>

#include "io/ImportProcess.hh"

class TFile;
class TTree;

class G4VProcess;
class G4VEmProcess;
class G4VEnergyLossProcess;
class G4VMultipleScattering;
class G4ParticleDefinition;
class G4PhysicsTable;

namespace geant_exporter
{
enum class TableSelection
{
    minimal,
    all
};

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
    GeantPhysicsTableWriter(TFile* root_file, TableSelection which_tables);

    // Write the tables on destruction
    ~GeantPhysicsTableWriter();

    // Write the physics tables from a given particle and physics process
    // Expected to be called within a G4ParticleTable iterator loop
    void operator()(const G4ParticleDefinition& particle,
                    const G4VProcess&           process);

  private:
    // Loop over EM processes and write tables to the ROOT file
    void fill_em_tables(const G4VEmProcess& em_process);
    // Loop over energy loss processes and write tables to the ROOT file
    void fill_energy_loss_tables(const G4VEnergyLossProcess& eloss_process);
    // Loop over multiple scattering processes and write tables to the ROOT
    // file
    void
    fill_multiple_scattering_tables(const G4VMultipleScattering& msc_process);
    // Write the remaining elements of this->table_ and fill the tables TTree
    void add_table(const G4PhysicsTable*      table,
                   celeritas::ImportTableType table_type);

  private:
    TFile* root_file_;
    // Whether to write tables that aren't used by physics
    TableSelection which_tables_;

    // TTree created by the constructor
    std::unique_ptr<TTree> tree_process_;
    // Temporary processs data for writing to the tree
    celeritas::ImportProcess process_;

    // Keep track of processes and tables already written
    struct PrevProcess
    {
        const G4ParticleDefinition* particle;
    };
    struct PrevTable
    {
        int                           particle_pdg;
        celeritas::ImportProcessClass process_class;
        celeritas::ImportTableType    table_type;
    };
    std::unordered_map<const G4VProcess*, PrevProcess>   written_processes_;
    std::unordered_map<const G4PhysicsTable*, PrevTable> written_tables_;
};

//---------------------------------------------------------------------------//
} // namespace geant_exporter
