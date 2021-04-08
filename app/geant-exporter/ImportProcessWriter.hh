//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportProcessWriter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iostream>
#include <unordered_map>
#include <vector>
#include <memory>

#include "io/detail/ImportProcess.hh"

using celeritas::ImportProcess;

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
 * Simplify the convoluted mechanism to store Geant4 process, model, and XS
 * table data for each available particle. It is expected to be used within a
 * Geant4 particle and process loops. After all processes have been stored, a
 * vector<ImportProcess> can be retrieved:
 *
 * \code
 *  ImportProcessWriter process_writer(TableSelection::all);
 *
 *  G4ParticleTable::G4PTblDicIterator& particle_iterator;
 *  while (particle_iterator())
 *  {
 *      const G4ProcessVector& process_list
 *            = *g4_particle_def.GetProcessManager()->GetProcessList();
 *
 *      for (int j; j < process_list.size(); j++)
 *      {
 *          process_writer(g4_particle_def, *process_list[j]);
 *      }
 *  }
 *  std::vector<ImportProcess> processes = process_writer.get();
 * \endcode
 */
class ImportProcessWriter
{
  public:
    // Construct with selected list of tables
    ImportProcessWriter(TableSelection which_tables);

    // Default destructor
    ~ImportProcessWriter();

    // Write the physics tables from a given particle and physics process
    // Expected to be called within a G4ParticleTable iterator loop
    void operator()(const G4ParticleDefinition& particle,
                    const G4VProcess&           process);

    // Fetch the full list of physics processes created using operator()
    std::vector<ImportProcess> get() { return processes_; }

  private:
    // Loop over EM processes and store them in processes_
    void store_em_tables(const G4VEmProcess& em_process);

    // Loop over energy loss processes and store them in processes_
    void store_energy_loss_tables(const G4VEnergyLossProcess& eloss_process);

    // Loop over multiple scattering processes and store them in processes_
    void
    store_multiple_scattering_tables(const G4VMultipleScattering& msc_process);

    // Write the remaining elements of this->process_
    void add_table(const G4PhysicsTable*      table,
                   celeritas::ImportTableType table_type);

  private:
    // Stored process data
    std::vector<ImportProcess> processes_;

    // Whether to write tables that aren't used by physics
    TableSelection which_tables_;

    // Temporary processs data for writing processes_
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
