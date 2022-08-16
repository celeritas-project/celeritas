//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/ImportProcessConverter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "celeritas/io/ImportElement.hh"
#include "celeritas/io/ImportMaterial.hh"
#include "celeritas/io/ImportProcess.hh"

using celeritas::ImportElement;
using celeritas::ImportMaterial;
using celeritas::ImportPhysicsVector;
using celeritas::ImportProcess;

class TFile;
class TTree;

class G4VProcess;
class G4VEmProcess;
class G4VEmModel;
class G4VEnergyLossProcess;
class G4VMultipleScattering;
class G4ParticleDefinition;
class G4PhysicsTable;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
enum class TableSelection
{
    minimal, //!< Store only lambda, dedx, and range
    all
};

//---------------------------------------------------------------------------//
/*!
 * Simplify the convoluted mechanism to store Geant4 process, model, and XS
 * table data.
 *
 * \c Operator() is expected to be used while looping over Geant4 particle and
 * process lists, and it returns a populated \c ImportProcess object. If said
 * process was already imported during a previous loop, it will return an
 * empty object. \c ImportProcess has an operator bool to check if said object
 * is not empty before adding it to the \c vector<ImportProcess> member of
 * \c ImportData .
 *
 * \code
 *  std::vector<ImportProcess> processes;
 *  ImportProcessConverter import(TableSelection::all, materials, elements);
 *
 *  G4ParticleTable::G4PTblDicIterator& particle_iterator
 *      = *(G4ParticleTable::GetParticleTable()->GetIterator());
 *  particle_iterator.reset();
 *
 *  while (particle_iterator())
 *  {
 *      const G4ParticleDefinition& g4_particle_def
 *          = *(particle_iterator.value());
 *      const G4ProcessVector& *process_list =
 *          *g4_particle_def.GetProcessManager()->GetProcessList();
 *
 *      for (int j = 0; j < process_list.size(); j++)
 *      {
 *          if (ImportProcess process
 *                  = import(g4_particle_def, *process_list[j]))
 *          {
 *              processes.push_back(std::move(process));
 *          }
 *      }
 *  }
 * \endcode
 */
class ImportProcessConverter
{
  public:
    // Construct with selected list of tables
    ImportProcessConverter(TableSelection                     which_tables,
                           const std::vector<ImportMaterial>& materials,
                           const std::vector<ImportElement>&  elements);

    // Default destructor
    ~ImportProcessConverter();

    // Return ImportProcess for a given particle and physics process
    ImportProcess operator()(const G4ParticleDefinition& particle,
                             const G4VProcess&           process);

  private:
    // Loop over EM processes and store them in this->process_
    void store_em_tables(const G4VEmProcess& em_process);

    // Loop over energy loss processes and store them in this->process_
    void store_energy_loss_tables(const G4VEnergyLossProcess& eloss_process);

    // Loop over multiple scattering processes and store them in this->process_
    void
    store_multiple_scattering_tables(const G4VMultipleScattering& msc_process);

    // Write the remaining elements of this->process_
    void add_table(const G4PhysicsTable*      table,
                   celeritas::ImportTableType table_type);

    // Store element cross section data into physics vectors
    ImportProcess::ModelMicroXS add_micro_xs(G4VEmModel& model);

    // Set up the physics vector energy grid for add_micro_xs(...)
    ImportPhysicsVector
    initialize_micro_xs_physics_vector(G4VEmModel& model, unsigned int mat_id);

  private:
    // Store material and element information for the element selector tables
    std::vector<ImportMaterial> materials_;
    std::vector<ImportElement>  elements_;

    // Whether to write tables that aren't used by physics
    TableSelection which_tables_;

    // Temporary process data returned by operator()
    ImportProcess process_;

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
} // namespace detail
} // namespace celeritas
