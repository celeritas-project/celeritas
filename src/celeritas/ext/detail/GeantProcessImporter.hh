//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantProcessImporter.hh
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
    minimal,  //!< Store only lambda, dedx, and range
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
 *  GeantProcessImporter import(TableSelection::all, materials, elements);
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
class GeantProcessImporter
{
  public:
    // Construct with selected list of tables
    GeantProcessImporter(TableSelection which_tables,
                         std::vector<ImportMaterial> const& materials,
                         std::vector<ImportElement> const& elements);

    // Default destructor
    ~GeantProcessImporter();

    // Return ImportProcess for a given particle and physics process
    ImportProcess operator()(G4ParticleDefinition const& particle,
                             G4VProcess const& process);

  private:
    // Save common process attributes
    template<class T>
    void store_common_process(T const& process);
    // Save "discrete" process
    void store_em_process(G4VEmProcess const& em_process);
    // Save "continuous/discrete" process
    void store_eloss_process(G4VEnergyLossProcess const& eloss_process);
    // Save multiple scattering data to this->process_
    void store_msc_process(G4VMultipleScattering const& msc_process);

    // Write the remaining elements of this->process_
    void add_table(G4PhysicsTable const* table,
                   celeritas::ImportTableType table_type);

  private:
    // Store material and element information for the element selector tables
    std::vector<ImportMaterial> const& materials_;
    std::vector<ImportElement> const& elements_;

    // Whether to write tables that aren't used by physics
    TableSelection which_tables_;

    // Temporary process data returned by operator()
    ImportProcess process_;

    // Keep track of processes and tables already written
    struct PrevProcess
    {
        G4ParticleDefinition const* particle;
    };

    struct PrevTable
    {
        int particle_pdg;
        celeritas::ImportProcessClass process_class;
        celeritas::ImportTableType table_type;
    };

    std::unordered_map<G4VProcess const*, PrevProcess> written_processes_;
    std::unordered_map<G4PhysicsTable const*, PrevTable> written_tables_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
