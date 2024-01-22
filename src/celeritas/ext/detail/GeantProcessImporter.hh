//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
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
#include "celeritas/io/ImportModel.hh"
#include "celeritas/io/ImportProcess.hh"

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
// DEPRECATED: remove in v0.5
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

    // Import processes
    ImportProcess operator()(G4ParticleDefinition const& particle,
                             G4VEmProcess const& process);
    ImportProcess operator()(G4ParticleDefinition const& particle,
                             G4VEnergyLossProcess const& process);
    std::vector<ImportMscModel>
    operator()(G4ParticleDefinition const& particle,
               G4VMultipleScattering const& process);

  private:
    //// TYPES ////

    struct PrevTable
    {
        int particle_pdg;
        celeritas::ImportProcessClass process_class;
        celeritas::ImportTableType table_type;
    };

    //// DATA ////

    // Store material and element information for the element selector tables
    std::vector<ImportMaterial> const& materials_;
    std::vector<ImportElement> const& elements_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
