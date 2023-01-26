//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ImportedProcessAdapter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <initializer_list>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/cont/Span.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/ValueGridBuilder.hh"
#include "celeritas/io/ImportPhysicsTable.hh"
#include "celeritas/io/ImportProcess.hh"

#include "Applicability.hh"
#include "PDGNumber.hh"
#include "Process.hh"

namespace celeritas
{
class ParticleParams;
struct ImportData;

//---------------------------------------------------------------------------//
/*!
 * Manage imported physics data.
 */
class ImportedProcesses
{
  public:
    //!@{
    //! Type aliases
    using ImportProcessId = OpaqueId<ImportProcess>;
    using key_type = std::pair<PDGNumber, ImportProcessClass>;
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    //!@}

  public:
    // Construct with imported data
    static std::shared_ptr<ImportedProcesses>
    from_import(ImportData const& data, SPConstParticles particle_params);

    // Construct with imported tables
    explicit ImportedProcesses(std::vector<ImportProcess> io);

    // Return physics tables for a particle type and process
    ImportProcessId find(key_type) const;

    // Get the table for the given process ID
    inline ImportProcess const& get(ImportProcessId id) const;

    // Number of imported processes
    inline ImportProcessId::size_type size() const;

  private:
    std::vector<ImportProcess> processes_;
    std::map<key_type, ImportProcessId> ids_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct step limits from imported physics data.
 */
class ImportedProcessAdapter
{
  public:
    //!@{
    //! Type aliases
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using StepLimitBuilders = Process::StepLimitBuilders;
    using SpanConstPDG = Span<PDGNumber const>;
    //!@}

  public:
    // Construct from shared table data
    ImportedProcessAdapter(SPConstImported imported,
                           SPConstParticles const& particles,
                           ImportProcessClass process_class,
                           SpanConstPDG pdg_numbers);

    // Construct from shared table data
    ImportedProcessAdapter(SPConstImported imported,
                           SPConstParticles const& particles,
                           ImportProcessClass process_class,
                           std::initializer_list<PDGNumber> pdg_numbers);

    // Construct step limits from the given particle/material type
    StepLimitBuilders step_limits(Applicability applic) const;

    // Get the lambda table for the given particle ID
    inline ImportPhysicsTable const& get_lambda(ParticleId id) const;

    // Access the imported processes
    SPConstImported const& processes() const { return imported_; }

  private:
    using ImportTableId = OpaqueId<ImportPhysicsTable>;
    using ImportProcessId = ImportedProcesses::ImportProcessId;

    struct ParticleProcessIds
    {
        ImportProcessId process;
        ImportTableId lambda;
        ImportTableId lambda_prim;
        ImportTableId dedx;
        ImportTableId range;
    };

    SPConstImported imported_;
    std::map<ParticleId, ParticleProcessIds> ids_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get the table for the given process ID.
 */
ImportProcess const& ImportedProcesses::get(ImportProcessId id) const
{
    CELER_EXPECT(id < this->size());
    return processes_[id.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Number of imported processes.
 */
auto ImportedProcesses::size() const -> ImportProcessId::size_type
{
    return processes_.size();
}

//---------------------------------------------------------------------------//
/*!
 * Get cross sections for the given particle ID.
 *
 * This is currently used for loading MSC data for calculating mean free paths.
 */
ImportPhysicsTable const&
ImportedProcessAdapter::get_lambda(ParticleId id) const
{
    auto iter = ids_.find(id);
    CELER_EXPECT(iter != ids_.end());
    ImportTableId tab = iter->second.lambda;
    CELER_ENSURE(tab);
    return imported_->get(iter->second.process).tables[tab.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
