//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ImportedModelAdapter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <initializer_list>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/cont/Span.hh"
#include "celeritas/Types.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"

#include "Applicability.hh"
#include "Model.hh"
#include "PDGNumber.hh"

namespace celeritas
{
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Construct microscopic cross section from imported physics data.
 */
class ImportedModelAdapter
{
  public:
    //!@{
    //! \name Type aliases
    using MicroXsBuilders = Model::MicroXsBuilders;
    using SpanConstPDG = Span<PDGNumber const>;
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    //!@}

  public:
    // Construct from shared table data
    ImportedModelAdapter(SPConstImported imported,
                         ParticleParams const& particles,
                         ImportProcessClass process_class,
                         ImportModelClass model_class,
                         SpanConstPDG pdg_numbers);

    // Construct from shared table data
    ImportedModelAdapter(SPConstImported imported,
                         ParticleParams const& particles,
                         ImportProcessClass process_class,
                         ImportModelClass model_class,
                         std::initializer_list<PDGNumber> pdg_numbers);

    // Construct micro cross sections from the given particle/material type
    MicroXsBuilders micro_xs(Applicability range) const;

  private:
    using ImportProcessId = ImportedProcesses::ImportProcessId;

    SPConstImported imported_;
    ImportModelClass model_class_;
    std::unordered_map<ParticleId, ImportProcessId> particle_to_process_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
