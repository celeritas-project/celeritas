//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/io/ImportProcess.hh"

#include "Applicability.hh"
#include "ImportedProcessAdapter.hh"
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
    using Energy = units::MevEnergy;
    using EnergyBounds = Array<Energy, 2>;
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

    // Get the xs energy grid bounds for the given material and particle
    EnergyBounds energy_grid_bounds(ParticleId, PhysMatId) const;

    // Get the low energy limit for the given particle
    Energy low_energy_limit(ParticleId) const;

    // Get the high energy limit for the given particle
    Energy high_energy_limit(ParticleId) const;

  private:
    //// TYPES ////

    using ImportProcessId = ImportedProcesses::ImportProcessId;

    //// DATA ////

    SPConstImported imported_;
    ImportModelClass model_class_;
    std::unordered_map<ParticleId, ImportProcessId> particle_to_process_;

    //// HELPER FUNCTIONS ////

    ImportModel const& get_model(ParticleId) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
