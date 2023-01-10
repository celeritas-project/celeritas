//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ImportedProcessAdapter.cc
//---------------------------------------------------------------------------//
#include "ImportedProcessAdapter.hh"

#include <algorithm>
#include <tuple>
#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/ValueGridBuilder.hh"
#include "celeritas/grid/ValueGridData.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportPhysicsTable.hh"
#include "celeritas/phys/Applicability.hh"

#include "PDGNumber.hh"
#include "ParticleParams.hh"  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
std::shared_ptr<ImportedProcesses>
ImportedProcesses::from_import(ImportData const& data,
                               SPConstParticles particle_params)
{
    CELER_EXPECT(std::all_of(
        data.processes.begin(),
        data.processes.end(),
        [](ImportProcess const& ip) { return static_cast<bool>(ip); }));
    CELER_EXPECT(particle_params);

    // Sort processes based on particle def IDs, process types, etc.
    auto processes = data.processes;
    auto particles = std::move(particle_params);

    auto to_process_key = [&particles](ImportProcess const& ip) {
        return std::make_tuple(particles->find(PDGNumber{ip.particle_pdg}),
                               ip.process_class);
    };

    std::sort(processes.begin(),
              processes.end(),
              [&to_process_key](ImportProcess const& left,
                                ImportProcess const& right) {
                  return to_process_key(left) < to_process_key(right);
              });

    return std::make_shared<ImportedProcesses>(std::move(processes));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with imported tabular data.
 */
ImportedProcesses::ImportedProcesses(std::vector<ImportProcess> io)
    : processes_(std::move(io))
{
    for (auto id : range(ImportProcessId{this->size()}))
    {
        ImportProcess const& ip = processes_[id.get()];

        auto insertion = ids_.insert(
            {key_type{PDGNumber{ip.particle_pdg}, ip.process_class}, id});
        CELER_VALIDATE(insertion.second,
                       << "encountered duplicate imported process class '"
                       << to_cstring(ip.process_class) << "' for PDG{"
                       << ip.particle_pdg
                       << "} (each particle must have at most one process of "
                          "a given type)");
    }

    CELER_ENSURE(processes_.size() == ids_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Return physics tables for a particle type and process.
 *
 * Returns 'invalid' ID if process is not present for the given particle type.
 */
auto ImportedProcesses::find(key_type particle_process) const -> ImportProcessId
{
    auto iter = ids_.find(particle_process);
    if (iter == ids_.end())
        return {};

    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from shared process data.
 */
ImportedProcessAdapter::ImportedProcessAdapter(SPConstImported imported,
                                               SPConstParticles const& particles,
                                               ImportProcessClass process_class,
                                               SpanConstPDG pdg_numbers)
    : imported_(std::move(imported))
{
    CELER_EXPECT(particles);
    CELER_EXPECT(!pdg_numbers.empty());
    for (PDGNumber pdg : pdg_numbers)
    {
        ParticleProcessIds proc_ids;
        proc_ids.process = imported_->find({pdg, process_class});
        CELER_VALIDATE(proc_ids.process,
                       << "imported process data is unavalable for PDG{"
                       << pdg.get() << "} (needed for '"
                       << to_cstring(process_class) << "')");

        // Loop through available tables
        auto const& tables = imported_->get(proc_ids.process).tables;
        for (auto table_id : range(ImportTableId(tables.size())))
        {
            // Map table types to IDs in our imported data
            ImportTableId* dst = nullptr;
            switch (tables[table_id.get()].table_type)
            {
                case ImportTableType::dedx:
                    dst = &proc_ids.dedx;
                    break;
                case ImportTableType::range:
                    dst = &proc_ids.range;
                    break;
                case ImportTableType::lambda:
                    dst = &proc_ids.lambda;
                    break;
                case ImportTableType::lambda_prim:
                    dst = &proc_ids.lambda_prim;
                    break;
                default:
                    //  Not a data type we care about
                    continue;
            }
            CELER_ASSERT(dst);
            CELER_VALIDATE(!*dst,
                           << "duplicate table type '"
                           << to_cstring(tables[table_id.get()].table_type)
                           << "' in process data for '"
                           << to_cstring(process_class)
                           << "' (each type should be unique to a process for "
                              "a given partice)");
            *dst = table_id;
        }

        auto particle_id = particles->find(pdg);
        CELER_VALIDATE(particle_id,
                       << "particle PDG{" << pdg.get()
                       << "} was not loaded (needed for '"
                       << to_cstring(process_class) << "')");

        // Save process data IDs for this particle type
        ids_[particle_id] = proc_ids;
    }
    CELER_ENSURE(ids_.size() == pdg_numbers.size());
}

//---------------------------------------------------------------------------//
/*!
 * Delegating constructor for a list of particles.
 */
ImportedProcessAdapter::ImportedProcessAdapter(
    SPConstImported imported,
    SPConstParticles const& particles,
    ImportProcessClass process_class,
    std::initializer_list<PDGNumber> pdg_numbers)
    : ImportedProcessAdapter(std::move(imported),
                             particles,
                             std::move(process_class),
                             {pdg_numbers.begin(), pdg_numbers.end()})
{
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given material and particle.
 */
auto ImportedProcessAdapter::step_limits(Applicability range) const
    -> StepLimitBuilders
{
    CELER_EXPECT(ids_.count(range.particle));
    CELER_EXPECT(range.material);

    // Get list of physics tables
    ParticleProcessIds const& ids = ids_.find(range.particle)->second;
    ImportProcess const& import_process = imported_->get(ids.process);

    auto get_vector = [&range, &import_process](ImportTableId table_id) {
        CELER_ASSERT(table_id < import_process.tables.size());
        const ImportPhysicsTable& tab = import_process.tables[table_id.get()];
        CELER_ASSERT(range.material < tab.physics_vectors.size());
        return tab.physics_vectors[range.material.get()];
    };

    StepLimitBuilders builders;

    // Construct cross section tables
    if (ids.lambda && ids.lambda_prim)
    {
        // Both unscaled and scaled values are present
        auto const& lo = get_vector(ids.lambda);
        CELER_ASSERT(lo.vector_type == ImportPhysicsVectorType::log);
        auto const& hi = get_vector(ids.lambda_prim);
        CELER_ASSERT(hi.vector_type == ImportPhysicsVectorType::log);
        builders[ValueGridType::macro_xs] = ValueGridXsBuilder::from_geant(
            make_span(lo.x), make_span(lo.y), make_span(hi.x), make_span(hi.y));
    }
    else if (ids.lambda_prim)
    {
        // Only high-energy (energy-scale) cross sections are presesnt
        auto const& vec = get_vector(ids.lambda_prim);
        CELER_ASSERT(vec.vector_type == ImportPhysicsVectorType::log);
        builders[ValueGridType::macro_xs] = ValueGridXsBuilder::from_scaled(
            make_span(vec.x), make_span(vec.y));
    }
    else if (ids.lambda)
    {
        // Only low-energy cross sections are presesnt
        auto const& vec = get_vector(ids.lambda);
        CELER_ASSERT(vec.vector_type == ImportPhysicsVectorType::log);

        ValueGridType vgt
            = (import_process.process_class == ImportProcessClass::msc)
                  ? ValueGridType::msc_mfp
                  : ValueGridType::macro_xs;
        builders[vgt] = ValueGridLogBuilder::from_geant(make_span(vec.x),
                                                        make_span(vec.y));
    }

    // Construct slowing-down data
    if (ids.dedx)
    {
        auto const& vec = get_vector(ids.dedx);
        CELER_ASSERT(vec.vector_type == ImportPhysicsVectorType::log);
        builders[ValueGridType::energy_loss] = ValueGridLogBuilder::from_geant(
            make_span(vec.x), make_span(vec.y));
    }

    // Construct range limiters
    if (ids.range)
    {
        auto const& vec = get_vector(ids.range);
        CELER_ASSERT(vec.vector_type == ImportPhysicsVectorType::log);
        builders[ValueGridType::range] = ValueGridLogBuilder::from_range(
            make_span(vec.x), make_span(vec.y));
    }

    return builders;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
