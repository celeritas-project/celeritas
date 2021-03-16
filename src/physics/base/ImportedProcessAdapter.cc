//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportedProcessAdapter.cc
//---------------------------------------------------------------------------//
#include "ImportedProcessAdapter.hh"

#include "base/Range.hh"
#include "physics/base/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with imported tabular data.
 */
ImportedProcesses::ImportedProcesses(std::vector<ImportProcess> io)
    : processes_(std::move(io))
{
    for (auto id : range(ImportProcessId{this->size()}))
    {
        const ImportProcess& ip = processes_[id.get()];

        auto insertion = ids_.insert(
            {key_type{PDGNumber{ip.particle_pdg}, ip.process_class}, id});
        CELER_VALIDATE(insertion.second,
                       "Encountered duplicate imported process class '"
                           << to_cstring(ip.process_class) << "' for PDG{"
                           << ip.particle_pdg << "}");
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
                                               const SPConstParticles& particles,
                                               ImportProcessClass process_class,
                                               SpanConstPDG       pdg_numbers)
    : imported_(std::move(imported))
{
    CELER_EXPECT(particles);
    CELER_EXPECT(!pdg_numbers.empty());
    for (PDGNumber pdg : pdg_numbers)
    {
        ParticleProcessIds proc_ids;
        proc_ids.process = imported_->find({pdg, process_class});
        CELER_VALIDATE(proc_ids.process,
                       "Imported process data is unavalable for '"
                           << to_cstring(process_class) << "' for PDG{"
                           << pdg.get() << "}");

        // Loop through available tables
        const auto& tables = imported_->get(proc_ids.process).tables;
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
                           "Duplicate table type '"
                               << to_cstring(tables[table_id.get()].table_type)
                               << "' in process data for '"
                               << to_cstring(process_class) << "'");
            *dst = table_id;
        }

        auto particle_id = particles->find(pdg);
        CELER_VALIDATE(particle_id,
                       "Particle PDG{" << pdg.get() << "} is not available");

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
    SPConstImported                  imported,
    const SPConstParticles&          particles,
    ImportProcessClass               process_class,
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
    const ParticleProcessIds& ids = ids_.find(range.particle)->second;
    const ImportProcess&      import_process = imported_->get(ids.process);

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
        const auto& lo = get_vector(ids.lambda);
        CELER_ASSERT(lo.vector_type == ImportPhysicsVectorType::log);
        const auto& hi = get_vector(ids.lambda_prim);
        CELER_ASSERT(hi.vector_type == ImportPhysicsVectorType::log);
        builders[ValueGridType::macro_xs] = ValueGridXsBuilder::from_geant(
            make_span(lo.x), make_span(lo.y), make_span(hi.x), make_span(hi.y));
    }
    else if (ids.lambda_prim)
    {
        // Only high-energy (energy-scale) cross sections are presesnt
        const auto& vec = get_vector(ids.lambda_prim);
        CELER_ASSERT(vec.vector_type == ImportPhysicsVectorType::log);
        builders[ValueGridType::macro_xs] = ValueGridXsBuilder::from_scaled(
            make_span(vec.x), make_span(vec.y));
    }
    else if (ids.lambda)
    {
        // Only low-energy cross sections are presesnt
        const auto& vec = get_vector(ids.lambda);
        CELER_ASSERT(vec.vector_type == ImportPhysicsVectorType::log);
        builders[ValueGridType::macro_xs] = ValueGridLogBuilder::from_geant(
            make_span(vec.x), make_span(vec.y));
    }

    // Construct slowing-down data
    if (ids.dedx)
    {
        const auto& vec = get_vector(ids.dedx);
        CELER_ASSERT(vec.vector_type == ImportPhysicsVectorType::log);
        builders[ValueGridType::energy_loss] = ValueGridLogBuilder::from_geant(
            make_span(vec.x), make_span(vec.y));
    }

    // Construct range limiters
    if (ids.range)
    {
        const auto& vec = get_vector(ids.range);
        CELER_ASSERT(vec.vector_type == ImportPhysicsVectorType::log);
        builders[ValueGridType::range] = ValueGridLogBuilder::from_geant(
            make_span(vec.x), make_span(vec.y));
    }

    return builders;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
