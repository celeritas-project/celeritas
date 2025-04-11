//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ImportedProcessAdapter.cc
//---------------------------------------------------------------------------//
#include "ImportedProcessAdapter.hh"

#include <algorithm>
#include <exception>
#include <tuple>
#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/ValueGridBuilder.hh"
#include "celeritas/grid/ValueGridType.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportPhysicsTable.hh"

#include "Applicability.hh"
#include "PDGNumber.hh"
#include "ParticleParams.hh"  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
IPAContextException::IPAContextException(ParticleId id,
                                         ImportProcessClass ipc,
                                         MaterialId mid)
{
    std::stringstream os;
    os << "Particle ID=" << id.unchecked_get() << ", process '"
       << to_cstring(ipc) << ", material ID=" << mid.unchecked_get();
    what_ = os.str();
}

//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
std::shared_ptr<ImportedProcesses>
ImportedProcesses::from_import(ImportData const& data,
                               SPConstParticles particle_params)
{
    CELER_EXPECT(std::all_of(
        data.processes.begin(), data.processes.end(), LogicalTrue{}));
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
    : imported_(std::move(imported)), process_class_(process_class)
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
auto ImportedProcessAdapter::step_limits(Applicability const& applic) const
    -> StepLimitBuilders
{
    try
    {
        return this->step_limits_impl(applic);
    }
    catch (...)
    {
        std::throw_with_nested(IPAContextException(
            applic.particle, process_class_, applic.material));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given material and particle.
 */
auto ImportedProcessAdapter::step_limits_impl(
    Applicability const& applic) const -> StepLimitBuilders
{
    CELER_EXPECT(ids_.count(applic.particle));
    CELER_EXPECT(applic.material);

    // Get list of physics tables
    ParticleProcessIds const& ids = ids_.find(applic.particle)->second;
    ImportProcess const& import_process = imported_->get(ids.process);

    auto get_vector = [&applic, &import_process](ImportTableId table_id) {
        CELER_ASSERT(table_id < import_process.tables.size());
        ImportPhysicsTable const& tab = import_process.tables[table_id.get()];
        CELER_ASSERT(applic.material < tab.physics_vectors.size());
        return tab.physics_vectors[applic.material.get()];
    };

    StepLimitBuilders builders;

    // Construct cross section tables
    if (import_process.process_class == ImportProcessClass::msc)
    {
        // No cross sections
    }
    else if (ids.lambda && ids.lambda_prim)
    {
        // Both unscaled and scaled values are present
        builders[ValueGridType::macro_xs] = ValueGridXsBuilder::from_geant(
            get_vector(ids.lambda), get_vector(ids.lambda_prim));
    }
    else if (ids.lambda_prim)
    {
        // Only high-energy (energy-scale) cross sections are presesnt
        builders[ValueGridType::macro_xs]
            = ValueGridXsBuilder::from_scaled(get_vector(ids.lambda_prim));
    }
    else if (ids.lambda)
    {
        builders[ValueGridType::macro_xs]
            = ValueGridLogBuilder::from_geant(get_vector(ids.lambda));
    }

    // Construct slowing-down data
    if (ids.dedx)
    {
        builders[ValueGridType::energy_loss]
            = ValueGridLogBuilder::from_geant(get_vector(ids.dedx));
    }

    return builders;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
