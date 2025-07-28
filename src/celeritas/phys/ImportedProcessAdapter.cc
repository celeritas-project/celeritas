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
#include <utility>

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/SoftEqual.hh"
#include "celeritas/Types.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportPhysicsTable.hh"

#include "Applicability.hh"
#include "PDGNumber.hh"
#include "ParticleParams.hh"  // IWYU pragma: keep

namespace celeritas
{
namespace
{
using SpanConstDbl = Span<double const>;

bool is_nonnegative(SpanConstDbl vec)
{
    return std::all_of(vec.begin(), vec.end(), [](double v) { return v >= 0; });
}

bool is_contiguous_increasing(inp::UniformGrid const& lower,
                              inp::UniformGrid const& upper)
{
    return lower.y.size() >= 2 && upper.y.size() >= 2
           && std::exp(lower.x[Bound::lo]) > 0
           && lower.x[Bound::hi] > lower.x[Bound::lo]
           && upper.x[Bound::hi] > upper.x[Bound::lo]
           && soft_equal(lower.x[Bound::hi], upper.x[Bound::lo]);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
IPAContextException::IPAContextException(ParticleId id,
                                         ImportProcessClass ipc,
                                         PhysMatId mid)
{
    std::stringstream os;
    os << "Particle ID=" << id.unchecked_get() << ", process '" << ipc
       << ", material ID=" << mid.unchecked_get();
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
    CELER_EXPECT(
        std::all_of(data.processes.begin(), data.processes.end(), Identity{}));
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
                       << ip.process_class << "' for PDG{" << ip.particle_pdg
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
        auto particle_id = particles->find(pdg);
        CELER_VALIDATE(particle_id,
                       << "particle PDG{" << pdg.get()
                       << "} was not loaded (needed for '" << process_class
                       << "')");

        ids_[particle_id] = imported_->find({pdg, process_class});
        CELER_VALIDATE(ids_[particle_id],
                       << "imported process data is unavalable for PDG{"
                       << pdg.get() << "} (needed for '" << process_class
                       << "')");
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
auto ImportedProcessAdapter::macro_xs(Applicability const& applic) const
    -> XsGrid
{
    CELER_EXPECT(ids_.count(applic.particle));
    CELER_EXPECT(applic.material);

    auto const& process_id = ids_.find(applic.particle)->second;
    ImportProcess const& import_process = imported_->get(process_id);

    // Get cross section tables
    XsGrid result;
    if (import_process.lambda)
    {
        CELER_ASSERT(applic.material < import_process.lambda.grids.size());
        auto grid = import_process.lambda.grids[applic.material.get()];
        CELER_ASSERT(grid);
        CELER_ASSERT(std::exp(grid.x[Bound::lo]) > 0 && grid.y.size() >= 2);
        CELER_ASSERT(is_nonnegative(make_span(grid.y)));
        result.lower = std::move(grid);
    }
    if (import_process.lambda_prim)
    {
        CELER_ASSERT(applic.material < import_process.lambda_prim.grids.size());
        auto grid = import_process.lambda_prim.grids[applic.material.get()];
        CELER_ASSERT(grid);
        CELER_ASSERT(std::exp(grid.x[Bound::lo]) > 0 && grid.y.size() >= 2);
        CELER_ASSERT(is_nonnegative(make_span(grid.y)));
        result.upper = std::move(grid);
    }
    if (result.lower && result.upper)
    {
        CELER_ASSERT(is_contiguous_increasing(result.lower, result.upper));
        CELER_ASSERT(
            soft_equal(result.lower.x[Bound::hi], result.upper.x[Bound::lo]));
        CELER_ASSERT(soft_equal(
            result.lower.y.back(),
            result.upper.y.front() / std::exp(result.upper.x[Bound::lo])));
        result.lower.x[Bound::hi] = result.upper.x[Bound::lo];
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the energy loss for the given material and particle.
 */
auto ImportedProcessAdapter::energy_loss(Applicability const& applic) const
    -> EnergyLossGrid
{
    CELER_EXPECT(ids_.count(applic.particle));
    CELER_EXPECT(applic.material);

    auto const& process_id = ids_.find(applic.particle)->second;
    ImportProcess const& import_process = imported_->get(process_id);

    EnergyLossGrid result;
    if (import_process.dedx)
    {
        CELER_ASSERT(applic.material < import_process.dedx.grids.size());
        auto grid = import_process.dedx.grids[applic.material.get()];
        CELER_ASSERT(grid);
        CELER_ASSERT(std::exp(grid.x[Bound::lo]) > 0 && grid.y.size() >= 2);
        result = std::move(grid);
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
