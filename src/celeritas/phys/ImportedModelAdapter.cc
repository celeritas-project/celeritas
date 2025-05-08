//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ImportedModelAdapter.cc
//---------------------------------------------------------------------------//
#include "ImportedModelAdapter.hh"

#include <map>
#include <type_traits>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/EnumStringMapper.hh"
#include "celeritas/Types.hh"
#include "celeritas/io/ImportModel.hh"

#include "Applicability.hh"
#include "PDGNumber.hh"
#include "ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from shared process data.
 */
ImportedModelAdapter::ImportedModelAdapter(SPConstImported imported,
                                           ParticleParams const& particles,
                                           ImportProcessClass process_class,
                                           ImportModelClass model_class,
                                           SpanConstPDG pdg_numbers)
    : imported_(std::move(imported)), model_class_(model_class)
{
    CELER_EXPECT(!pdg_numbers.empty());

    // Build a mapping of particle ID to imported process ID
    for (PDGNumber pdg : pdg_numbers)
    {
        auto particle_id = particles.find(pdg);
        CELER_ASSERT(particle_id);
        auto process_id = imported_->find({pdg, process_class});
        CELER_ASSERT(process_id);
        particle_to_process_[particle_id] = process_id;
    }

    CELER_ENSURE(particle_to_process_.size() == pdg_numbers.size());
}

//---------------------------------------------------------------------------//
/*!
 * Delegating constructor for a list of particles.
 */
ImportedModelAdapter::ImportedModelAdapter(
    SPConstImported imported,
    ParticleParams const& particles,
    ImportProcessClass process_class,
    ImportModelClass model_class,
    std::initializer_list<PDGNumber> pdg_numbers)
    : ImportedModelAdapter(std::move(imported),
                           particles,
                           process_class,
                           model_class,
                           {pdg_numbers.begin(), pdg_numbers.end()})
{
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given material and particle.
 */
auto ImportedModelAdapter::micro_xs(Applicability applic) const -> XsTable
{
    CELER_EXPECT(applic.material);

    // Get the micro xs grids for the given model, particle, and material
    ImportModel const& model = this->get_model(applic.particle);
    CELER_ASSERT(applic.material < model.materials.size());
    ImportModelMaterial const& imm
        = model.materials[applic.material.unchecked_get()];

    XsTable grids(imm.micro_xs.size());
    for (size_type elcomp_idx : range(grids.size()))
    {
        auto grid = imm.micro_xs[elcomp_idx];
        CELER_ASSERT(grid);
        CELER_ASSERT(std::exp(grid.x[Bound::lo]) > 0 && grid.y.size() >= 2);
        grids[elcomp_idx].lower = std::move(grid);
    }
    return grids;
}

//---------------------------------------------------------------------------//
/*!
 * Get the xs energy grid bounds for the given material and particle.
 */
auto ImportedModelAdapter::energy_grid_bounds(ParticleId pid,
                                              PhysMatId mid) const -> EnergyBounds
{
    CELER_EXPECT(pid && mid);

    auto const& xs = this->get_model(pid).materials;
    CELER_ASSERT(mid < xs.size());
    auto const& energy = xs[mid.get()].energy;

    CELER_ENSURE(energy[Bound::lo] < energy[Bound::hi]);
    return {Energy(energy[Bound::lo]), Energy(energy[Bound::hi])};
}

//---------------------------------------------------------------------------//
/*!
 * Get the model's low energy limit.
 *
 * Note that the model may not actually be valid down to this energy if the
 * production cut is larger than this value.
 */
auto ImportedModelAdapter::low_energy_limit(ParticleId pid) const -> Energy
{
    return Energy{
        static_cast<real_type>(this->get_model(pid).low_energy_limit)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the model's high energy limit.
 */
auto ImportedModelAdapter::high_energy_limit(ParticleId pid) const -> Energy
{
    return Energy{
        static_cast<real_type>(this->get_model(pid).high_energy_limit)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the imported model for the given particle.
 */
ImportModel const& ImportedModelAdapter::get_model(ParticleId particle) const
{
    // Get the imported process that applies for the given particle
    auto proc = particle_to_process_.find(particle);
    CELER_ASSERT(proc != particle_to_process_.end());
    ImportProcess const& import_process = imported_->get(proc->second);

    auto mod_iter = std::find_if(import_process.models.begin(),
                                 import_process.models.end(),
                                 [this](ImportModel const& m) {
                                     return m.model_class == model_class_;
                                 });
    CELER_VALIDATE(mod_iter != import_process.models.end(),
                   << "missing imported model " << model_class_);
    return *mod_iter;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
