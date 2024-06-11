//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
#include "celeritas/Types.hh"
#include "celeritas/grid/ValueGridBuilder.hh"
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
auto ImportedModelAdapter::micro_xs(Applicability applic) const
    -> MicroXsBuilders
{
    CELER_EXPECT(applic.material);

    // Get the micro xs grids for the given model, particle, and material
    ImportModel const& model = this->get_model(applic.particle);
    CELER_ASSERT(applic.material < model.materials.size());
    ImportModelMaterial const& imm
        = model.materials[applic.material.unchecked_get()];

    MicroXsBuilders builders(imm.micro_xs.size());
    for (size_type elcomp_idx : range(builders.size()))
    {
        builders[elcomp_idx] = ValueGridLogBuilder::from_geant(
            make_span(imm.energy), make_span(imm.micro_xs[elcomp_idx]));
    }
    return builders;
}

//---------------------------------------------------------------------------//
/*!
 * Get the xs energy grid bounds for the given material and particle.
 */
auto ImportedModelAdapter::energy_grid_bounds(ParticleId pid,
                                              MaterialId mid) const
    -> EnergyBounds
{
    CELER_EXPECT(pid && mid);

    auto const& xs = this->get_model(pid).materials;
    CELER_ASSERT(mid < xs.size());
    EnergyBounds result{Energy(xs[mid.get()].energy.front()),
                        Energy(xs[mid.get()].energy.back())};

    CELER_ENSURE(result[0] < result[1]);
    return result;
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
                   << "missing imported model " << to_cstring(model_class_));
    return *mod_iter;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
