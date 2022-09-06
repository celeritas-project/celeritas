//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ImportedModelAdapter.cc
//---------------------------------------------------------------------------//
#include "ImportedModelAdapter.hh"

#include "celeritas/Types.hh"
#include "celeritas/grid/ValueGridBuilder.hh"
#include "celeritas/io/ImportPhysicsVector.hh"

#include "PDGNumber.hh"
#include "ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from shared process data.
 */
ImportedModelAdapter::ImportedModelAdapter(SPConstImported       imported,
                                           const ParticleParams& particles,
                                           ImportProcessClass    process_class,
                                           ImportModelClass      model_class,
                                           SpanConstPDG          pdg_numbers)
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
    SPConstImported                  imported,
    const ParticleParams&            particles,
    ImportProcessClass               process_class,
    ImportModelClass                 model_class,
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

    // Get the imported process that applies for the given particle
    auto proc = particle_to_process_.find(applic.particle);
    CELER_ASSERT(proc != particle_to_process_.end());
    const ImportProcess& import_process = imported_->get(proc->second);

    // Get the micro xs grids for the given model, particle, and material
    auto xs = import_process.micro_xs.find(model_class_);
    CELER_ASSERT(xs != import_process.micro_xs.end());
    CELER_ASSERT(applic.material < xs->second.size());
    const auto& elem_phys_vectors = xs->second[applic.material.get()];

    MicroXsBuilders builders;

    for (size_type i : range(elem_phys_vectors.size()))
    {
        const auto& vec = elem_phys_vectors.at(i);
        CELER_ASSERT(vec.vector_type == ImportPhysicsVectorType::log);
        ElementId el{i};
        builders[el] = ValueGridLogBuilder::from_geant(make_span(vec.x),
                                                       make_span(vec.y));
    }

    return builders;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
