//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhotoelectricProcess.cc
//---------------------------------------------------------------------------//
#include "PhotoelectricProcess.hh"

#include <utility>
#include "LivermorePEModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
PhotoelectricProcess::PhotoelectricProcess(SPConstParticles   particles,
                                           SPConstMaterials   materials,
                                           ImportPhysicsTable xs)
    : particles_(std::move(particles))
    , materials_(std::move(materials))
    , xs_(std::move(xs))
{
    CELER_EXPECT(particles_);
    CELER_EXPECT(materials_);
    CELER_EXPECT(xs_.table_type == ImportTableType::lambda_prim);
    CELER_EXPECT(!xs_.physics_vectors.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Construct with atomic relaxation data.
 */
PhotoelectricProcess::PhotoelectricProcess(SPConstParticles   particles,
                                           SPConstMaterials   materials,
                                           ImportPhysicsTable xs,
                                           SPConstAtomicRelax atomic_relaxation,
                                           size_type vacancy_stack_size)
    : PhotoelectricProcess(
        std::move(particles), std::move(materials), std::move(xs))
{
    atomic_relaxation_  = std::move(atomic_relaxation);
    vacancy_stack_size_ = vacancy_stack_size;
    CELER_ENSURE(atomic_relaxation_);
    CELER_ENSURE(vacancy_stack_size_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto PhotoelectricProcess::build_models(ModelIdGenerator next_id) const
    -> VecModel
{
    LivermorePEModel::ReadData load_data;
    if (atomic_relaxation_)
    {
        // Construct model with atomic relaxation enabled
        return {std::make_shared<LivermorePEModel>(next_id(),
                                                   *particles_,
                                                   *materials_,
                                                   load_data,
                                                   atomic_relaxation_,
                                                   vacancy_stack_size_)};
    }
    else
    {
        // Construct model without atomic relaxation
        return {std::make_shared<LivermorePEModel>(
            next_id(), *particles_, *materials_, load_data)};
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto PhotoelectricProcess::step_limits(Applicability range) const
    -> StepLimitBuilders
{
    CELER_EXPECT(range.material < xs_.physics_vectors.size());
    CELER_EXPECT(range.particle == particles_->find(pdg::gamma()));

    const auto& vec = xs_.physics_vectors[range.material.get()];
    CELER_ASSERT(vec.vector_type == ImportPhysicsVectorType::log);

    StepLimitBuilders builders;
    builders[size_type(ValueGridType::macro_xs)]
        = std::make_unique<ValueGridXsBuilder>(
            vec.x.front(), vec.x.front(), vec.x.back(), vec.y);

    return builders;
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string PhotoelectricProcess::label() const
{
    return "Photoelectric effect";
}

//---------------------------------------------------------------------------//
} // namespace celeritas
