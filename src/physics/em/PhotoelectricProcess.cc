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
                                           ImportPhysicsTable xs_lo,
                                           ImportPhysicsTable xs_hi,
                                           SPConstData        data)
    : particles_(std::move(particles))
    , xs_lo_(std::move(xs_lo))
    , xs_hi_(std::move(xs_hi))
    , data_(std::move(data))
{
    CELER_EXPECT(particles_);
    CELER_EXPECT(xs_lo_.table_type == ImportTableType::lambda);
    CELER_EXPECT(xs_hi_.table_type == ImportTableType::lambda_prim);
    CELER_EXPECT(!xs_lo_.physics_vectors.empty());
    CELER_EXPECT(xs_lo_.physics_vectors.size()
                 == xs_hi_.physics_vectors.size());
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with atomic relaxation data.
 */
PhotoelectricProcess::PhotoelectricProcess(SPConstParticles   particles,
                                           ImportPhysicsTable xs_lo,
                                           ImportPhysicsTable xs_hi,
                                           SPConstData        data,
                                           SPConstAtomicRelax atomic_relaxation,
                                           size_type vacancy_stack_size)
    : PhotoelectricProcess(std::move(particles),
                           std::move(xs_lo),
                           std::move(xs_hi),
                           std::move(data))
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
    if (atomic_relaxation_)
    {
        // Construct model with atomic relaxation enabled
        return {std::make_shared<LivermorePEModel>(next_id(),
                                                   *particles_,
                                                   *data_,
                                                   *atomic_relaxation_,
                                                   vacancy_stack_size_)};
    }
    else
    {
        // Construct model without atomic relaxation
        return {std::make_shared<LivermorePEModel>(
            next_id(), *particles_, *data_)};
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto PhotoelectricProcess::step_limits(Applicability range) const
    -> StepLimitBuilders
{
    CELER_EXPECT(range.material < xs_lo_.physics_vectors.size());
    CELER_EXPECT(range.particle == particles_->find(pdg::gamma()));

    const auto& lo = xs_lo_.physics_vectors[range.material.get()];
    const auto& hi = xs_hi_.physics_vectors[range.material.get()];
    CELER_ASSERT(lo.vector_type == ImportPhysicsVectorType::log);
    CELER_ASSERT(hi.vector_type == ImportPhysicsVectorType::log);

    StepLimitBuilders builders;
    builders[size_type(ValueGridType::macro_xs)]
        = ValueGridXsBuilder::from_geant(
            make_span(lo.x), make_span(lo.y), make_span(hi.x), make_span(hi.y));
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
