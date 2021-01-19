//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ComptonProcess.cc
//---------------------------------------------------------------------------//
#include "ComptonProcess.hh"

#include <utility>
#include "KleinNishinaModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
ComptonProcess::ComptonProcess(SPConstParticles   particles,
                               ImportPhysicsTable xs_lo,
                               ImportPhysicsTable xs_hi)
    : particles_(std::move(particles))
    , xs_lo_(std::move(xs_lo))
    , xs_hi_(std::move(xs_hi))
{
    CELER_EXPECT(particles_);
    CELER_EXPECT(xs_lo_.table_type == ImportTableType::lambda);
    CELER_EXPECT(xs_hi_.table_type == ImportTableType::lambda_prim);
    CELER_EXPECT(!xs_lo_.physics_vectors.empty());
    CELER_EXPECT(xs_lo_.physics_vectors.size()
                 == xs_hi_.physics_vectors.size());
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto ComptonProcess::build_models(ModelIdGenerator next_id) const -> VecModel
{
    return {std::make_shared<KleinNishinaModel>(next_id(), *particles_)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto ComptonProcess::step_limits(Applicability range) const -> StepLimitBuilders
{
    CELER_EXPECT(range.material < xs_lo_.physics_vectors.size());
    CELER_EXPECT(range.particle == particles_->find(pdg::gamma()));

    const auto& lo = xs_lo_.physics_vectors[range.material.get()];
    const auto& hi = xs_hi_.physics_vectors[range.material.get()];
    CELER_ASSERT(lo.vector_type == ImportPhysicsVectorType::log);
    CELER_ASSERT(hi.vector_type == ImportPhysicsVectorType::log);

    // The only Compton model we use is Klein-Nishina. In the case of more
    // refined energies, switch based on the given applicability or hope that
    // the input has already preprocessed the energy dependence.
    StepLimitBuilders builders;
    builders.macro_xs
        = std::make_unique<ValueGridXsBuilder>(ValueGridXsBuilder::from_geant(
            make_span(lo.x), make_span(lo.y), make_span(hi.x), make_span(hi.y)));
    return builders;
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string ComptonProcess::label() const
{
    return "Compton scattering";
}

//---------------------------------------------------------------------------//
} // namespace celeritas
