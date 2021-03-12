//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EIonisationProcess.cc
//---------------------------------------------------------------------------//
#include "EIonisationProcess.hh"

#include "MollerBhabhaModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct process from host data.
 */
EIonisationProcess::EIonisationProcess(SPConstParticles   particles,
                                       ImportPhysicsTable xs)
    : particles_(std::move(particles)), xs_(std::move(xs))
{
    CELER_EXPECT(particles_);
    CELER_EXPECT(xs_.table_type == ImportTableType::lambda);
    CELER_EXPECT(!xs_.physics_vectors.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto EIonisationProcess::build_models(ModelIdGenerator next_id) const
    -> VecModel
{
    return {std::make_shared<MollerBhabhaModel>(next_id(), *particles_)};
}

//---------------------------------------------------------------------------//
/*!
 * Construct process from host data.
 */
auto EIonisationProcess::step_limits(Applicability range) const
    -> StepLimitBuilders
{
    CELER_EXPECT(range.material < xs_.physics_vectors.size());
    CELER_EXPECT(range.particle == particles_->find(pdg::electron())
                 || range.particle == particles_->find(pdg::positron()));

    const auto& lo = xs_.physics_vectors[range.material.get()];
    const auto& hi = xs_.physics_vectors[range.material.get()];
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
std::string EIonisationProcess::label() const
{
    return "Electron/positron ionisation";
}

//---------------------------------------------------------------------------//
} // namespace celeritas
