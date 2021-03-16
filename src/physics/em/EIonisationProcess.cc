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
EIonisationProcess::EIonisationProcess(const Input& input)
    : particles_(std::move(input.particles))
    , lambda_table_(std::move(input.lambda))
    , dedx_table_(std::move(input.dedx))
    , range_table_(std::move(input.range))
{
    CELER_EXPECT(particles_);
    CELER_EXPECT(lambda_table_.table_type == ImportTableType::lambda);
    CELER_EXPECT(dedx_table_.table_type == ImportTableType::dedx);
    CELER_EXPECT(range_table_.table_type == ImportTableType::range);
    CELER_EXPECT(!lambda_table_.physics_vectors.empty());
    CELER_EXPECT(!dedx_table_.physics_vectors.empty());
    CELER_EXPECT(!range_table_.physics_vectors.empty());
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
 * Get cross section values.
 */
auto EIonisationProcess::step_limits(Applicability range) const
    -> StepLimitBuilders
{
    CELER_EXPECT(range.material);
    CELER_EXPECT(range.material < lambda_table_.physics_vectors.size());
    CELER_EXPECT(range.material < dedx_table_.physics_vectors.size());
    CELER_EXPECT(range.material < range_table_.physics_vectors.size());
    CELER_EXPECT(range.particle == particles_->find(pdg::electron())
                 || range.particle == particles_->find(pdg::positron()));

    const auto& xs_lambda = lambda_table_.physics_vectors[range.material.get()];
    const auto& xs_eloss  = dedx_table_.physics_vectors[range.material.get()];
    const auto& xs_range  = range_table_.physics_vectors[range.material.get()];
    CELER_ASSERT(xs_lambda.vector_type == ImportPhysicsVectorType::log);
    CELER_ASSERT(xs_eloss.vector_type == ImportPhysicsVectorType::log);
    CELER_ASSERT(xs_range.vector_type == ImportPhysicsVectorType::log);

    StepLimitBuilders builders;
    builders[size_type(ValueGridType::macro_xs)]
        = std::make_unique<ValueGridXsBuilder>(xs_lambda.x.front(),
                                               xs_lambda.x.front(),
                                               xs_lambda.x.back(),
                                               xs_lambda.y);
    builders[size_type(ValueGridType::energy_loss)]
        = std::make_unique<ValueGridXsBuilder>(xs_eloss.x.front(),
                                               xs_eloss.x.front(),
                                               xs_eloss.x.back(),
                                               xs_eloss.y);
    builders[size_type(ValueGridType::range)]
        = std::make_unique<ValueGridXsBuilder>(xs_range.x.front(),
                                               xs_range.x.front(),
                                               xs_range.x.back(),
                                               xs_range.y);

#if 0
    // From PR #184
    builders[ValueGridType::macro_xs] = ValueGridLogBuilder::from_geant(
        make_span(xs_lambda.x), make_span(xs_lambda.y));

    builders[ValueGridType::energy_loss] = ValueGridLogBuilder::from_geant(
        make_span(xs_eloss.x), make_span(xs_eloss.y));

    builders[ValueGridType::range] = ValueGridRangeBuilder::from_geant(
        make_span(xs_range.x), make_span(xs_range.y));
#endif

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
