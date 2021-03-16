//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MockProcess.cc
//---------------------------------------------------------------------------//
#include "MockProcess.hh"

#include "MockModel.hh"
#include "physics/material/MaterialView.hh"

using namespace celeritas;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
MockProcess::MockProcess(Input data) : data_(std::move(data))
{
    CELER_EXPECT(data_.materials);
    CELER_EXPECT(!data_.label.empty());
    CELER_EXPECT(!data_.applic.empty());
    CELER_EXPECT(data_.interact);
    CELER_EXPECT(data_.xs >= celeritas::zero_quantity());
    CELER_EXPECT(data_.energy_loss >= 0);
}

//---------------------------------------------------------------------------//
auto MockProcess::build_models(ModelIdGenerator next_id) const -> VecModel
{
    VecModel result;
    for (const Applicability& applic : data_.applic)
    {
        result.push_back(
            std::make_shared<MockModel>(next_id(), applic, data_.interact));
    }
    return result;
}

//---------------------------------------------------------------------------//
auto MockProcess::step_limits(Applicability range) const -> StepLimitBuilders
{
    CELER_EXPECT(range.material);
    CELER_EXPECT(range.particle);

    using VecReal = std::vector<real_type>;

    MaterialView mat(data_.materials->host_pointers(), range.material);
    real_type    numdens = mat.number_density();

    StepLimitBuilders builders;
    if (data_.xs > zero_quantity())
    {
        real_type xs = unit_cast(data_.xs) * numdens;
        builders[ValueGridType::macro_xs]
            = std::make_unique<ValueGridLogBuilder>(
                range.lower.value(), range.upper.value(), VecReal{xs, xs});
    }
    if (data_.energy_loss > 0)
    {
        real_type eloss_rate = data_.energy_loss * numdens;
        builders[ValueGridType::energy_loss]
            = std::make_unique<ValueGridLogBuilder>(
                range.lower.value(),
                range.upper.value(),
                VecReal{eloss_rate, eloss_rate});

        builders[ValueGridType::range]
            = std::make_unique<ValueGridRangeBuilder>(
                range.lower.value(),
                range.upper.value(),
                VecReal{range.lower.value() / eloss_rate,
                        range.upper.value() / eloss_rate});
    }

    return builders;
}

//---------------------------------------------------------------------------//
std::string MockProcess::label() const
{
    return data_.label;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
