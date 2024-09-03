//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/MockProcess.cc
//---------------------------------------------------------------------------//
#include "MockProcess.hh"

#include <algorithm>

#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/grid/ValueGridBuilder.hh"

#include "MockModel.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
MockProcess::MockProcess(Input data) : data_(std::move(data))
{
    CELER_EXPECT(data_.materials);
    CELER_EXPECT(!data_.label.empty());
    CELER_EXPECT(!data_.applic.empty());
    CELER_EXPECT(data_.interact);
    CELER_EXPECT(
        data_.xs.empty()
        || std::any_of(data_.xs.begin(), data_.xs.end(), [](BarnMicroXs x) {
               return x > zero_quantity();
           }));
    CELER_EXPECT(data_.energy_loss >= zero_quantity());
}

//---------------------------------------------------------------------------//
auto MockProcess::build_models(ActionIdIter start_id) const -> VecModel
{
    MockModel::Input input;
    input.materials = data_.materials;
    input.cb = data_.interact;
    input.xs = data_.xs;

    VecModel result;
    for (Applicability const& applic : data_.applic)
    {
        input.id = *start_id++;
        input.applic = applic;
        result.push_back(std::make_shared<MockModel>(input));
    }
    return result;
}

//---------------------------------------------------------------------------//
auto MockProcess::step_limits(Applicability applic) const -> StepLimitBuilders
{
    CELER_EXPECT(applic.material);
    CELER_EXPECT(applic.particle);

    using VecDbl = std::vector<double>;

    MaterialView mat(data_.materials->host_ref(), applic.material);
    real_type numdens = mat.number_density();

    StepLimitBuilders builders;
    if (!data_.xs.empty())
    {
        VecDbl xs_grid;
        for (auto xs : data_.xs)
        {
            xs_grid.push_back(native_value_from(xs) * numdens);
        }
        builders[ValueGridType::macro_xs]
            = std::make_unique<ValueGridLogBuilder>(
                applic.lower.value(), applic.upper.value(), xs_grid);
    }
    if (data_.energy_loss > zero_quantity())
    {
        auto eloss_rate = native_value_to<units::MevEnergy>(
            native_value_from(data_.energy_loss) * numdens);
        builders[ValueGridType::energy_loss]
            = std::make_unique<ValueGridLogBuilder>(
                applic.lower.value(),
                applic.upper.value(),
                VecDbl{eloss_rate.value(), eloss_rate.value()});

        builders[ValueGridType::range] = std::make_unique<ValueGridLogBuilder>(
            applic.lower.value(),
            applic.upper.value(),
            VecDbl{applic.lower.value() / eloss_rate.value(),
                   applic.upper.value() / eloss_rate.value()});
    }

    return builders;
}

//---------------------------------------------------------------------------//
bool MockProcess::use_integral_xs() const
{
    return data_.use_integral_xs;
}

//---------------------------------------------------------------------------//
std::string_view MockProcess::label() const
{
    return data_.label;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
