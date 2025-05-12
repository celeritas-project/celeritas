//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/MockProcess.cc
//---------------------------------------------------------------------------//
#include "MockProcess.hh"

#include <algorithm>

#include "corecel/sys/ActionRegistry.hh"

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
auto MockProcess::macro_xs(Applicability applic) const -> XsGrid
{
    CELER_EXPECT(applic.material);
    CELER_EXPECT(applic.particle);

    XsGrid grid;
    if (!data_.xs.empty())
    {
        MaterialView mat(data_.materials->host_ref(), applic.material);
        real_type numdens = mat.number_density();

        grid.lower.x
            = {std::log(applic.lower.value()), std::log(applic.upper.value())};
        for (auto xs : data_.xs)
        {
            grid.lower.y.push_back(native_value_from(xs) * numdens);
        }
    }
    return grid;
}

//---------------------------------------------------------------------------//
auto MockProcess::energy_loss(Applicability applic) const -> EnergyLossGrid
{
    CELER_EXPECT(applic.material);
    CELER_EXPECT(applic.particle);

    using VecDbl = std::vector<double>;

    EnergyLossGrid grid;
    if (data_.energy_loss > zero_quantity())
    {
        MaterialView mat(data_.materials->host_ref(), applic.material);
        auto eloss_rate = native_value_to<units::MevEnergy>(
            native_value_from(data_.energy_loss) * mat.number_density());

        grid.x
            = {std::log(applic.lower.value()), std::log(applic.upper.value())};
        grid.y = VecDbl(3, eloss_rate.value());
    }
    return grid;
}

//---------------------------------------------------------------------------//
bool MockProcess::supports_integral_xs() const
{
    return data_.supports_integral_xs;
}

//---------------------------------------------------------------------------//
bool MockProcess::applies_at_rest() const
{
    return data_.applies_at_rest;
}

//---------------------------------------------------------------------------//
std::string_view MockProcess::label() const
{
    return data_.label;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
