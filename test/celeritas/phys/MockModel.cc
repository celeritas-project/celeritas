//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/MockModel.cc
//---------------------------------------------------------------------------//
#include "MockModel.hh"

#include <sstream>

#include "celeritas/grid/ValueGridBuilder.hh"
#include "celeritas/mat/MaterialView.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
MockModel::MockModel(Input data) : data_(std::move(data))
{
    CELER_EXPECT(data_.id);
    CELER_EXPECT(data_.materials);
    CELER_EXPECT(data_.applic);
    CELER_EXPECT(data_.cb);
}

auto MockModel::applicability() const -> SetApplicability
{
    return {data_.applic};
}

auto MockModel::micro_xs(Applicability range) const -> MicroXsBuilders
{
    CELER_EXPECT(range.material);
    CELER_EXPECT(range.particle);

    MicroXsBuilders builders;
    MaterialView mat(data_.materials->host_ref(), range.material);
    if (!data_.xs.empty())
    {
        builders.resize(mat.num_elements());
        for (auto elcomp_idx : celeritas::range(mat.num_elements()))
        {
            std::vector<double> xs_grid;
            for (auto xs : data_.xs)
            {
                xs_grid.push_back(native_value_from(xs));
            }

            builders[elcomp_idx] = std::make_unique<ValueGridLogBuilder>(
                range.lower.value(), range.upper.value(), xs_grid);
        }
    }
    return builders;
}

void MockModel::execute(CoreParams const&, CoreStateHost&) const
{
    // Shouldn't be called?
}

void MockModel::execute(CoreParams const&, CoreStateDevice&) const
{
    // Inform calling test code that we've been executed
    data_.cb(this->action_id());
}

std::string MockModel::label() const
{
    return std::string("mock-model-") + std::to_string(data_.id.get() - 4);
}

std::string MockModel::description() const
{
    std::ostringstream os;
    os << "MockModel(" << (data_.id.get() - 4)
       << ", p=" << data_.applic.particle.get()
       << ", emin=" << data_.applic.lower.value()
       << ", emax=" << data_.applic.upper.value() << ")";
    return os.str();
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
