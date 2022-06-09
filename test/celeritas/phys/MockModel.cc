//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/MockModel.cc
//---------------------------------------------------------------------------//
#include "MockModel.hh"

#include <sstream>

#include "celeritas/grid/ValueGridBuilder.hh"
#include "celeritas/mat/MaterialView.hh"

namespace celeritas_test
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

    using VecReal = std::vector<real_type>;

    MicroXsBuilders builders;

    celeritas::MaterialView mat(data_.materials->host_ref(), range.material);
    real_type               numdens = mat.number_density();

    CELER_ASSERT(data_.applic.particle == range.particle);
    // if (data_.applic.particle == range.particle)
    {
        // TODO: add materials with multiple elements and calculate partial
        // macro cdf tables
        if (mat.num_elements() > 1 && !data_.xs.empty())
        {
            for (const auto& elcomp : mat.elements())
            {
                VecReal xs_grid;
                for (auto xs : data_.xs)
                {
                    xs_grid.push_back(native_value_from(xs) * numdens);
                }
                builders[elcomp.element]
                    = std::make_unique<celeritas::ValueGridLogBuilder>(
                        range.lower.value(), range.upper.value(), xs_grid);
            }
        }
    }
    return builders;
}

void MockModel::execute(CoreHostRef const&) const
{
    // Shouldn't be called?
}

void MockModel::execute(CoreDeviceRef const&) const
{
    // Inform calling test code that we've been launched
    data_.cb(this->action_id());
}

std::string MockModel::label() const
{
    return std::string("mock-model-") + std::to_string(data_.id.get());
}

std::string MockModel::description() const
{
    std::ostringstream os;
    os << "MockModel(" << data_.id.get()
       << ", p=" << data_.applic.particle.get()
       << ", emin=" << data_.applic.lower.value()
       << ", emax=" << data_.applic.upper.value() << ")";
    return os.str();
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
