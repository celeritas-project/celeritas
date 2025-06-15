//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/MockModel.cc
//---------------------------------------------------------------------------//
#include "MockModel.hh"

#include <sstream>

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
    label_ = "mock-model-";
    label_ += std::to_string(data_.id.get() - 4);

    std::ostringstream os;
    os << "MockModel(" << (data_.id.get() - 4)
       << ", p=" << data_.applic.particle.get()
       << ", emin=" << data_.applic.lower.value()
       << ", emax=" << data_.applic.upper.value() << ")";
    description_ = std::move(os).str();
}

auto MockModel::applicability() const -> SetApplicability
{
    return {data_.applic};
}

auto MockModel::micro_xs(Applicability range) const -> XsTable
{
    CELER_EXPECT(range.material);
    CELER_EXPECT(range.particle);

    XsTable grids;
    MaterialView mat(data_.materials->host_ref(), range.material);
    if (!data_.xs.empty())
    {
        grids.resize(mat.num_elements());
        for (auto elcomp_idx : celeritas::range(mat.num_elements()))
        {
            grids[elcomp_idx].x = {std::log(range.lower.value()),
                                   std::log(range.upper.value())};
            for (auto xs : data_.xs)
            {
                grids[elcomp_idx].y.push_back(native_value_from(xs));
            }
        }
    }
    return grids;
}

void MockModel::step(CoreParams const&, CoreStateHost&) const
{
    // Shouldn't be called?
}

void MockModel::step(CoreParams const&, CoreStateDevice&) const
{
    // Inform calling test code that we've been executed
    data_.cb(this->action_id());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
