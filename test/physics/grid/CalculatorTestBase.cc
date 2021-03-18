//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CalculatorTestBase.cc
//---------------------------------------------------------------------------//
#include "CalculatorTestBase.hh"

#include <cmath>
#include <vector>
#include "base/CollectionBuilder.hh"
#include "base/Range.hh"
#include "base/SoftEqual.hh"
#include "physics/grid/Interpolator.hh"

using namespace celeritas;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
void CalculatorTestBase::build(real_type emin, real_type emax, size_type count)
{
    CELER_EXPECT(count >= 2);
    data_.log_energy
        = UniformGridData::from_bounds(std::log(emin), std::log(emax), count);

    std::vector<real_type> temp_xs(count);

    // Interpolate xs grid: linear in bin, log in energy
    Interpolator<Interp::linear, Interp::log, real_type> calc_xs(
        {0.0, emin}, {count - 1.0, emax});
    for (auto i : range(temp_xs.size()))
    {
        temp_xs[i] = calc_xs(i);
    }

    value_storage_ = {};
    data_.value    = make_builder(&value_storage_)
                      .insert_back(temp_xs.begin(), temp_xs.end());
    value_ref_ = value_storage_;

    CELER_ENSURE(data_);
    CELER_ENSURE(soft_equal(emax, value_ref_[data_.value].back()));
}

//---------------------------------------------------------------------------//
/*!
 * Set the index above which data is 1/E
 */
void CalculatorTestBase::set_prime_index(size_type i)
{
    CELER_EXPECT(data_);
    CELER_EXPECT(i < data_.log_energy.size);
    data_.prime_index = i;
}

//---------------------------------------------------------------------------//
/*!
 * Get cross sections that can be modified.
 */
auto CalculatorTestBase::mutable_values() -> SpanReal
{
    CELER_EXPECT(data_);
    return value_storage_[data_.value];
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
