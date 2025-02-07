//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/CalculatorTestBase.cc
//---------------------------------------------------------------------------//
#include "CalculatorTestBase.hh"

#include <cmath>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/grid/UniformGrid.hh"
#include "corecel/math/SoftEqual.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
void CalculatorTestBase::build(real_type emin, real_type emax, size_type count)
{
    this->build({emin, emax}, count, [](real_type energy) { return energy; });
    CELER_ENSURE(soft_equal(emax, value_ref_[data_.value].back()));
}

//---------------------------------------------------------------------------//
/*!
 * Set the index above which data is scaled by 1/E.
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
/*!
 * Construct from an arbitrary function.
 */
void CalculatorTestBase::build_impl(Real2 bounds,
                                    size_type count,
                                    XsFunc calc_xs,
                                    BC bc)
{
    CELER_EXPECT(bounds[1] > bounds[0]);
    CELER_EXPECT(count >= 2);
    CELER_EXPECT(calc_xs);

    data_.log_energy = UniformGridData::from_bounds(
        std::log(bounds[0]), std::log(bounds[1]), count);

    UniformGrid loge{data_.log_energy};
    CELER_ASSERT(loge.size() == count);

    std::vector<real_type> temp_xs(loge.size());
    temp_xs.front() = calc_xs(bounds[0]);
    for (auto i : range<std::size_t>(1, loge.size() - 1))
    {
        temp_xs[i] = calc_xs(std::exp(loge[i]));
    }
    temp_xs.back() = calc_xs(bounds[1]);

    value_storage_ = {};
    CollectionBuilder build(&value_storage_);
    data_.value = build.insert_back(temp_xs.begin(), temp_xs.end());
    if (bc != BC::size_)
    {
        Data value_ref{value_storage_};
        auto deriv = SplineDerivCalculator(bc)(data_, value_ref);
        data_.derivative = build.insert_back(deriv.begin(), deriv.end());
    }
    value_ref_ = value_storage_;

    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Scale cross sections at or above this index by a factor of E.
 */
void CalculatorTestBase::convert_to_prime(size_type prime_index)
{
    CELER_EXPECT(data_);
    CELER_EXPECT(prime_index < data_.log_energy.size);
    CELER_EXPECT(data_.prime_index == XsGridData::no_scaling());

    UniformGrid loge{data_.log_energy};
    SpanReal values = value_storage_[data_.value];

    for (auto i : range(prime_index, data_.log_energy.size))
    {
        values[i] *= std::exp(loge[i]);
    }
    data_.prime_index = prime_index;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
