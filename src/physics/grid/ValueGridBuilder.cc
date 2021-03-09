//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ValueGridBuilder.cc
//---------------------------------------------------------------------------//
#include "ValueGridBuilder.hh"

#include <algorithm>
#include <cmath>
#include "base/SoftEqual.hh"
#include "physics/grid/UniformGrid.hh"
#include "physics/grid/XsGridInterface.hh"
#include "physics/grid/ValueGridInserter.hh"

namespace celeritas
{
namespace
{
using SpanConstReal = ValueGridXsBuilder::SpanConstReal;
//---------------------------------------------------------------------------//
//// HELPER FUNCTIONS ////
//---------------------------------------------------------------------------//
bool is_contiguous_increasing(SpanConstReal first, SpanConstReal second)
{
    return first.size() >= 2 && second.size() >= 2 && first.front() > 0
           && first.back() > first.front()
           && soft_equal(second.front(), first.back())
           && second.back() > second.front();
}

bool has_same_log_spacing(SpanConstReal first, SpanConstReal second)
{
    auto calc_log_delta = [](SpanConstReal vec) {
        return vec.back() / (vec.front() * (vec.size() - 1));
    };
    real_type delta[] = {calc_log_delta(first), calc_log_delta(second)};
    return soft_equal(delta[0], delta[1]);
}

bool is_nonnegative(SpanConstReal vec)
{
    return std::all_of(
        vec.begin(), vec.end(), [](real_type v) { return v >= 0; });
}

bool is_on_grid_point(real_type value, real_type lo, real_type hi, size_type size)
{
    if (value < lo || value > hi)
        return false;

    real_type delta = (hi - lo) / size;
    return soft_zero(std::fmod(value - lo, delta));
}

bool is_monotonic_increasing(SpanConstReal grid)
{
    CELER_EXPECT(!grid.empty());
    auto iter = grid.begin();
    auto prev = *iter++;
    while (iter != grid.end())
    {
        if (*iter <= prev)
            return false;
        prev = *iter++;
    }
    return true;
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
// BASE CLASS
//---------------------------------------------------------------------------//
//! Default destructor
ValueGridBuilder::~ValueGridBuilder() = default;

//---------------------------------------------------------------------------//
// XS BUILDER
//---------------------------------------------------------------------------//
/*!
 * Construct XS arrays from imported data from Geant4.
 */
std::unique_ptr<ValueGridXsBuilder>
ValueGridXsBuilder::from_geant(SpanConstReal lambda_energy,
                               SpanConstReal lambda,
                               SpanConstReal lambda_prim_energy,
                               SpanConstReal lambda_prim)
{
    CELER_EXPECT(is_contiguous_increasing(lambda_energy, lambda_prim_energy));
    CELER_EXPECT(has_same_log_spacing(lambda_energy, lambda_prim_energy));
    CELER_EXPECT(lambda.size() == lambda_energy.size());
    CELER_EXPECT(lambda_prim.size() == lambda_prim_energy.size());
    CELER_EXPECT(soft_equal(lambda.back(),
                            lambda_prim.front() / lambda_prim_energy.front()));
    CELER_EXPECT(is_nonnegative(lambda) && is_nonnegative(lambda_prim));

    // Concatenate the two XS vectors: insert the scaled (lambda_prim) value at
    // the coincident point.
    VecReal xs(lambda.size() + lambda_prim.size() - 1);
    auto    dst = std::copy(lambda.begin(), lambda.end() - 1, xs.begin());
    dst         = std::copy(lambda_prim.begin(), lambda_prim.end(), dst);
    CELER_ASSERT(dst == xs.end());

    // Construct the grid
    return std::make_unique<ValueGridXsBuilder>(lambda_energy.front(),
                                                lambda_prim_energy.front(),
                                                lambda_prim_energy.back(),
                                                VecReal(std::move(xs)));
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
ValueGridXsBuilder::ValueGridXsBuilder(real_type emin,
                                       real_type eprime,
                                       real_type emax,
                                       VecReal   xs)
    : log_emin_(std::log(emin))
    , log_eprime_(std::log(eprime))
    , log_emax_(std::log(emax))
    , xs_(std::move(xs))
{
    CELER_EXPECT(emin > 0);
    CELER_EXPECT(eprime > emin);
    CELER_EXPECT(emax > eprime);
    CELER_EXPECT(xs_.size() >= 2);
    CELER_EXPECT(
        is_on_grid_point(log_eprime_, log_emin_, log_emax_, xs_.size() - 1));
    CELER_EXPECT(is_nonnegative(make_span(xs)));
}

//---------------------------------------------------------------------------//
/*!
 * Construct on device.
 */
auto ValueGridXsBuilder::build(ValueGridInserter insert) const -> ValueGridId
{
    auto log_energy
        = UniformGridData::from_bounds(log_emin_, log_emax_, xs_.size());

    // Find and check prime energy index
    UniformGrid grid{log_energy};
    auto        prime_index = grid.find(log_eprime_);
    CELER_ASSERT(soft_equal(grid[prime_index], log_eprime_));

    return insert(
        UniformGridData::from_bounds(log_emin_, log_emax_, xs_.size()),
        prime_index,
        make_span(xs_));
}

//---------------------------------------------------------------------------//
// LOG BUILDER
//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
ValueGridLogBuilder::ValueGridLogBuilder(real_type emin,
                                         real_type emax,
                                         VecReal   value)
    : log_emin_(std::log(emin))
    , log_emax_(std::log(emax))
    , value_(std::move(value))
{
    CELER_EXPECT(emin > 0);
    CELER_EXPECT(emax > emin);
    CELER_EXPECT(value_.size() >= 2);
}

//---------------------------------------------------------------------------//
/*!
 * Construct on device.
 */
auto ValueGridLogBuilder::build(ValueGridInserter insert) const -> ValueGridId
{
    return insert(
        UniformGridData::from_bounds(log_emin_, log_emax_, value_.size()),
        this->value());
}

//---------------------------------------------------------------------------//
// RANGE BUILDER
//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
ValueGridRangeBuilder::ValueGridRangeBuilder(real_type emin,
                                             real_type emax,
                                             VecReal   xs)
    : Base(emin, emax, std::move(xs))
{
    CELER_EXPECT(is_monotonic_increasing(this->value()));
}

//---------------------------------------------------------------------------//
// GENERIC BUILDER
//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
ValueGridGenericBuilder::ValueGridGenericBuilder(VecReal grid,
                                                 VecReal value,
                                                 Interp  grid_interp,
                                                 Interp  value_interp)
    : grid_(std::move(grid))
    , value_(std::move(value))
    , grid_interp_(grid_interp)
    , value_interp_(value_interp)
{
    CELER_EXPECT(grid_.size() >= 2
                 && is_monotonic_increasing(make_span(grid_)));
    CELER_EXPECT(value_.size() == grid_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data with linear interpolation.
 */
ValueGridGenericBuilder::ValueGridGenericBuilder(VecReal grid, VecReal value)
    : ValueGridGenericBuilder(
        std::move(grid), std::move(value), Interp::linear, Interp::linear)
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct grid data in the given mutable insert.
 */
auto ValueGridGenericBuilder::build(ValueGridInserter insert) const
    -> ValueGridId
{
    insert({make_span(grid_), grid_interp_},
           {make_span(value_), value_interp_});
    CELER_NOT_IMPLEMENTED("generic grids");
    return {};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
