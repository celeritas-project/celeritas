//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ValueGridBuilder.cc
//---------------------------------------------------------------------------//
#include "ValueGridBuilder.hh"

#include <cmath>
#include "base/SoftEqual.hh"
#include "physics/grid/UniformGrid.hh"
#include "physics/grid/XsGridInterface.hh"
#include "physics/grid/ValueGridStore.hh"

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
           && first.back() > first.front() && second.front() == first.back()
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
ValueGridXsBuilder
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

    // Concatenate the two XS vectors: store the scaled (lambda_prim) value at
    // the coincident point.
    VecReal xs(lambda.size() + lambda_prim.size() - 1);
    auto    dst = std::copy(lambda.begin(), lambda.end() - 1, xs.begin());
    dst         = std::copy(lambda_prim.begin(), lambda_prim.end(), dst);
    CELER_ASSERT(dst == xs.end());

    // Construct the grid
    return {lambda_energy.front(),
            lambda_prim_energy.front(),
            lambda_prim_energy.back(),
            std::move(xs)};
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
 * Get the storage type and requirements for the value grid.
 */
auto ValueGridXsBuilder::storage() const -> Storage
{
    return {ValueGridType::xs, xs_.size()};
}

//---------------------------------------------------------------------------//
/*!
 * Construct on device.
 */
void ValueGridXsBuilder::build(ValueGridStore* store) const
{
    XsGridPointers ptrs;
    // Construct log(energy) grid
    ptrs.log_energy
        = UniformGridPointers::from_bounds(log_emin_, log_emax_, xs_.size());
    {
        // Find and check prime energy index
        UniformGrid grid{ptrs.log_energy};
        ptrs.prime_index = grid.find(log_eprime_);
        CELER_ASSERT(soft_equal(grid[ptrs.prime_index], log_eprime_));
    }

    // Provide reference to values
    ptrs.value = make_span(xs_);

    // Copy data to store
    store->push_back(ptrs);
}

//---------------------------------------------------------------------------//
// LOG BUILDER
//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
ValueGridLogBuilder::ValueGridLogBuilder(real_type emin,
                                         real_type emax,
                                         VecReal   xs)
    : log_emin_(std::log(emin)), log_emax_(std::log(emax)), xs_(std::move(xs))
{
    CELER_EXPECT(emin > 0);
    CELER_EXPECT(emax > emin);
    CELER_EXPECT(xs_.size() >= 2);
}

//---------------------------------------------------------------------------//
/*!
 * Get the storage type and requirements for the energy grid.
 */
auto ValueGridLogBuilder::storage() const -> Storage
{
    return {ValueGridType::xs, xs_.size()};
}

//---------------------------------------------------------------------------//
/*!
 * Construct on device.
 */
void ValueGridLogBuilder::build(ValueGridStore* store) const
{
    XsGridPointers ptrs;

    // Construct log(energy) grid
    ptrs.log_energy
        = UniformGridPointers::from_bounds(log_emin_, log_emax_, xs_.size());

    // Provide reference to values
    ptrs.value = make_span(xs_);

    // Copy data to store
    store->push_back(ptrs);
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
 * Get the storage type and requirements for the energy grid.
 */
auto ValueGridGenericBuilder::storage() const -> Storage
{
    return {ValueGridType::generic, 2 * grid_.size()};
}

//---------------------------------------------------------------------------//
/*!
 * Construct on device.
 */
void ValueGridGenericBuilder::build(ValueGridStore* store) const
{
    GenericGridPointers ptrs;

    // Provide reference to values
    ptrs.grid         = make_span(grid_);
    ptrs.value        = make_span(value_);
    ptrs.grid_interp  = grid_interp_;
    ptrs.value_interp = value_interp_;

    // Copy data to store
    store->push_back(ptrs);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
