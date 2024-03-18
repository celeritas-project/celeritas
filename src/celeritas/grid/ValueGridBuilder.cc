//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/ValueGridBuilder.cc
//---------------------------------------------------------------------------//
#include "ValueGridBuilder.hh"

#include <algorithm>
#include <cmath>
#include <utility>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/grid/UniformGrid.hh"
#include "corecel/grid/UniformGridData.hh"
#include "corecel/math/SoftEqual.hh"

#include "ValueGridInserter.hh"
#include "VectorUtils.hh"

namespace celeritas
{
namespace
{
using SpanConstDbl = ValueGridXsBuilder::SpanConstDbl;
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
bool is_contiguous_increasing(SpanConstDbl first, SpanConstDbl second)
{
    return first.size() >= 2 && second.size() >= 2 && first.front() > 0
           && first.back() > first.front()
           && soft_equal(second.front(), first.back())
           && second.back() > second.front();
}

double calc_log_delta(SpanConstDbl vec)
{
    return std::pow(vec.back() / vec.front(), double(1) / (vec.size() - 1));
}

bool has_log_spacing(SpanConstDbl vec)
{
    double delta = calc_log_delta(vec);
    for (auto i : range(vec.size() - 1))
    {
        if (!soft_equal(delta, vec[i + 1] / vec[i]))
            return false;
    }
    return true;
}

bool is_nonnegative(SpanConstDbl vec)
{
    return std::all_of(vec.begin(), vec.end(), [](double v) { return v >= 0; });
}

bool is_on_grid_point(double value, double lo, double hi, size_type size)
{
    if (value < lo || value > hi)
        return false;

    double delta = (hi - lo) / size;
    return soft_mod(value - lo, delta);
}

//---------------------------------------------------------------------------//
}  // namespace

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
ValueGridXsBuilder::from_geant(SpanConstDbl lambda_energy,
                               SpanConstDbl lambda,
                               SpanConstDbl lambda_prim_energy,
                               SpanConstDbl lambda_prim)
{
    CELER_EXPECT(is_contiguous_increasing(lambda_energy, lambda_prim_energy));
    CELER_EXPECT(has_log_spacing(lambda_energy)
                 && has_log_spacing(lambda_prim_energy));
    CELER_EXPECT(lambda.size() == lambda_energy.size());
    CELER_EXPECT(lambda_prim.size() == lambda_prim_energy.size());
    CELER_EXPECT(soft_equal(lambda.back(),
                            lambda_prim.front() / lambda_prim_energy.front()));
    CELER_EXPECT(is_nonnegative(lambda) && is_nonnegative(lambda_prim));

    double const log_delta_lo = calc_log_delta(lambda_energy);
    double const log_delta_hi = calc_log_delta(lambda_prim_energy);
    CELER_VALIDATE(
        soft_equal(log_delta_lo, log_delta_hi),
        << "Lower and upper energy grids have inconsistent spacing: "
           "log delta E for lower grid is "
        << log_delta_lo << " log(MeV) per bin but upper is " << log_delta_hi);

    // Concatenate the two XS vectors: insert the scaled (lambda_prim) value at
    // the coincident point.
    VecDbl xs(lambda.size() + lambda_prim.size() - 1);
    auto dst = std::copy(lambda.begin(), lambda.end() - 1, xs.begin());
    dst = std::copy(lambda_prim.begin(), lambda_prim.end(), dst);
    CELER_ASSERT(dst == xs.end());

    // Construct the grid
    return std::make_unique<ValueGridXsBuilder>(lambda_energy.front(),
                                                lambda_prim_energy.front(),
                                                lambda_prim_energy.back(),
                                                VecDbl(std::move(xs)));
}

//---------------------------------------------------------------------------//
/*!
 * Construct XS arrays from scaled (*E) data from Geant4.
 */
std::unique_ptr<ValueGridXsBuilder>
ValueGridXsBuilder::from_scaled(SpanConstDbl lambda_prim_energy,
                                SpanConstDbl lambda_prim)
{
    CELER_EXPECT(lambda_prim.size() == lambda_prim_energy.size());
    CELER_EXPECT(has_log_spacing(lambda_prim_energy));
    CELER_EXPECT(is_nonnegative(lambda_prim));

    return std::make_unique<ValueGridXsBuilder>(
        lambda_prim_energy.front(),
        lambda_prim_energy.front(),
        lambda_prim_energy.back(),
        VecDbl{lambda_prim.begin(), lambda_prim.end()});
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
ValueGridXsBuilder::ValueGridXsBuilder(double emin,
                                       double eprime,
                                       double emax,
                                       VecDbl xs)
    : log_emin_(std::log(emin))
    , log_eprime_(std::log(eprime))
    , log_emax_(std::log(emax))
    , xs_(std::move(xs))
{
    CELER_EXPECT(emin > 0);
    CELER_EXPECT(eprime >= emin);
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

    // Find and check prime energy index. Due to floating-point roundoff,
    // \c log_eprime might not be *exactly* on a grid point, and it's possible
    // the index below that of the correct grid point will be returned instead.
    // Check and correct for this.
    UniformGrid grid{log_energy};
    auto prime_index = grid.find(log_eprime_);
    if (soft_equal<real_type>(grid[prime_index + 1], log_eprime_))
        ++prime_index;
    CELER_ASSERT(prime_index + 1 < xs_.size());
    CELER_ASSERT(soft_equal<real_type>(grid[prime_index], log_eprime_));

    return insert(
        UniformGridData::from_bounds(log_emin_, log_emax_, xs_.size()),
        prime_index,
        make_span(xs_));
}

//---------------------------------------------------------------------------//
// LOG BUILDER
//---------------------------------------------------------------------------//
/*!
 * Construct arrays from log-spaced geant data.
 */
auto ValueGridLogBuilder::from_geant(SpanConstDbl energy, SpanConstDbl value)
    -> UPLogBuilder
{
    CELER_EXPECT(!energy.empty());
    CELER_EXPECT(has_log_spacing(energy));
    CELER_EXPECT(value.size() == energy.size());

    return std::make_unique<ValueGridLogBuilder>(
        energy.front(), energy.back(), VecDbl{value.begin(), value.end()});
}

//---------------------------------------------------------------------------//
/*!
 * Construct XS arrays from log-spaced geant range data.
 *
 * Range data must be monotonically increasing, since it's the integral of the
 * (always nonnegative) stopping power -dE/dx . If not monotonic then the
 * inverse range cannot be calculated.
 */
auto ValueGridLogBuilder::from_range(SpanConstDbl energy, SpanConstDbl value)
    -> UPLogBuilder
{
    CELER_EXPECT(!energy.empty());
    CELER_EXPECT(is_monotonic_increasing(value));
    CELER_EXPECT(value.front() > 0);
    return ValueGridLogBuilder::from_geant(energy, value);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
ValueGridLogBuilder::ValueGridLogBuilder(double emin, double emax, VecDbl value)
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
/*!
 * Access values.
 */
auto ValueGridLogBuilder::value() const -> SpanConstDbl
{
    return make_span(value_);
}

//---------------------------------------------------------------------------//
// GENERIC BUILDER
//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
ValueGridGenericBuilder::ValueGridGenericBuilder(VecDbl grid,
                                                 VecDbl value,
                                                 Interp grid_interp,
                                                 Interp value_interp)
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
ValueGridGenericBuilder::ValueGridGenericBuilder(VecDbl grid, VecDbl value)
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
// ON-THE-FLY
//---------------------------------------------------------------------------//
/*!
 * Always return an 'invalid' ID.
 */
auto ValueGridOTFBuilder::build(ValueGridInserter) const -> ValueGridId
{
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
