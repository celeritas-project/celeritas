//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "corecel/grid/VectorUtils.hh"
#include "corecel/math/SoftEqual.hh"

#include "XsGridInserter.hh"

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

bool is_nonnegative(SpanConstDbl vec)
{
    return std::all_of(vec.begin(), vec.end(), [](double v) { return v >= 0; });
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
                               inp::Interpolation interp,
                               SpanConstDbl lambda_prim_energy,
                               SpanConstDbl lambda_prim,
                               inp::Interpolation interp_prim)
{
    CELER_EXPECT(is_contiguous_increasing(lambda_energy, lambda_prim_energy));
    CELER_EXPECT(has_log_spacing(lambda_energy)
                 && has_log_spacing(lambda_prim_energy));
    CELER_EXPECT(lambda.size() == lambda_energy.size());
    CELER_EXPECT(lambda_prim.size() == lambda_prim_energy.size());
    CELER_EXPECT(soft_equal(lambda.back(),
                            lambda_prim.front() / lambda_prim_energy.front()));
    CELER_EXPECT(is_nonnegative(lambda) && is_nonnegative(lambda_prim));

    // Construct the grid
    return std::make_unique<ValueGridXsBuilder>(
        GridInput{lambda_energy.front(),
                  lambda_prim_energy.front(),
                  VecDbl{lambda.begin(), lambda.end()},
                  interp},
        GridInput{lambda_prim_energy.front(),
                  lambda_prim_energy.back(),
                  VecDbl{lambda_prim.begin(), lambda_prim.end()},
                  interp_prim});
}

//---------------------------------------------------------------------------//
/*!
 * Construct XS arrays from scaled (*E) data from Geant4.
 */
std::unique_ptr<ValueGridXsBuilder>
ValueGridXsBuilder::from_scaled(SpanConstDbl lambda_prim_energy,
                                SpanConstDbl lambda_prim,
                                inp::Interpolation interp)
{
    CELER_EXPECT(lambda_prim.size() == lambda_prim_energy.size());
    CELER_EXPECT(has_log_spacing(lambda_prim_energy));
    CELER_EXPECT(is_nonnegative(lambda_prim));

    return std::make_unique<ValueGridXsBuilder>(
        GridInput{
            lambda_prim_energy.front(), lambda_prim_energy.front(), {}, interp},
        GridInput{lambda_prim_energy.front(),
                  lambda_prim_energy.back(),
                  VecDbl{lambda_prim.begin(), lambda_prim.end()},
                  interp});
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
ValueGridXsBuilder::ValueGridXsBuilder(GridInput grid, GridInput grid_prime)
    : lower_(std::move(grid)), upper_(std::move(grid_prime))
{
    CELER_EXPECT((!lower_.value.empty() || !upper_.value.empty())
                 && (lower_.value.empty() || upper_.value.empty()
                     || lower_.emax == upper_.emin));
    CELER_EXPECT(lower_.value.empty()
                 || (lower_.emin > 0 && lower_.emax > lower_.emin
                     && lower_.value.size() >= 2));
    CELER_EXPECT(upper_.value.empty()
                 || (upper_.emin > 0 && upper_.emax > upper_.emin
                     && upper_.value.size() >= 2));
    CELER_EXPECT(is_nonnegative(make_span(lower_.value)));
    CELER_EXPECT(is_nonnegative(make_span(upper_.value)));
}

//---------------------------------------------------------------------------//
/*!
 * Construct on device.
 */
auto ValueGridXsBuilder::build(XsGridInserter insert) const -> ValueGridId
{
    auto lower = !lower_.value.empty()
                     ? UniformGridData::from_bounds(std::log(lower_.emin),
                                                    std::log(lower_.emax),
                                                    lower_.value.size())
                     : UniformGridData{};
    auto upper = !upper_.value.empty()
                     ? UniformGridData::from_bounds(std::log(upper_.emin),
                                                    std::log(upper_.emax),
                                                    upper_.value.size())
                     : UniformGridData{};
    return insert(lower,
                  make_span(lower_.value),
                  lower_.interp,
                  upper,
                  make_span(upper_.value),
                  upper_.interp);
}

//---------------------------------------------------------------------------//
// LOG BUILDER
//---------------------------------------------------------------------------//
/*!
 * Construct arrays from log-spaced geant data.
 */
auto ValueGridLogBuilder::from_geant(SpanConstDbl energy,
                                     SpanConstDbl value,
                                     inp::Interpolation interp) -> UPLogBuilder
{
    CELER_EXPECT(!energy.empty());
    CELER_EXPECT(has_log_spacing(energy));
    CELER_EXPECT(value.size() == energy.size());

    return std::make_unique<ValueGridLogBuilder>(
        GridInput{energy.front(),
                  energy.back(),
                  VecDbl{value.begin(), value.end()},
                  interp});
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
ValueGridLogBuilder::ValueGridLogBuilder(GridInput grid)
    : grid_(std::move(grid))
{
    CELER_EXPECT(grid_.emin > 0);
    CELER_EXPECT(grid_.emax > grid_.emin);
    CELER_EXPECT(grid_.value.size() >= 2);
}

//---------------------------------------------------------------------------//
/*!
 * Construct on device.
 */
auto ValueGridLogBuilder::build(XsGridInserter insert) const -> ValueGridId
{
    return insert(UniformGridData::from_bounds(std::log(grid_.emin),
                                               std::log(grid_.emax),
                                               grid_.value.size()),
                  this->value(),
                  grid_.interp);
}

//---------------------------------------------------------------------------//
/*!
 * Access values.
 */
auto ValueGridLogBuilder::value() const -> SpanConstDbl
{
    return make_span(grid_.value);
}

//---------------------------------------------------------------------------//
// ON-THE-FLY
//---------------------------------------------------------------------------//
/*!
 * Always return an 'invalid' ID.
 */
auto ValueGridOTFBuilder::build(XsGridInserter) const -> ValueGridId
{
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
