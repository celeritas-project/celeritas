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
#include "corecel/cont/Span.hh"
#include "corecel/grid/UniformGrid.hh"
#include "corecel/grid/UniformGridData.hh"
#include "corecel/grid/VectorUtils.hh"
#include "corecel/math/SoftEqual.hh"

#include "XsGridInserter.hh"

namespace celeritas
{
namespace
{
using SpanConstDbl = Span<double const>;
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
ValueGridXsBuilder::from_geant(inp::Grid const& lower, inp::Grid const& upper)
{
    CELER_EXPECT(lower && upper);
    CELER_EXPECT(
        is_contiguous_increasing(make_span(lower.x), make_span(upper.x)));
    CELER_EXPECT(has_log_spacing(make_span(lower.x))
                 && has_log_spacing(make_span(upper.x)));
    CELER_EXPECT(soft_equal(lower.y.back(), upper.y.front() / upper.x.front()));
    CELER_EXPECT(is_nonnegative(make_span(lower.y))
                 && is_nonnegative(make_span(upper.y)));

    // Construct the grid
    return std::make_unique<ValueGridXsBuilder>(
        GridInput{lower.x.front(), upper.x.front(), lower.y},
        GridInput{upper.x.front(), upper.x.back(), upper.y});
}

//---------------------------------------------------------------------------//
/*!
 * Construct XS arrays from scaled (*E) data from Geant4.
 */
std::unique_ptr<ValueGridXsBuilder>
ValueGridXsBuilder::from_scaled(inp::Grid const& upper)
{
    CELER_EXPECT(upper.y.size() == upper.x.size());
    CELER_EXPECT(has_log_spacing(make_span(upper.x)));
    CELER_EXPECT(is_nonnegative(make_span(upper.y)));

    return std::make_unique<ValueGridXsBuilder>(
        GridInput{upper.x.front(), upper.x.front(), {}},
        GridInput{upper.x.front(), upper.x.back(), upper.y});
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
ValueGridXsBuilder::ValueGridXsBuilder(GridInput lower, GridInput upper)
    : lower_(std::move(lower)), upper_(std::move(upper))
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
    return insert(
        lower, make_span(lower_.value), upper, make_span(upper_.value));
}

//---------------------------------------------------------------------------//
// LOG BUILDER
//---------------------------------------------------------------------------//
/*!
 * Construct arrays from log-spaced geant data.
 */
auto ValueGridLogBuilder::from_geant(VecDbl const& x, VecDbl const& y)
    -> UPLogBuilder
{
    CELER_EXPECT(!x.empty());
    CELER_EXPECT(has_log_spacing(make_span(x)));
    CELER_EXPECT(y.size() == x.size());

    return std::make_unique<ValueGridLogBuilder>(
        GridInput{x.front(), x.back(), y});
}

//---------------------------------------------------------------------------//
/*!
 * Construct arrays from log-spaced geant data.
 */
auto ValueGridLogBuilder::from_geant(inp::Grid const& grid) -> UPLogBuilder
{
    CELER_EXPECT(!grid.x.empty());
    CELER_EXPECT(has_log_spacing(make_span(grid.x)));
    CELER_EXPECT(grid.y.size() == grid.x.size());

    return std::make_unique<ValueGridLogBuilder>(
        GridInput{grid.x.front(), grid.x.back(), grid.y});
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
    auto const& value = grid_.value;
    return insert(UniformGridData::from_bounds(
                      std::log(grid_.emin), std::log(grid_.emax), value.size()),
                  make_span(value));
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
