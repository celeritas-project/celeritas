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

#include "XsGridInserter.hh"

namespace celeritas
{
namespace
{
using SpanConstDbl = Span<double const>;

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
 * Construct from raw data.
 */
ValueGridXsBuilder::ValueGridXsBuilder(inp::UniformGrid lower,
                                       inp::UniformGrid upper)
    : lower_(std::move(lower)), upper_(std::move(upper))
{
    CELER_EXPECT(
        (lower_ || upper_)
        && (!lower_ || !upper_ || lower_.x[Bound::hi] == upper_.x[Bound::lo]));
    CELER_EXPECT(
        !lower_ || (std::exp(lower_.x[Bound::lo]) > 0 && lower_.y.size() >= 2));
    CELER_EXPECT(
        !upper_ || (std::exp(upper_.x[Bound::lo]) > 0 && upper_.y.size() >= 2));
    CELER_EXPECT(is_nonnegative(make_span(lower_.y)));
    CELER_EXPECT(is_nonnegative(make_span(upper_.y)));
}

//---------------------------------------------------------------------------//
/*!
 * Construct on device.
 */
auto ValueGridXsBuilder::build(XsGridInserter insert) const -> ValueGridId
{
    return insert(lower_, upper_);
}

//---------------------------------------------------------------------------//
// LOG BUILDER
//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
ValueGridLogBuilder::ValueGridLogBuilder(inp::UniformGrid grid)
    : grid_(std::move(grid))
{
    CELER_EXPECT(grid_);
    CELER_EXPECT(std::exp(grid_.x[Bound::lo]) > 0 && grid_.y.size() >= 2);
}

//---------------------------------------------------------------------------//
/*!
 * Construct on device.
 */
auto ValueGridLogBuilder::build(XsGridInserter insert) const -> ValueGridId
{
    return insert(grid_);
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
