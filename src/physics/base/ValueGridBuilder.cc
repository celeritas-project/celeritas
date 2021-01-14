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
#include "base/UniformGrid.hh"

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
    return first.size() >= 2 && second.size() >= 2 && second.size() >= 2
           && first.front() > 0 && first.back() > first.front()
           && second.front() == first.back() && second.back() > second.back();
}

bool has_same_log_spacing(SpanConstReal first, SpanConstReal second)
{
    auto calc_log_delta = [](SpanConstReal vec) {
        return vec.back() / (vec.front() * (vec.size() - 1));
    };
    return soft_equal(calc_log_delta(first), calc_log_delta(second));
}

bool is_nonnegative(SpanConstReal vec)
{
    return std::all_of(
        vec.begin(), vec.end(), [](real_type v) { return v > 0; });
}

bool is_on_grid_point(real_type value, real_type lo, real_type hi, size_type size)
{
    if (value < lo || value > hi)
        return false;

    real_type delta = (hi - lo) / size;
    return soft_zero(std::fmod(value - lo, delta));
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
//! Default destructor
ValueGridBuilder::~ValueGridBuilder() = default;

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
    REQUIRE(is_contiguous_increasing(lambda_energy, lambda_prim_energy));
    REQUIRE(has_same_log_spacing(lambda_prim, lambda_prim_energy));
    REQUIRE(lambda.size() == lambda_energy.size());
    REQUIRE(lambda_prim.size() == lambda_prim_energy.size());
    REQUIRE(soft_equal(lambda.back(),
                       lambda_prim.front() * lambda_prim_energy.front()));
    REQUIRE(is_nonnegative(lambda) && is_nonnegative(lambda_prim));

    // Concatenate the two XS vectors
    std::vector<real_type> xs(lambda.size() + lambda_prim.size() - 1);
    auto dst = std::copy(lambda.begin(), lambda.end(), xs.begin());
    dst      = std::copy(lambda_prim.begin(), lambda_prim.end(), dst);
    CHECK(dst == xs.end());

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
ValueGridXsBuilder::ValueGridXsBuilder(real_type              emin,
                                       real_type              eprime,
                                       real_type              emax,
                                       std::vector<real_type> xs)
    : log_emin_(std::log(emin))
    , log_eprime_(std::log(eprime))
    , log_emax_(std::log(emax))
    , xs_(std::move(xs))
{
    REQUIRE(emin > 0);
    REQUIRE(eprime > emin);
    REQUIRE(emax > eprime);
    REQUIRE(xs_.size() >= 2);
    REQUIRE(
        is_on_grid_point(log_eprime_, log_emin_, log_emax_, xs_.size() - 1));
}

//---------------------------------------------------------------------------//
/*!
 * Get the storage type and requirements for the energy grid.
 */
auto ValueGridXsBuilder::energy_storage() const -> EnergyStorage
{
    return {EnergyLookup::uniform_log, 0};
}

//---------------------------------------------------------------------------//
/*!
 * Get the storage type and requirements for the value grid.
 */
auto ValueGridXsBuilder::value_storage() const -> ValueStorage
{
    return {ValueCalculation::linear_scaled, xs_.size()};
}

//---------------------------------------------------------------------------//
/*!
 * Construct on device.
 */
void ValueGridXsBuilder::build(ValueGridStore*) const
{
    // TODO: finish implementation
    CHECK_UNREACHABLE;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
