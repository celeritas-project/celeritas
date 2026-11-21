//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/IntersectionSolver.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/FerrariSolver.hh"
#include "orange/OrangeTypes.hh"

#include "QuadraticSolver.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Input data for solving a planar intersection
struct PlanarIntersectionInput
{
    real_type displacement;
    real_type dot_pos;
    real_type dot_dir;
    SurfaceState on_surface;
};

//! Input data for solving a quadratic intersection
struct QuadraticIntersectionInput
{
    real_type a;
    real_type half_b;
    real_type c;
    SurfaceState on_surface;
};

//! Input data for solving a quartic intersection
struct QuarticIntersectionInput
{
    FerrariSolver::Real5 coeffs;
    SurfaceState on_surface;
};

//---------------------------------------------------------------------------//
//! Solve a planar intersection
inline CELER_FUNCTION auto solve_intersections(
    PlanarIntersectionInput const& input) -> Array<real_type, 1>
{
    real_type const dist = (input.displacement - input.dot_pos) / input.dot_dir;
    bool const valid = celeritas::logical_all(
        input.on_surface == SurfaceState::off, dist > 0);
    return {valid ? dist : no_intersection()};
}

//! Solve a quadratic intersection
inline CELER_FUNCTION auto solve_intersections(
    QuadraticIntersectionInput const& input) -> QuadraticSolver::Intersections
{
    return QuadraticSolver::solve_general(
        input.a, input.half_b, input.c, input.on_surface);
}

//! Solve a quartic intersection
inline CELER_FUNCTION auto solve_intersections(
    QuarticIntersectionInput const& input) -> FerrariSolver::result_type
{
    FerrariSolver solve{};
    if (input.on_surface == SurfaceState::on)
    {
        auto [a, b, c, d, e] = input.coeffs;
        return solve(FerrariSolver::Real4{a, b, c, d});
    }
    return solve(input.coeffs);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
