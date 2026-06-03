//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/detail/ArrayUtilsImpl.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Traits for operations on real vectors
template<class T>
struct RealVecTraits;

template<>
struct RealVecTraits<float>
{
    //! Threshold for rotation
    static constexpr float min_accurate_sintheta = 0.07f;
    //! Threshold for Z axis coincidence: ~sqrt(eps_mach)
    static constexpr float min_coincident_sintheta = 4e-04f;
};

template<>
struct RealVecTraits<double>
{
    //! Threshold for rotation
    static constexpr double min_accurate_sintheta = 0.005;
    //! Threshold for Z axis coincidence: ~sqrt(eps_mach)
    static constexpr double min_coincident_sintheta = 2e-08;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
