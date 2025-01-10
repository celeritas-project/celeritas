//------------------------------- -*- C++ -*- -------------------------------//
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//---------------------------------------------------------------------------//
/*!
 * \file ArrayUtilsImpl.hh
 */
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"

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
    static CELER_CONSTEXPR_FUNCTION float min_accurate_sintheta()
    {
        return 0.07f;
    }
};

template<>
struct RealVecTraits<double>
{
    //! Threshold for rotation
    static CELER_CONSTEXPR_FUNCTION double min_accurate_sintheta()
    {
        return 0.005;
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
