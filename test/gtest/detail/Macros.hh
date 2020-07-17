//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Macros.hh
//---------------------------------------------------------------------------//
#pragma once

//! Soft equivalence macro
#define EXPECT_SOFT_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2(::celeritas::detail::IsSoftEquiv, expected, actual)

//! Soft equivalence macro with relative error
#define EXPECT_SOFT_NEAR(expected, actual, rel_error) \
    EXPECT_PRED_FORMAT3(                              \
        ::celeritas::detail::IsSoftEquiv, expected, actual, rel_error)

#include "Macros.i.hh"

//---------------------------------------------------------------------------//
