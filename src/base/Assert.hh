//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Assert.hh
//---------------------------------------------------------------------------//
#ifndef base_Assert_hh
#define base_Assert_hh

#include "celeritas_config.h"
#include "Macros.hh"
#include <cassert>

#define CELERITAS_ASSERT_(COND) \
    do                          \
    {                           \
        assert(COND);           \
    } while (0)
#define CELERITAS_NOASSERT_(COND) \
    do                            \
    {                             \
        if (false && (COND)) {}   \
    } while (0)

#ifdef CELERITAS_DEBUG
#    define REQUIRE(x) CELERITAS_ASSERT_(x)
#    define CHECK(x) CELERITAS_ASSERT_(x)
#    define ENSURE(x) CELERITAS_ASSERT_(x)
#else
#    define REQUIRE(x) CELERITAS_NOASSERT_(x)
#    define CHECK(x) CELERITAS_NOASSERT_(x)
#    define ENSURE(x) CELERITAS_NOASSERT_(x)
#endif

#endif // base_Assert_hh
