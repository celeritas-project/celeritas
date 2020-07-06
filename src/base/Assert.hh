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
#ifndef __CUDA_ARCH__
#    include <cassert>
#endif

#define CELER_ASSERT_(COND) \
    do                      \
    {                       \
        assert(COND);       \
    } while (0)
#define CELER_NOASSERT_(COND)   \
    do                          \
    {                           \
        if (false && (COND)) {} \
    } while (0)

#ifdef CELERITAS_DEBUG
#    define REQUIRE(x) CELER_ASSERT_(x)
#    define CHECK(x) CELER_ASSERT_(x)
#    define ENSURE(x) CELER_ASSERT_(x)
#else
#    define REQUIRE(x) CELER_NOASSERT_(x)
#    define CHECK(x) CELER_NOASSERT_(x)
#    define ENSURE(x) CELER_NOASSERT_(x)
#endif

#endif // base_Assert_hh
