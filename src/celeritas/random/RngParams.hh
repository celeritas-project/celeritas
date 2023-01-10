//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/RngParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"

// IWYU pragma: begin_exports
#if (CELERITAS_RNG == CELERITAS_RNG_CURAND) \
    || (CELERITAS_RNG == CELERITAS_RNG_HIPRAND)
#    include "CuHipRngParams.hh"
#elif (CELERITAS_RNG == CELERITAS_RNG_XORWOW)
#    include "XorwowRngParams.hh"
#endif

#include "RngParamsFwd.hh"
// IWYU pragma: end_exports
