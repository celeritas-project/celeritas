//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/params/RngParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

// IWYU pragma: begin_exports
#if (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_CURAND) \
    || (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_HIPRAND)
#    include "CuHipRngParams.hh"
#elif (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW)
#    include "XorwowRngParams.hh"
#endif

#include "RngParamsFwd.hh"
// IWYU pragma: end_exports
