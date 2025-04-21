//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CartMapFieldData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Types.hh"

#include "FieldDriverOptions.hh"

#if CELERITAS_USE_COVFIE
#    include "detail/CartMapFieldData.covfie.hh"
#else

namespace celeritas
{
//! Real type for cartesian map field data
using cartmap_real_type = float;

template<Ownership W, MemSpace M>
struct CartMapFieldParamsData
{
    FieldDriverOptions options;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // CELERITAS_USE_COVFIE