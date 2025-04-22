//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CartMapFieldData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#if CELERITAS_USE_COVFIE || __DOXYGEN__
#    include "CartMapFieldData.covfie.hh"
#else
#    include "corecel/Types.hh"

#    include "FieldDriverOptions.hh"
namespace celeritas
{
//---------------------------------------------------------------------------//
template<Ownership W, MemSpace M>
struct CartMapFieldParamsData
{
    FieldDriverOptions options;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // CELERITAS_USE_COVFIE
