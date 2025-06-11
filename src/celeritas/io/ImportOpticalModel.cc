//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportOpticalModel.cc
//---------------------------------------------------------------------------//
#include "ImportOpticalModel.hh"

#include "corecel/io/EnumStringMapper.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//

char const* to_cstring(ImportModelClass imc)
{
    static EnumStringMapper<ImportModelClass> const to_cstring_impl{
        "absorption", "rayleigh", "wls", "wls2"};
    return to_cstring_impl(imc);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
