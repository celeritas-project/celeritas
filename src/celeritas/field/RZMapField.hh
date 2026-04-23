//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#if CELERITAS_USE_COVFIE || __DOXYGEN__
#    include "RZMapField.covfie.hh"
#else
#    include "RZMapFieldData.hh"

#    include "detail/NotImplementedField.hh"
namespace celeritas
{
//---------------------------------------------------------------------------//
//! Dummy class for R-Z map magnetic field when no backend is available.
using RZMapField = detail::NotImplementedField<RZMapFieldParamsData>;
//---------------------------------------------------------------------------//
}  // namespace celeritas
#endif  // CELERITAS_USE_COVFIE
