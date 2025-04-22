//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CartMapField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#if CELERITAS_USE_COVFIE || __DOXYGEN__
#    include "CartMapField.covfie.hh"
#else
#    include "detail/NotImplementedField.hh"
namespace celeritas
{
//---------------------------------------------------------------------------//
//! Dummy class for cartesian map magnetic field when no backend is available.
using CartMapField = detail::NotImplementedField;
}  // namespace celeritas
#endif  // CELERITAS_USE_COVFIE
