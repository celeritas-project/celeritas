//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/detail/LengthUnits.hh
//! \brief NOTE: only use inside geocel; prefer celeritas/Units.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Types.hh"
#include "corecel/math/Constant.hh"

namespace celeritas
{
namespace lengthunits
{
//---------------------------------------------------------------------------//
#define CELER_ICC inline constexpr Constant

#if CELERITAS_UNITS == CELERITAS_UNITS_CGS
CELER_ICC meter{100};
CELER_ICC centimeter{1};
CELER_ICC millimeter{0.1};
inline constexpr char const native_label[] = "cm";
#elif CELERITAS_UNITS == CELERITAS_UNITS_SI
CELER_ICC meter{1};
CELER_ICC centimeter{0.01};
CELER_ICC millimeter{0.001};
inline constexpr char const native_label[] = "m";
#elif CELERITAS_UNITS == CELERITAS_UNITS_CLHEP
CELER_ICC meter{1000};
CELER_ICC centimeter{10};
CELER_ICC millimeter{1};
inline constexpr char const native_label[] = "mm";
#else
#    error "CELERITAS_UNITS is undefined"
#endif

#undef CELER_ICC
//---------------------------------------------------------------------------//
}  // namespace lengthunits
}  // namespace celeritas
