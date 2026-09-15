//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/RZMapMagneticField.cc
//---------------------------------------------------------------------------//
#include "RZMapMagneticField.hh"

#include "celeritas/field/RZMapField.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// RZMAPMAGNETICFIELD IMPLEMENTATION
//---------------------------------------------------------------------------//

Real3 RZAdapterField::operator()(Real3 const& pos) const
{
    RZMapField calc_field{data};
    return calc_field(pos);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
