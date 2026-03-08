//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/InteractionIO.cc
//---------------------------------------------------------------------------//
#include "InteractionIO.hh"

#include "celeritas/Quantities.hh"

#include "SecondaryIO.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write a host-side Interaction to a stream for debugging.
 */
std::ostream& operator<<(std::ostream& os, Interaction const& i)
{
    os << "Interaction{";
    os << "Action{" << static_cast<int>(i.action) << "}, " << i.energy.value()
       << " MeV, " << i.direction << ", {" << i.secondaries << '}';
    if (i.energy_deposition > zero_quantity())
    {
        os << " + " << i.energy_deposition.value() << " MeV";
    }
    os << '}';
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
