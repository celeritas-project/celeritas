//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InteractionIO.cc
//---------------------------------------------------------------------------//
#include "InteractionIO.hh"

#include "base/ArrayIO.hh"
#include "base/SpanIO.hh"
#include "physics/base/Units.hh"
#include "SecondaryIO.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write a host-side Interaction to a stream for debugging.
 */
std::ostream& operator<<(std::ostream& os, const Interaction& i)
{
    os << "Interaction{";
    if (i)
    {
        os << "Action{" << static_cast<int>(i.action) << "}, "
           << i.energy.value() << " MeV, " << i.direction << ", {"
           << i.secondaries << '}';
        if (i.energy_deposition > zero_quantity())
        {
            os << " + " << i.energy_deposition.value() << " MeV";
        }
    }
    os << '}';
    return os;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
