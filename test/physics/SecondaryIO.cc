//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SecondaryIO.cc
//---------------------------------------------------------------------------//
#include "SecondaryIO.hh"

#include "base/ArrayIO.hh"
#include "physics/base/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write a host-side Secondary to a stream for debugging.
 */
std::ostream& operator<<(std::ostream& os, const Secondary& s)
{
    os << "Secondary{";
    if (s)
    {
        os << "ParticleDefId{" << s.def_id.unchecked_get() << "}, "
           << s.energy / units::mega_electron_volt << " * MeV, "
           << s.direction;
    }
    os << '}';
    return os;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
