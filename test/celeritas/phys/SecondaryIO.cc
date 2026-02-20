//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/SecondaryIO.cc
//---------------------------------------------------------------------------//
#include "SecondaryIO.hh"

#include "celeritas/Quantities.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write a host-side Secondary to a stream for debugging.
 */
std::ostream& operator<<(std::ostream& os, Secondary const& s)
{
    os << "Secondary{";
    if (s)
    {
        os << "ParticleId{" << s.particle_id.unchecked_get() << "}, "
           << s.energy.value() << " * MeV, " << s.direction;
    }
    os << '}';
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
