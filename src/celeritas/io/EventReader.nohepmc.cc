//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/EventReader.nohepmc.cc
//---------------------------------------------------------------------------//
#include "EventReader.hh"

#include "corecel/Assert.hh"
#include "celeritas/phys/Primary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
EventReader::EventReader(const char*, SPConstParticles)
{
    CELER_NOT_CONFIGURED("HepMC3");
}

EventReader::~EventReader() = default;

auto EventReader::operator()() -> VecPrimary
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
