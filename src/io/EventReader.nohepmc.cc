//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EventReader.nohepmc.cc
//---------------------------------------------------------------------------//
#include "EventReader.hh"

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
EventReader::EventReader(const char*, SPConstParticles)
{
    CELER_NOT_CONFIGURED("HepMC3");
}

EventReader::~EventReader() = default;

EventReader::result_type EventReader::operator()()
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
