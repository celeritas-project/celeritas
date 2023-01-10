//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/EventReader.nohepmc.cc
//---------------------------------------------------------------------------//
#include "EventReader.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
EventReader::EventReader(char const*, SPConstParticles)
{
    CELER_NOT_CONFIGURED("HepMC3");
}

EventReader::~EventReader() = default;

auto EventReader::operator()() -> result_type
{
    (void)sizeof(event_count_);
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
