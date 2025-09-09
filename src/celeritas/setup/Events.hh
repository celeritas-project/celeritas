//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/Events.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "celeritas/inp/Events.hh"

namespace celeritas
{
struct Primary;
class ParticleParams;

namespace setup
{
//---------------------------------------------------------------------------//
// Load events
std::vector<std::vector<Primary>>
events(inp::Events const& inp,
       std::shared_ptr<ParticleParams const> const& particles);

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
