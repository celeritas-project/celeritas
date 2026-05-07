//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/GeantOffloadUtils.hh
//---------------------------------------------------------------------------//
#pragma once

class G4Step;

namespace celeritas
{
namespace optical
{
struct GeneratorDistributionData;
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Populate a \c GeneratorDistributionData with \c G4Step data
optical::GeneratorDistributionData distribution_from_step(G4Step const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
