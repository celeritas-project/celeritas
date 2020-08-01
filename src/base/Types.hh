//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include "Array.hh"
#include "OpaqueId.hh"

namespace celeritas
{
template<typename T, std::size_t N>
struct array;
template<typename T, std::size_t N>
class span;

struct Thread;
//---------------------------------------------------------------------------//
using size_type    = std::size_t;
using ssize_type   = int;
using real_type    = double;
using RealPointer3 = array<real_type*, 3>;
using Real3        = array<real_type, 3>;
using SpanReal3    = span<real_type, 3>;

//! Index of the current CUDA thread, with type safety for containers.
using ThreadId = OpaqueId<Thread, unsigned int>;

//---------------------------------------------------------------------------//

enum class Interp
{
    Linear,
    Log
};

// SimulationStage definition imported from GeantX
enum class SimulationStage : short
{
    BeginStage,           // Actions at the beginning of the step
    ComputeIntLStage,     // Physics interaction length computation stage
    GeometryStepStage,    // Compute geometry transport length
    PrePropagationStage,  // Special msc stage for step limit phase
    // GeometryStepStage,     // Compute geometry transport length
    PropagationStage,      // Propagation in field stage
    PostPropagationStage,  // Special msc stage for along-step action stage
    // MSCStage,              // Multiple scattering stage
    AlongStepActionStage,  // Along step action stage (continuous part of the interaction)
    PostStepActionStage,   // Post step action stage (discrete part of the interaction)
    AtRestActionStage,     // At-rest action stage (at-rest part of the interaction)
    SteppingActionsStage   // User actions
};

} // namespace celeritas
