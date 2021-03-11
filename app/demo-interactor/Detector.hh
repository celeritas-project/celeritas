//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Detector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "DetectorInterface.hh"
#include "base/OpaqueId.hh"
#include "base/StackAllocator.hh"

namespace demo_interactor
{
//---------------------------------------------------------------------------//
/*!
 * Store a detector hit into the buffer.
 *
 * Input, processing, and resetting must all be done in separate kernels.
 */
class Detector
{
  public:
    //!@{
    //! Type aliases
    using Params        = DetectorParamsData;
    using State         = DetectorStateData<celeritas::Ownership::reference,
                                    celeritas::MemSpace::native>;
    using SpanConstHits = celeritas::Span<const Hit>;
    using HitId         = celeritas::OpaqueId<Hit>;
    //!@}

  public:
    // Construct from pointers
    inline CELER_FUNCTION Detector(const Params& params, const State& state);

    //// BUFFER INPUT ////

    // Record a hit
    inline CELER_FUNCTION void buffer_hit(const Hit& hit);

    //// BUFFER PROCESSING ////

    // View all hits (*not* for use in same kernel as record/clear)
    inline CELER_FUNCTION HitId::size_type num_hits() const;

    // Process a hit from the buffer to the grid
    inline CELER_FUNCTION void process_hit(HitId id);

    //// BUFFER CLEARING ////

    // Clear the buffer after processing all hits.
    inline CELER_FUNCTION void clear_buffer();

  private:
    const Params&                  params_;
    const State&                   state_;
    celeritas::StackAllocator<Hit> allocate_;
};

//---------------------------------------------------------------------------//
} // namespace demo_interactor

#include "Detector.i.hh"
